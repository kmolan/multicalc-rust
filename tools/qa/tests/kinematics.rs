#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks forward kinematics and geometric Jacobians against MuJoCo's own solve of the same model.
//!
//! The fixture carries the model itself — one entry per joint saying what it does, what it hangs
//! off, where it sits, which way its axis points, where it turns about, and what reading counts as
//! not having moved — so the tree built here is the model MuJoCo was given, not a transcription of
//! it.

use std::path::Path;

use multicalc::kinematics::{JacobianFrame, Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::spatial::{Quaternion, SE3, SO3};
use multicalc_qa::load::*;
use multicalc_qa::schema::Fixture;

/// The vendored models a fixture's `model_file` field names, relative to this crate.
#[must_use]
fn menagerie() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../third_party/menagerie")
}

/// Covers every fixture, the Franka's eleven joints included.
const MAX_JOINTS: usize = 16;

/// Row `index` of a matrix value, as a three-vector.
fn row3(data: &[f64], columns: usize, index: usize) -> Vector3D<f64> {
    let start = index * columns;
    Vector::new([data[start], data[start + 1], data[start + 2]])
}

/// The model the fixture carries, built as a tree, with the joint count it was written for.
fn tree_from_fixture(fx: &Fixture) -> (KinematicTree<MAX_JOINTS, MAX_JOINTS, f64>, usize) {
    let case = fx.case.as_str();

    let kinds: Vec<char> = fx.inputs["joint_kinds"].as_str().chars().collect();
    let parents = fx.inputs["parents"].as_vector();
    let zero_offsets = fx.inputs["zero_offsets"].as_vector();
    let (_, origin_position_columns, origin_positions) = fx.inputs["origin_positions"].as_matrix();
    let (_, origin_quaternion_columns, origin_quaternions) =
        fx.inputs["origin_quaternions"].as_matrix();
    let (_, axis_columns, axes) = fx.inputs["axes"].as_matrix();
    let (_, anchor_columns, anchors) = fx.inputs["anchors"].as_matrix();
    let (_, joint_count, _) = fx.inputs["joint_positions"].as_matrix();

    assert_eq!(joint_count, kinds.len(), "{case}: joint count");
    assert!(
        joint_count <= MAX_JOINTS,
        "{case}: {joint_count} joints exceeds the {MAX_JOINTS} this test builds for"
    );

    // One joint per fixture entry, and one parent: -1 is the world, anything else an earlier
    // joint by position.
    let mut joints = Vec::with_capacity(joint_count);
    let mut joint_parents = Vec::with_capacity(joint_count);
    for index in 0..joint_count {
        let start = index * origin_quaternion_columns;
        let orientation = Quaternion::new(
            origin_quaternions[start],
            origin_quaternions[start + 1],
            origin_quaternions[start + 2],
            origin_quaternions[start + 3],
        )
        .try_normalized()
        .unwrap_or_else(|| unreachable!("{case}: joint {index} has a zero origin quaternion"));
        let origin = SE3::from_parts(
            SO3::from_quaternion(orientation),
            row3(&origin_positions, origin_position_columns, index),
        );
        let axis = row3(&axes, axis_columns, index);
        let anchor = row3(&anchors, anchor_columns, index);
        let zero_offset = zero_offsets[index];

        joints.push(match kinds[index] {
            'R' => Joint::revolute(axis, origin)
                .with_anchor(anchor)
                .with_zero_offset(zero_offset),
            'P' => Joint::prismatic(axis, origin).with_zero_offset(zero_offset),
            'F' => Joint::fixed(origin),
            other => unreachable!("{case}: joint {index} has kind {other:?}"),
        });
        joint_parents.push(match parents[index] {
            p if p < 0.0 => JointParent::World,
            p => JointParent::Joint(p as usize),
        });
    }

    let tree = KinematicTree::<MAX_JOINTS, MAX_JOINTS, f64>::try_from_joints(&joints, &joint_parents)
        .unwrap_or_else(|e| unreachable!("{case}: building the tree: {e}"));
    (tree, joint_count)
}

/// The readings for one configuration row, padded out to the tree's width.
fn readings_at(
    configurations: &[f64],
    joint_count: usize,
    configuration: usize,
) -> Vector<MAX_JOINTS, f64> {
    let start = configuration * joint_count;
    Vector::from_fn(|index| {
        if index < joint_count {
            configurations[start + index]
        } else {
            0.0
        }
    })
}

#[test]
fn forward_kinematics_matches_mujoco() {
    for fx in load_dir("kinematics") {
        // A bespoke fixture whose configuration is not one scalar per joint slot (a floating
        // base's is seven-wide) — checked by its own test below instead of this shared loop.
        if fx.case == "unitree_go1_floating_base" {
            continue;
        }
        let case = fx.case.as_str();
        let tolerance = fx.tolerances.f64;
        let (tree, joint_count) = tree_from_fixture(&fx);
        let (configuration_count, _, configurations) = fx.inputs["joint_positions"].as_matrix();

        let (_, position_columns, want_positions) = fx.expected["world_positions"].as_matrix();
        let (_, quaternion_columns, want_quaternions) =
            fx.expected["world_quaternions"].as_matrix();

        for configuration in 0..configuration_count {
            let start = configuration * joint_count;
            let readings = readings_at(&configurations, joint_count, configuration);
            let state = tree
                .forward_kinematics(&readings)
                .unwrap_or_else(|e| unreachable!("{case}: configuration {configuration}: {e}"));

            assert_eq!(state.len(), joint_count, "{case}: settled joint count");
            assert!(
                state.pose(joint_count).is_none(),
                "{case}: a pose was reported past the model's joints"
            );

            for index in 0..joint_count {
                let pose = state.pose(index).unwrap_or_else(|| {
                    unreachable!("{case}: configuration {configuration}: joint {index} unsettled")
                });
                let want_row = start + index;

                let got = *pose.translation().as_array();
                let want = row3(&want_positions, position_columns, want_row);
                for axis in 0..3 {
                    assert!(
                        close(got[axis], want[axis], tolerance),
                        "{case}: configuration {configuration}, joint {index}, \
                         position component {axis}: got {}, want {}",
                        got[axis],
                        want[axis]
                    );
                }

                // A quaternion and its negative name the same turn, so both sides are compared in
                // their scalar-positive form rather than component by component as written.
                let got = pose.rotation().quaternion().as_array();
                let want_start = want_row * quaternion_columns;
                let flip = if got[0] < 0.0 { -1.0 } else { 1.0 };
                let want_flip = if want_quaternions[want_start] < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                for component in 0..4 {
                    assert!(
                        close(
                            flip * got[component],
                            want_flip * want_quaternions[want_start + component],
                            tolerance
                        ),
                        "{case}: configuration {configuration}, joint {index}, \
                         quaternion component {component}: got {}, want {}",
                        flip * got[component],
                        want_flip * want_quaternions[want_start + component]
                    );
                }
            }
        }
    }
}

#[test]
fn geometric_jacobian_matches_mujoco() {
    for fx in load_dir("kinematics") {
        // A bespoke fixture whose configuration is not one scalar per joint slot (a floating
        // base's is seven-wide) — checked by its own test below instead of this shared loop.
        if fx.case == "unitree_go1_floating_base" {
            continue;
        }
        let case = fx.case.as_str();
        let tolerance = fx.tolerances.f64;
        let (tree, joint_count) = tree_from_fixture(&fx);
        let (configuration_count, _, configurations) = fx.inputs["joint_positions"].as_matrix();

        // Six rows per body, bodies in tree order, one such block per configuration.
        let (jacobian_rows, jacobian_columns, want_jacobians) = fx
            .expected
            .get("world_jacobians")
            .unwrap_or_else(|| {
                unreachable!("{case}: fixture carries no Jacobian goldens; regenerate it")
            })
            .as_matrix();
        assert_eq!(
            jacobian_columns, joint_count,
            "{case}: Jacobian column count"
        );
        assert_eq!(
            jacobian_rows,
            configuration_count * joint_count * 6,
            "{case}: Jacobian row count"
        );

        for configuration in 0..configuration_count {
            let readings = readings_at(&configurations, joint_count, configuration);
            let state = tree
                .forward_kinematics(&readings)
                .unwrap_or_else(|e| unreachable!("{case}: configuration {configuration}: {e}"));

            for tool_index in 0..joint_count {
                let jacobian = tree
                    .geometric_jacobian(&state, tool_index, JacobianFrame::World)
                    .unwrap_or_else(|e| {
                        unreachable!(
                            "{case}: configuration {configuration}, body {tool_index}: {e}"
                        )
                    });
                let got = jacobian.matrix();
                let block = ((configuration * joint_count) + tool_index) * 6;

                for row in 0..6 {
                    for column in 0..joint_count {
                        let got = *got.get(row, column).unwrap_or_else(|| {
                            unreachable!("{case}: row {row}, column {column} is outside the block")
                        });
                        let want = want_jacobians[(block + row) * jacobian_columns + column];
                        assert!(
                            close(got, want, tolerance),
                            "{case}: configuration {configuration}, body {tool_index}, \
                             row {row}, column {column}: got {got}, want {want}"
                        );
                    }
                }
            }
        }
    }
}

/// The vendored Unitree Go1: a floating base carrying twelve hinge joints, whose forward
/// kinematics and Jacobian this checks against MuJoCo's own solve of the same file.
///
/// Self-contained rather than routed through `tree_from_fixture`/`readings_at` above: those
/// assume one scalar reading per joint slot everywhere, which a floating base's seven-wide
/// configuration and six-wide velocity break. The tree here is built by parsing the vendored file
/// itself (so this also exercises real-file parsing, not just the math), and each fixture row is
/// MuJoCo's own 19-wide `qpos` — exactly the layout `KinematicTree::config_offset` produces for a
/// tree whose first joint is floating, so it reads straight into the configuration vector with no
/// reindexing.
#[test]
fn unitree_go1_forward_kinematics_and_jacobian_matches_mujoco() {
    let fx = load_dir("kinematics")
        .into_iter()
        .find(|fx| fx.case == "unitree_go1_floating_base")
        .unwrap_or_else(|| unreachable!("fixture unitree_go1_floating_base is missing"));
    let case = fx.case.as_str();
    let tolerance = fx.tolerances.f64;

    let path = menagerie().join(fx.inputs["model_file"].as_str());
    let model = multicalc_mjcf::load_path(&path)
        .unwrap_or_else(|e| unreachable!("{case}: loading {path:?}: {e}"));
    let tree = model
        .kinematic_tree::<13, 19>()
        .unwrap_or_else(|e| unreachable!("{case}: building the tree: {e}"));
    let body_count = tree.len();

    let (configuration_count, configuration_columns, configurations) =
        fx.inputs["configurations"].as_matrix();
    assert_eq!(configuration_columns, 19, "{case}: configuration width");

    let (_, position_columns, want_positions) = fx.expected["world_positions"].as_matrix();
    let (_, quaternion_columns, want_quaternions) = fx.expected["world_quaternions"].as_matrix();
    let (jacobian_rows, jacobian_columns, want_jacobians) =
        fx.expected["world_jacobians"].as_matrix();
    assert_eq!(jacobian_columns, 18, "{case}: Jacobian column count");
    assert_eq!(
        jacobian_rows,
        configuration_count * body_count * 6,
        "{case}: Jacobian row count"
    );

    for configuration in 0..configuration_count {
        let start = configuration * configuration_columns;
        let reading = Vector::<19, f64>::from_fn(|index| configurations[start + index]);
        let state = tree
            .forward_kinematics(&reading)
            .unwrap_or_else(|e| unreachable!("{case}: configuration {configuration}: {e}"));

        for index in 0..body_count {
            let pose = state.pose(index).unwrap_or_else(|| {
                unreachable!("{case}: configuration {configuration}: body {index} unsettled")
            });
            let want_row = configuration * body_count + index;

            let got = *pose.translation().as_array();
            let want = row3(&want_positions, position_columns, want_row);
            for axis in 0..3 {
                assert!(
                    close(got[axis], want[axis], tolerance),
                    "{case}: configuration {configuration}, body {index}, \
                     position component {axis}: got {}, want {}",
                    got[axis],
                    want[axis]
                );
            }

            // A quaternion and its negative name the same turn, so both sides are compared in
            // their scalar-positive form rather than component by component as written.
            let got_q = pose.rotation().quaternion().as_array();
            let want_start = want_row * quaternion_columns;
            let flip = if got_q[0] < 0.0 { -1.0 } else { 1.0 };
            let want_flip = if want_quaternions[want_start] < 0.0 {
                -1.0
            } else {
                1.0
            };
            for component in 0..4 {
                assert!(
                    close(
                        flip * got_q[component],
                        want_flip * want_quaternions[want_start + component],
                        tolerance
                    ),
                    "{case}: configuration {configuration}, body {index}, \
                     quaternion component {component}: got {}, want {}",
                    flip * got_q[component],
                    want_flip * want_quaternions[want_start + component]
                );
            }

            let jacobian = tree
                .geometric_jacobian(&state, index, JacobianFrame::World)
                .unwrap_or_else(|e| {
                    unreachable!("{case}: configuration {configuration}, body {index}: {e}")
                });
            let got_j = jacobian.matrix();
            let block = want_row * 6;
            for row in 0..6 {
                for column in 0..jacobian_columns {
                    let got = *got_j.get(row, column).unwrap_or_else(|| {
                        unreachable!("{case}: row {row}, column {column} out of range")
                    });
                    let want = want_jacobians[(block + row) * jacobian_columns + column];
                    assert!(
                        close(got, want, tolerance),
                        "{case}: configuration {configuration}, body {index}, \
                         row {row}, column {column}: got {got}, want {want}"
                    );
                }
            }
        }
    }
}
