#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks forward kinematics against MuJoCo's own solve of the same model.
//!
//! The fixture carries the model itself — one entry per joint saying what it does, what it hangs
//! off, where it sits, which way its axis points, where it turns about, and what reading counts as
//! not having moved — so the tree built here is the model MuJoCo was given, not a transcription of
//! it.

use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::spatial::{Quaternion, SE3, SO3};
use multicalc_qa::load::*;

/// Covers every fixture, the Franka's eleven joints included.
const MAX_JOINTS: usize = 16;

/// Row `index` of a matrix value, as a three-vector.
fn row3(data: &[f64], columns: usize, index: usize) -> Vector3D<f64> {
    let start = index * columns;
    Vector::new([data[start], data[start + 1], data[start + 2]])
}

#[test]
fn forward_kinematics_matches_mujoco() {
    for fx in load_dir("kinematics") {
        let case = fx.case.as_str();
        let tolerance = fx.tolerances.f64;

        let kinds: Vec<char> = fx.inputs["joint_kinds"].as_str().chars().collect();
        let parents = fx.inputs["parents"].as_vector();
        let zero_offsets = fx.inputs["zero_offsets"].as_vector();
        let (_, origin_position_columns, origin_positions) =
            fx.inputs["origin_positions"].as_matrix();
        let (_, origin_quaternion_columns, origin_quaternions) =
            fx.inputs["origin_quaternions"].as_matrix();
        let (_, axis_columns, axes) = fx.inputs["axes"].as_matrix();
        let (_, anchor_columns, anchors) = fx.inputs["anchors"].as_matrix();
        let (configuration_count, joint_count, configurations) =
            fx.inputs["joint_positions"].as_matrix();

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

        let tree = KinematicTree::<MAX_JOINTS, f64>::try_from_joints(&joints, &joint_parents)
            .unwrap_or_else(|e| unreachable!("{case}: building the tree: {e}"));

        let (_, position_columns, want_positions) = fx.expected["world_positions"].as_matrix();
        let (_, quaternion_columns, want_quaternions) =
            fx.expected["world_quaternions"].as_matrix();

        for configuration in 0..configuration_count {
            let start = configuration * joint_count;
            let readings = Vector::<MAX_JOINTS, f64>::from_fn(|index| {
                if index < joint_count {
                    configurations[start + index]
                } else {
                    0.0
                }
            });
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
