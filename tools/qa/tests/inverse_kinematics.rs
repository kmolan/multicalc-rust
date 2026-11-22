#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the inverse-kinematics solver against mink's differential IK on the same MuJoCo model.
//!
//! Every case pins the pose reached and whether the solve converged. Joint angles are pinned only
//! where the fixture carries them, which is the one case that is not redundant and is seeded next
//! to its solution — anywhere else the two solvers are free to settle on different configurations
//! that put the frame in the same place, and pinning one of them would be pinning an accident.

use std::path::Path;

use multicalc::kinematics::{
    InverseKinematics, InverseKinematicsTermination, Joint, JointParent, KinematicTree,
};
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

/// How closely the two solvers' joint readings have to agree, where the fixture pins them.
///
/// Looser than the pose bound because a pose residual maps onto a larger joint error through the
/// Jacobian, and because the two solvers take different routes to the same answer. Both are driven
/// to a 1e-9 residual here, which lands them within about 7e-9 of each other; this leaves a wide
/// margin on that while still being far tighter than the difference a *different branch* would
/// show, which is radians.
const JOINT_TOLERANCE: f64 = 1e-7;

/// Row `index` of a matrix value, as a three-vector.
fn row3(data: &[f64], columns: usize, index: usize) -> Vector3D<f64> {
    let start = index * columns;
    Vector::new([data[start], data[start + 1], data[start + 2]])
}

/// The model the fixture carries, built as a tree, with the joint count it was written for.
fn tree_from_fixture(fixture: &Fixture) -> (KinematicTree<MAX_JOINTS, MAX_JOINTS, f64>, usize) {
    let case = fixture.case.as_str();

    let kinds: Vec<char> = fixture.inputs["joint_kinds"].as_str().chars().collect();
    let parents = fixture.inputs["parents"].as_vector();
    let zero_offsets = fixture.inputs["zero_offsets"].as_vector();
    let (_, origin_position_columns, origin_positions) =
        fixture.inputs["origin_positions"].as_matrix();
    let (_, origin_quaternion_columns, origin_quaternions) =
        fixture.inputs["origin_quaternions"].as_matrix();
    let (_, axis_columns, axes) = fixture.inputs["axes"].as_matrix();
    let (_, anchor_columns, anchors) = fixture.inputs["anchors"].as_matrix();
    let joint_count = kinds.len();

    assert!(
        joint_count <= MAX_JOINTS,
        "{case}: {joint_count} joints exceeds the {MAX_JOINTS} this test builds for"
    );

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
            parent if parent < 0.0 => JointParent::World,
            parent => JointParent::Joint(parent as usize),
        });
    }

    let tree =
        KinematicTree::<MAX_JOINTS, MAX_JOINTS, f64>::try_from_joints(&joints, &joint_parents)
            .unwrap_or_else(|err| unreachable!("{case}: building the tree: {err}"));
    (tree, joint_count)
}

/// A fixture vector padded out to the tree's width.
fn padded(values: &[f64]) -> Vector<MAX_JOINTS, f64> {
    Vector::from_fn(|index| values.get(index).copied().unwrap_or(0.0))
}

/// A position and a scalar-first quaternion from the fixture, as a pose.
fn pose_from(position: &[f64], quaternion: &[f64], what: &str) -> SE3<f64> {
    let orientation = Quaternion::new(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        .try_normalized()
        .unwrap_or_else(|| unreachable!("{what}: zero quaternion"));
    SE3::from_parts(
        SO3::from_quaternion(orientation),
        Vector::new([position[0], position[1], position[2]]),
    )
}

#[test]
fn inverse_kinematics_matches_mink() {
    for fixture in load_dir("inverse_kinematics") {
        // A bespoke fixture whose configuration is not one scalar per joint slot (a floating
        // base's is seven-wide) — checked by its own test below instead of this shared loop.
        if fixture.case == "unitree_go1_floating_base_ik" {
            continue;
        }
        let case = fixture.case.as_str();
        let tolerance = fixture.tolerances.f64;
        let (tree, joint_count) = tree_from_fixture(&fixture);

        let tool_index = fixture.inputs["tool_index"].as_int() as usize;
        let target = pose_from(
            &fixture.inputs["target_position"].as_vector(),
            &fixture.inputs["target_quaternion"].as_vector(),
            case,
        );
        let seed = padded(&fixture.inputs["seed"].as_vector());

        // Driven to the same residual the golden was, so a configuration comparison reflects the
        // two solvers disagreeing rather than either one stopping early.
        let report = InverseKinematics::<MAX_JOINTS, f64>::new()
            .with_position_tolerance(1e-9)
            .with_orientation_tolerance(1e-9)
            .solve(&tree, tool_index, target, &seed)
            .unwrap_or_else(|err| unreachable!("{case}: {err}"));

        // mink only ever writes a fixture for a solve it converged, so the crate's solver has to
        // converge too.
        assert_eq!(
            fixture.expected["converged"].as_int(),
            1,
            "{case}: fixture holds a solve that did not converge"
        );
        assert_eq!(
            report.termination,
            InverseKinematicsTermination::Converged,
            "{case}: terminated {:?} with position error {}",
            report.termination,
            report.position_error
        );

        // The pose reached is the property both solvers must agree on, whichever configuration
        // each of them picked to get there.
        let reached = pose_from(
            &fixture.expected["reached_position"].as_vector(),
            &fixture.expected["reached_quaternion"].as_vector(),
            case,
        );
        let solved = tree
            .forward_kinematics(&report.joint_positions)
            .unwrap_or_else(|err| unreachable!("{case}: resolving the answer: {err}"))
            .pose(tool_index)
            .unwrap_or_else(|| unreachable!("{case}: no pose at the tool index"));

        let got = *solved.translation().as_array();
        let want = *reached.translation().as_array();
        for axis in 0..3 {
            assert!(
                close(got[axis], want[axis], tolerance),
                "{case}: reached position component {axis}: got {}, want {}",
                got[axis],
                want[axis]
            );
        }

        // Compared as a turn between the two, since a quaternion and its negative name the same
        // orientation.
        let turn = (reached.rotation().inverse() * solved.rotation())
            .log()
            .norm();
        assert!(
            close(turn, 0.0, tolerance),
            "{case}: reached orientation is {turn} rad from the golden"
        );

        // Only the non-redundant, closely-seeded case carries a configuration to compare.
        if let Some(want_readings) = fixture.expected.get("joint_positions") {
            let want_readings = want_readings.as_vector();
            for (index, want) in want_readings.iter().take(joint_count).enumerate() {
                let got = report.joint_positions.get(index).copied().unwrap_or(0.0);
                let want = *want;
                assert!(
                    (got - want).abs() <= JOINT_TOLERANCE,
                    "{case}: joint {index}: got {got}, want {want}"
                );
            }
        }
    }
}

/// The vendored Unitree Go1: eighteen degrees of freedom (the trunk's floating base plus twelve
/// hinges) solving to place one foot, checked against mink's differential IK on the same file.
///
/// Self-contained rather than routed through `tree_from_fixture`/`padded` above, for the same
/// reason as the kinematics test: those assume one scalar reading per joint slot everywhere,
/// which the floating base does not fit. The tree here is built by parsing the vendored file
/// itself, and only the reached pose is compared — the twelve spare degrees of freedom are
/// exactly the case the fixture was written for (see `_unitree_go1_floating_base_ik`'s docstring),
/// so there is no configuration golden to pin.
#[test]
fn unitree_go1_inverse_kinematics_matches_mink() {
    let fixture = load_dir("inverse_kinematics")
        .into_iter()
        .find(|fixture| fixture.case == "unitree_go1_floating_base_ik")
        .unwrap_or_else(|| unreachable!("fixture unitree_go1_floating_base_ik is missing"));
    let case = fixture.case.as_str();
    let tolerance = fixture.tolerances.f64;

    let path = menagerie().join(fixture.inputs["model_file"].as_str());
    let model = multicalc_robot_model::mjcf::load_path(&path)
        .unwrap_or_else(|err| unreachable!("{case}: loading {path:?}: {err}"));
    let tree = model
        .kinematic_tree::<13, 19>()
        .unwrap_or_else(|err| unreachable!("{case}: building the tree: {err}"));
    let tool_index = model
        .bodies()
        .iter()
        .position(|body| body.name() == "FR_calf")
        .unwrap_or_else(|| unreachable!("{case}: model has no body called FR_calf"));

    let seed_values = fixture.inputs["seed"].as_vector();
    let seed = Vector::<19, f64>::from_fn(|index| seed_values[index]);
    let target = pose_from(
        &fixture.inputs["target_position"].as_vector(),
        &fixture.inputs["target_quaternion"].as_vector(),
        case,
    );

    // Driven to the same residual the golden was, so a pose comparison reflects the two solvers
    // disagreeing rather than either one stopping early.
    let report = InverseKinematics::<19, f64>::new()
        .with_position_tolerance(1e-9)
        .with_orientation_tolerance(1e-9)
        .solve(&tree, tool_index, target, &seed)
        .unwrap_or_else(|err| unreachable!("{case}: {err}"));

    assert_eq!(
        fixture.expected["converged"].as_int(),
        1,
        "{case}: fixture holds a solve that did not converge"
    );
    assert_eq!(
        report.termination,
        InverseKinematicsTermination::Converged,
        "{case}: terminated {:?} with position error {}",
        report.termination,
        report.position_error
    );

    let reached = pose_from(
        &fixture.expected["reached_position"].as_vector(),
        &fixture.expected["reached_quaternion"].as_vector(),
        case,
    );
    let solved = tree
        .forward_kinematics(&report.joint_positions)
        .unwrap_or_else(|err| unreachable!("{case}: resolving the answer: {err}"))
        .pose(tool_index)
        .unwrap_or_else(|| unreachable!("{case}: no pose at the tool index"));

    let got = *solved.translation().as_array();
    let want = *reached.translation().as_array();
    for axis in 0..3 {
        assert!(
            close(got[axis], want[axis], tolerance),
            "{case}: reached position component {axis}: got {}, want {}",
            got[axis],
            want[axis]
        );
    }

    let turn = (reached.rotation().inverse() * solved.rotation())
        .log()
        .norm();
    assert!(
        close(turn, 0.0, tolerance),
        "{case}: reached orientation is {turn} rad from the golden"
    );
}
