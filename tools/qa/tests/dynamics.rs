#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks a single rigid body's accelerations against MuJoCo's own solve of the same body.
//!
//! Each fixture states a body's mass properties, where it is pointing, how fast it is turning, and
//! what is pushing on it; the golden is what MuJoCo says that body does. The generator also works
//! every case out again in numpy before writing it, so a frame read the wrong way round fails at
//! generation time rather than here.
//!
//! The articulated cases check inverse dynamics, the joint-space inertia matrix and forward
//! dynamics against Pinocchio's solve of the same model, with MuJoCo's `mj_inverse` asserted
//! against Pinocchio at generation time. The fixture carries the model — joint kinds, parents,
//! origins, axes, anchors, masses, centres of mass, rotational inertias, armature, damping and
//! friction loss — so the model built here is the model the oracle was given.

use multicalc::dynamics::{ArticulatedBody, RigidBody};
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::spatial::{Quaternion, SE3, SO3, SpatialInertia, Wrench};
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

/// Covers every fixture, the Franka's eleven slots included.
const MAX_JOINTS: usize = 16;

#[test]
fn dynamics_goldens() {
    let fixtures = load_dir("dynamics");
    let mut checked = 0;
    for fixture in &fixtures {
        match fixture.case.as_str() {
            "free_body_spinning_no_torque" | "free_body_tilted_with_wrench" => {
                check_accelerations(fixture);
            }
            "articulated_double_pendulum" | "articulated_franka_panda" => {
                check_articulated(fixture);
            }
            other => panic!("no check registered for dynamics fixture {other}"),
        }
        checked += 1;
    }
    assert_eq!(
        checked, 4,
        "expected four dynamics fixtures, found {checked}"
    );
}

fn check_accelerations(fixture: &Fixture) {
    let mass = fixture.inputs["mass"].as_scalar();
    let center_of_mass = to_vector::<3>(&fixture.inputs["center_of_mass"]);
    let rotational_inertia = to_matrix::<3, 3>(&fixture.inputs["rotational_inertia"]);
    let gravity = to_vector::<3>(&fixture.inputs["gravity"]);
    let orientation = to_vector::<4>(&fixture.inputs["orientation"]);
    let angular_rate = to_vector::<3>(&fixture.inputs["angular_rate"]);
    let force = to_vector::<3>(&fixture.inputs["force"]);
    let torque = to_vector::<3>(&fixture.inputs["torque"]);
    let tolerance = fixture.tolerances.f64;

    let inertia = SpatialInertia::new(mass, center_of_mass, rotational_inertia).unwrap();
    let body = RigidBody::new(inertia, gravity).unwrap();
    let facing = SO3::from_quaternion(
        Quaternion::new(
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
        )
        .try_normalized()
        .unwrap(),
    );

    let acceleration = body.accelerations(facing, angular_rate, Wrench::new(force, torque));

    assert_vector(
        &acceleration.linear(),
        &fixture.expected["linear_acceleration"],
        tolerance,
        &format!("{}: straight-line", fixture.case),
    );
    assert_vector(
        &acceleration.angular(),
        &fixture.expected["angular_acceleration"],
        tolerance,
        &format!("{}: turning", fixture.case),
    );
}

/// Row `index` of a matrix value, as a three-vector.
fn row3(data: &[f64], columns: usize, index: usize) -> Vector3D<f64> {
    let start = index * columns;
    Vector::new([data[start], data[start + 1], data[start + 2]])
}

/// One row of a K x N matrix value, padded out to the model's width.
fn row_readings(data: &[f64], joint_count: usize, row: usize) -> Vector<MAX_JOINTS, f64> {
    let start = row * joint_count;
    Vector::from_fn(|index| {
        if index < joint_count {
            data[start + index]
        } else {
            0.0
        }
    })
}

/// The model the fixture carries, with the joint count it was written for.
///
/// `friction` chooses whether the joint-friction fields come across: with them zeroed the torque
/// is the oracle-pinned rigid-body term alone, which the fixture stores separately.
fn body_from_fixture(
    fixture: &Fixture,
    friction: bool,
) -> (ArticulatedBody<MAX_JOINTS, MAX_JOINTS, f64>, usize) {
    let case = fixture.case.as_str();

    let kinds: Vec<char> = fixture.inputs["joint_kinds"].as_str().chars().collect();
    let parents = fixture.inputs["parents"].as_vector();
    let zero_offsets = fixture.inputs["zero_offsets"].as_vector();
    let armatures = fixture.inputs["armatures"].as_vector();
    let dampings = fixture.inputs["dampings"].as_vector();
    let friction_losses = fixture.inputs["friction_losses"].as_vector();
    let masses = fixture.inputs["masses"].as_vector();
    let (_, origin_position_columns, origin_positions) =
        fixture.inputs["origin_positions"].as_matrix();
    let (_, origin_quaternion_columns, origin_quaternions) =
        fixture.inputs["origin_quaternions"].as_matrix();
    let (_, axis_columns, axes) = fixture.inputs["axes"].as_matrix();
    let (_, anchor_columns, anchors) = fixture.inputs["anchors"].as_matrix();
    let (_, center_columns, centers) = fixture.inputs["centers_of_mass"].as_matrix();
    let (_, inertia_columns, rotational_inertias) =
        fixture.inputs["rotational_inertias"].as_matrix();
    let (_, joint_count, _) = fixture.inputs["joint_positions"].as_matrix();

    assert_eq!(joint_count, kinds.len(), "{case}: joint count");
    assert!(
        joint_count <= MAX_JOINTS,
        "{case}: {joint_count} joints exceeds the {MAX_JOINTS} this test builds for"
    );

    let mut joints = Vec::with_capacity(joint_count);
    let mut joint_parents = Vec::with_capacity(joint_count);
    let mut inertias = Vec::with_capacity(joint_count);
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
        let (damping, friction_loss) = if friction {
            (dampings[index], friction_losses[index])
        } else {
            (0.0, 0.0)
        };

        let joint = match kinds[index] {
            'R' => Joint::revolute(axis, origin).with_anchor(anchor),
            'P' => Joint::prismatic(axis, origin),
            'F' => Joint::fixed(origin),
            other => unreachable!("{case}: joint {index} has kind {other:?}"),
        };
        joints.push(
            joint
                .with_zero_offset(zero_offsets[index])
                .with_armature(armatures[index])
                .with_damping(damping)
                .with_friction_loss(friction_loss),
        );
        joint_parents.push(match parents[index] {
            parent if parent < 0.0 => JointParent::World,
            parent => JointParent::Joint(parent as usize),
        });

        // A slot the model gives no mass carries none, which is what a weld tool frame looks like.
        inertias.push(if masses[index] == 0.0 {
            None
        } else {
            let start = index * inertia_columns;
            let rotational =
                Matrix::from_fn(|row, column| rotational_inertias[start + row * 3 + column]);
            Some(
                SpatialInertia::new(
                    masses[index],
                    row3(&centers, center_columns, index),
                    rotational,
                )
                .unwrap_or_else(|err| unreachable!("{case}: slot {index} inertia: {err:?}")),
            )
        });
    }

    let tree =
        KinematicTree::<MAX_JOINTS, MAX_JOINTS, f64>::try_from_joints(&joints, &joint_parents)
            .unwrap_or_else(|err| unreachable!("{case}: building the tree: {err}"));
    let body = ArticulatedBody::new(tree, &inertias, Vector::new([0.0, 0.0, -9.81]))
        .unwrap_or_else(|err| unreachable!("{case}: building the body: {err:?}"));
    (body, joint_count)
}

fn check_articulated(fixture: &Fixture) {
    let case = fixture.case.as_str();
    let tolerance = fixture.tolerances.f64;
    let (body, joint_count) = body_from_fixture(fixture, true);
    let (rigid_body, _) = body_from_fixture(fixture, false);

    let (state_count, _, positions) = fixture.inputs["joint_positions"].as_matrix();
    let (_, _, velocities) = fixture.inputs["joint_velocities"].as_matrix();
    let (_, _, accelerations) = fixture.inputs["joint_accelerations"].as_matrix();

    let (_, _, want_torques) = fixture.expected["torques"].as_matrix();
    let (_, _, want_rigid_body_torques) = fixture.expected["rigid_body_torques"].as_matrix();
    let (_, _, want_inertias) = fixture.expected["joint_space_inertias"].as_matrix();
    let (_, _, want_accelerations) = fixture.expected["forward_accelerations"].as_matrix();

    for state in 0..state_count {
        let context = format!("{case} state {state}");
        let position = row_readings(&positions, joint_count, state);
        let velocity = row_readings(&velocities, joint_count, state);
        let acceleration = row_readings(&accelerations, joint_count, state);

        let torque = body
            .inverse_dynamics_at(&position, &velocity, &acceleration)
            .unwrap_or_else(|err| unreachable!("{context}: inverse dynamics: {err:?}"));
        let rigid_torque = rigid_body
            .inverse_dynamics_at(&position, &velocity, &acceleration)
            .unwrap_or_else(|err| unreachable!("{context}: rigid-body torque: {err:?}"));
        let inertia = body
            .joint_space_inertia_at(&position)
            .unwrap_or_else(|err| unreachable!("{context}: joint-space inertia: {err:?}"));

        let start = state * joint_count;
        for index in 0..joint_count {
            assert!(
                close(torque[index], want_torques[start + index], tolerance),
                "{context}: torque[{index}] got {}, want {}",
                torque[index],
                want_torques[start + index]
            );
            // The oracle-pinned rigid-body term on its own, so a friction bug and an RNEA bug
            // never look alike.
            assert!(
                close(
                    rigid_torque[index],
                    want_rigid_body_torques[start + index],
                    tolerance
                ),
                "{context}: rigid-body torque[{index}] got {}, want {}",
                rigid_torque[index],
                want_rigid_body_torques[start + index]
            );
        }

        let block = state * joint_count * joint_count;
        for row in 0..joint_count {
            for column in 0..joint_count {
                let want = want_inertias[block + row * joint_count + column];
                assert!(
                    close(inertia[(row, column)], want, tolerance),
                    "{context}: H[({row}, {column})] got {}, want {want}",
                    inertia[(row, column)]
                );
                assert!(
                    (inertia[(row, column)] - inertia[(column, row)]).abs() < 1e-12,
                    "{context}: H is not symmetric at ({row}, {column})"
                );
            }
        }

        // Pinocchio's `aba` carries armature but no joint friction, so its golden belongs to the
        // friction-free model fed the rigid-body torque.
        let rigid_applied = row_readings(&want_rigid_body_torques, joint_count, state);
        let recovered = rigid_body
            .forward_dynamics_at(&position, &velocity, &rigid_applied)
            .unwrap_or_else(|err| unreachable!("{context}: forward dynamics: {err:?}"));

        // The full torque through the full model returns the sampled acceleration, which is the
        // same round trip with damping and Coulomb friction in the loop on both sides.
        let full_applied = row_readings(&want_torques, joint_count, state);
        let with_friction = body
            .forward_dynamics_at(&position, &velocity, &full_applied)
            .unwrap_or_else(|err| unreachable!("{context}: forward dynamics: {err:?}"));

        for index in 0..joint_count {
            assert!(
                close(
                    recovered[index],
                    want_accelerations[start + index],
                    tolerance
                ),
                "{context}: acceleration[{index}] got {}, want {}",
                recovered[index],
                want_accelerations[start + index]
            );
            assert!(
                close(
                    with_friction[index],
                    accelerations[start + index],
                    tolerance
                ),
                "{context}: acceleration[{index}] got {}, did not return the sampled {}",
                with_friction[index],
                accelerations[start + index]
            );
        }
    }
}
