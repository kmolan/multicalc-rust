#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks a single rigid body's accelerations against MuJoCo's own solve of the same body.
//!
//! Each fixture states a body's mass properties, orientation, angular velocity and applied wrench;
//! the golden is MuJoCo's acceleration. The generator re-derives every case in numpy before writing
//! it, so a frame read the wrong way round fails at generation time rather than here.
//!
//! The articulated cases check inverse dynamics, the joint-space inertia matrix and forward
//! dynamics against Pinocchio's solve of the same model, with MuJoCo's `mj_inverse` asserted
//! against Pinocchio at generation time. The fixture carries the model — joint kinds, parents,
//! origins, axes, anchors, masses, centres of mass, rotational inertias, armature, damping and
//! friction loss — so the model built here is the model the oracle was given.

use multicalc::dynamics::RigidBody;
use multicalc::spatial::{Quaternion, SO3, SpatialInertia, Wrench};
use multicalc_qa::articulated::*;
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

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
