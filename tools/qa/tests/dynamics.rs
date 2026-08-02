#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks a single rigid body's accelerations against MuJoCo's own solve of the same body.
//!
//! Each fixture states a body's mass properties, where it is pointing, how fast it is turning, and
//! what is pushing on it; the golden is what MuJoCo says that body does. The generator also works
//! every case out again in numpy before writing it, so a frame read the wrong way round fails at
//! generation time rather than here.

use multicalc::dynamics::RigidBody;
use multicalc::spatial::{Quaternion, SO3, SpatialInertia, Wrench};
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
            other => panic!("no check registered for dynamics fixture {other}"),
        }
        checked += 1;
    }
    assert_eq!(
        checked, 2,
        "expected two dynamics fixtures, found {checked}"
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
