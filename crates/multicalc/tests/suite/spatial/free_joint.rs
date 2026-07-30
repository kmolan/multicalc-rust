//! Free-joint state tests: the number layout, place round trips, and nudging a state.

use multicalc::linear_algebra::Vector;
use multicalc::spatial::{FreeJointState, Quaternion, SE3, SO3, Twist};

/// Quaternions come in pairs that name the same direction, so compare the one with a positive
/// leading number on both sides.
fn leading_positive(place: [f64; 7]) -> [f64; 7] {
    if place[3] < 0.0 {
        let [x, y, z, w, qx, qy, qz] = place;
        [x, y, z, -w, -qx, -qy, -qz]
    } else {
        place
    }
}

#[test]
fn state_takes_seven_numbers_to_place_and_six_to_move() {
    assert_eq!(FreeJointState::<f64>::GENERALIZED_POSITION_DIMENSION, 7);
    assert_eq!(FreeJointState::<f64>::GENERALIZED_VELOCITY_DIMENSION, 6);
    assert_eq!(
        FreeJointState::<f64>::identity()
            .generalized_position()
            .len(),
        FreeJointState::<f64>::GENERALIZED_POSITION_DIMENSION
    );
    assert_eq!(
        FreeJointState::<f64>::identity()
            .generalized_velocity()
            .len(),
        FreeJointState::<f64>::GENERALIZED_VELOCITY_DIMENSION
    );
}

#[test]
fn identity_sits_at_the_origin_and_does_not_move() {
    let state = FreeJointState::<f64>::identity();
    assert_eq!(
        state.generalized_position(),
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(state.velocity(), Twist::zeros());
}

#[test]
fn generalized_velocity_is_the_twist_written_out() {
    let velocity = Twist::from_array([0.1, 0.2, 0.3, -0.4, 0.5, -0.6]);
    let state = FreeJointState::new(SE3::<f64>::identity(), velocity);
    // Same six numbers, same [v; ω] order.
    assert_eq!(state.generalized_velocity(), velocity.as_array());
}

#[test]
fn generalized_position_is_position_then_orientation() {
    let orientation = Quaternion::from_axis_angle(Vector::new([0.0, 0.0, 1.0]), 0.6);
    let pose = SE3::from_parts(
        SO3::from_quaternion(orientation),
        Vector::new([1.0, 2.0, 3.0]),
    );
    let state = FreeJointState::new(pose, Twist::zeros());

    let place = state.generalized_position();
    assert_eq!([place[0], place[1], place[2]], [1.0, 2.0, 3.0]);

    let stored = leading_positive(place);
    let want = leading_positive([
        1.0,
        2.0,
        3.0,
        orientation.w(),
        orientation.x(),
        orientation.y(),
        orientation.z(),
    ]);
    for index in 3..7 {
        assert!(
            (stored[index] - want[index]).abs() < 1e-12,
            "number {index}"
        );
    }
}

#[test]
fn generalized_numbers_roundtrip() {
    let orientation = Quaternion::from_axis_angle(Vector::new([0.2, -0.5, 0.84]), 1.1);
    let start = [
        1.0,
        -2.0,
        0.5,
        orientation.w(),
        orientation.x(),
        orientation.y(),
        orientation.z(),
    ];
    let velocity = [0.1, 0.2, 0.3, -0.4, 0.5, -0.6];

    let state = FreeJointState::from_generalized_vectors(start, velocity).unwrap();
    let back = leading_positive(state.generalized_position());
    let want = leading_positive(start);

    for index in 0..7 {
        assert!((back[index] - want[index]).abs() < 1e-12, "number {index}");
    }
    assert_eq!(state.generalized_velocity(), velocity);
}

#[test]
fn an_orientation_naming_no_direction_is_refused() {
    assert!(FreeJointState::from_generalized_vectors([0.0_f64; 7], [0.0; 6]).is_none());
}

#[test]
fn a_drifted_orientation_is_scaled_back_to_length() {
    // Twice the length of the identity orientation still faces the same way.
    let doubled = [0.0_f64, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
    let state = FreeJointState::from_generalized_vectors(doubled, [0.0; 6]).unwrap();
    let place = state.generalized_position();
    assert!((place[3] - 1.0).abs() < 1e-12);
    for (index, value) in place.iter().enumerate().skip(4) {
        assert!(value.abs() < 1e-12, "number {index}");
    }
}

#[test]
fn pose_and_velocity_read_back_as_given() {
    let pose = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
    let velocity = Twist::from_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let state = FreeJointState::new(pose, velocity);
    assert_eq!(state.pose(), pose);
    assert_eq!(state.velocity(), velocity);
    assert_eq!(state.generalized_position()[0], 1.0);
}

#[test]
fn works_at_single_precision() {
    let orientation = Quaternion::from_axis_angle(Vector::new([0.0_f32, 0.0, 1.0]), 0.6);
    let start = [
        1.0_f32,
        2.0,
        -0.5,
        orientation.w(),
        orientation.x(),
        orientation.y(),
        orientation.z(),
    ];
    let state = FreeJointState::from_generalized_vectors(start, [0.0_f32; 6]).unwrap();
    let back = state.generalized_position();

    for index in 0..7 {
        let scale = start[index].abs().max(1.0);
        assert!(
            (back[index] - start[index]).abs() < 1e-5 * scale,
            "number {index}"
        );
    }
}
