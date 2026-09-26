//! Thrust command tests: holding still, tipping the way it is asked to accelerate, the heading
//! turning the body about its push, the two directions that cannot be worked out, and a hand-off
//! to the attitude controller.

use multicalc::control::{
    GeometricAttitudeController, thrust_command_from_acceleration,
    thrust_command_from_acceleration_with_tilt_limit,
};
use multicalc::error::ControlError;
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::Numeric;

const GRAVITY: f64 = 9.81;

#[test]
fn holding_still_is_level_and_carries_the_weight() {
    let command =
        thrust_command_from_acceleration(Vector::new([0.0, 0.0, 0.0]), 0.0, GRAVITY).unwrap();
    assert!((command.thrust_acceleration() - GRAVITY).abs() < 1e-12);
    assert!(command.attitude().log().norm() < 1e-12);
}

#[test]
fn speeding_up_along_x_tips_that_way() {
    let command =
        thrust_command_from_acceleration(Vector::new([2.0, 0.0, 0.0]), 0.0, GRAVITY).unwrap();
    assert!(command.thrust_acceleration() > GRAVITY);
    let up_axis = command.attitude().act(Vector::new([0.0, 0.0, 1.0]));
    assert!(up_axis[0] > 0.0);
    assert!(up_axis[1].abs() < 1e-12);
}

#[test]
fn speeding_up_along_y_tips_that_way() {
    let command =
        thrust_command_from_acceleration(Vector::new([0.0, 2.0, 0.0]), 0.0, GRAVITY).unwrap();
    let up_axis = command.attitude().act(Vector::new([0.0, 0.0, 1.0]));
    assert!(up_axis[1] > 0.0);
    assert!(up_axis[0].abs() < 1e-12);
}

#[test]
fn climbing_stays_level_and_pushes_harder() {
    let command =
        thrust_command_from_acceleration(Vector::new([0.0, 0.0, 3.0]), 0.0, GRAVITY).unwrap();
    assert!((command.thrust_acceleration() - (GRAVITY + 3.0)).abs() < 1e-12);
    assert!(command.attitude().log().norm() < 1e-12);
}

#[test]
fn the_up_axis_lies_along_the_push() {
    let acceleration_command = Vector::new([1.5, -2.5, 0.75]);
    let command = thrust_command_from_acceleration(acceleration_command, 0.4, GRAVITY).unwrap();
    let push = acceleration_command + Vector::new([0.0, 0.0, GRAVITY]);
    let up_axis = command.attitude().act(Vector::new([0.0, 0.0, 1.0]));
    assert!((up_axis - push.normalized()).norm() < 1e-12);
}

#[test]
fn thrust_matches_the_size_of_the_push() {
    let acceleration_command = Vector::new([1.5, -2.5, 0.75]);
    let command = thrust_command_from_acceleration(acceleration_command, 0.4, GRAVITY).unwrap();
    let push = acceleration_command + Vector::new([0.0, 0.0, GRAVITY]);
    assert!((command.thrust_acceleration() - push.norm()).abs() < 1e-12);
}

#[test]
fn heading_turns_the_body_about_its_push() {
    let quarter_turn = core::f64::consts::FRAC_PI_2;
    let command =
        thrust_command_from_acceleration(Vector::new([0.0, 0.0, 0.0]), quarter_turn, GRAVITY)
            .unwrap();
    // The body stays level, so its nose points along +y.
    let nose = command.attitude().act(Vector::new([1.0, 0.0, 0.0]));
    assert!((nose - Vector::new([0.0, 1.0, 0.0])).norm() < 1e-12);
}

#[test]
fn thrust_force_scales_with_mass() {
    let command =
        thrust_command_from_acceleration(Vector::new([1.0, 0.0, 0.0]), 0.0, GRAVITY).unwrap();
    assert_eq!(command.thrust_force(0.0), 0.0);
    assert!((command.thrust_force(2.0) - 2.0 * command.thrust_acceleration()).abs() < 1e-12);
}

#[test]
fn free_fall_has_no_thrust_direction() {
    assert_eq!(
        thrust_command_from_acceleration(Vector::new([0.0, 0.0, -GRAVITY]), 0.0, GRAVITY).err(),
        Some(ControlError::UndefinedThrustDirection)
    );
}

#[test]
fn pushing_along_the_heading_has_no_heading_direction() {
    // The push comes out as [5, 0, 0], straight along the heading.
    assert_eq!(
        thrust_command_from_acceleration(Vector::new([5.0, 0.0, -GRAVITY]), 0.0, GRAVITY).err(),
        Some(ControlError::UndefinedHeadingDirection)
    );
}

#[test]
fn rejects_non_finite() {
    assert_eq!(
        thrust_command_from_acceleration(Vector::new([f64::NAN, 0.0, 0.0]), 0.0, GRAVITY).err(),
        Some(ControlError::NonFinite)
    );
    assert_eq!(
        thrust_command_from_acceleration(Vector::new([0.0, 0.0, 0.0]), f64::NAN, GRAVITY).err(),
        Some(ControlError::NonFinite)
    );
    assert_eq!(
        thrust_command_from_acceleration(Vector::new([0.0, 0.0, 0.0]), 0.0, f64::NAN).err(),
        Some(ControlError::NonFinite)
    );
}

#[test]
fn hands_over_to_the_attitude_controller() {
    let command =
        thrust_command_from_acceleration(Vector::new([1.0, -0.5, 0.4]), 0.2, GRAVITY).unwrap();

    let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
    let attitude_controller = GeometricAttitudeController::new(6.0, 1.2, inertia).unwrap();
    let still = Vector::new([0.0, 0.0, 0.0]);
    let target = command.attitude();
    let torque = attitude_controller.torque(target, still, target, still, still);
    assert!(torque.norm() < 1e-14, "torque {}", torque.norm());
}

#[test]
fn works_at_f32() {
    fn check<T: Numeric>(tolerance: T) {
        let gravity = T::from_f64(GRAVITY);
        let acceleration_command =
            Vector::new([T::from_f64(1.5), T::from_f64(-2.5), T::from_f64(0.75)]);
        let command =
            thrust_command_from_acceleration(acceleration_command, T::from_f64(0.4), gravity)
                .unwrap();
        let push = acceleration_command + Vector::new([T::ZERO, T::ZERO, gravity]);
        let up_axis = command
            .attitude()
            .act(Vector::new([T::ZERO, T::ZERO, T::ONE]));
        assert!((up_axis - push.normalized()).norm() < tolerance);
        assert!((command.thrust_acceleration() - push.norm()).abs() < tolerance);
    }
    check(1e-5_f32);
}

#[test]
fn modest_request_is_unchanged_by_the_tilt_limit() {
    let acceleration_command = Vector::new([1.5, -2.5, 0.75]);
    let heading = 0.4;
    let generous_limit = 1.3; // well past what this request needs
    let capped = thrust_command_from_acceleration_with_tilt_limit(
        acceleration_command,
        heading,
        GRAVITY,
        generous_limit,
    )
    .unwrap();
    let uncapped =
        thrust_command_from_acceleration(acceleration_command, heading, GRAVITY).unwrap();

    assert!(!capped.tilt_bound());
    assert_eq!(capped.attitude(), uncapped.attitude());
    assert!((capped.thrust_acceleration() - uncapped.thrust_acceleration()).abs() < 1e-12);
}

#[test]
fn large_lateral_request_is_capped_at_the_limit_with_vertical_preserved() {
    let max_tilt = 0.5;
    let command = thrust_command_from_acceleration_with_tilt_limit(
        Vector::new([50.0, 0.0, 0.0]),
        0.0,
        GRAVITY,
        max_tilt,
    )
    .unwrap();

    assert!(command.tilt_bound());
    let up_axis = command.attitude().act(Vector::new([0.0, 0.0, 1.0]));
    // The attitude sits right at the cap...
    assert!((up_axis[2].acos() - max_tilt).abs() < 1e-9);
    // ...and the vertical part of the push — thrust_acceleration projected onto up — is exactly
    // what it was before capping: gravity, since the wanted acceleration had no vertical part.
    assert!((command.thrust_acceleration() * up_axis[2] - GRAVITY).abs() < 1e-9);
}

#[test]
fn capping_preserves_the_direction_of_the_lateral_push() {
    let acceleration_command = Vector::new([40.0, 30.0, 0.0]);
    let command =
        thrust_command_from_acceleration_with_tilt_limit(acceleration_command, 0.0, GRAVITY, 0.4)
            .unwrap();

    assert!(command.tilt_bound());
    let up_axis = command.attitude().act(Vector::new([0.0, 0.0, 1.0]));
    // Same azimuth as the uncapped push — scaled down, not turned.
    assert!((up_axis[0] * 30.0 - up_axis[1] * 40.0).abs() < 1e-9);
    assert!(up_axis[0] > 0.0 && up_axis[1] > 0.0);
}

#[test]
fn tilt_limit_still_reports_undefined_directions() {
    assert_eq!(
        thrust_command_from_acceleration_with_tilt_limit(
            Vector::new([0.0, 0.0, -GRAVITY]),
            0.0,
            GRAVITY,
            0.5
        )
        .err(),
        Some(ControlError::UndefinedThrustDirection)
    );
    assert_eq!(
        thrust_command_from_acceleration_with_tilt_limit(
            Vector::new([5.0, 0.0, -GRAVITY]),
            0.0,
            GRAVITY,
            0.5
        )
        .err(),
        Some(ControlError::UndefinedHeadingDirection)
    );
}

#[test]
fn descent_faster_than_free_fall_is_still_capped() {
    // A lateral request paired with a descent so steep the push already points below level
    // (vertical = az + gravity < 0). Capping still has to engage here: previously the vertical >
    // 0 guard skipped it entirely, silently handing back a command tilted way past max_tilt while
    // claiming tilt_bound() == false.
    let max_tilt = 0.5;
    for descent in [12.0, 60.0] {
        let command = thrust_command_from_acceleration_with_tilt_limit(
            Vector::new([50.0, 0.0, -descent]),
            0.0,
            GRAVITY,
            max_tilt,
        )
        .unwrap();

        assert!(command.tilt_bound());
        let up_axis = command.attitude().act(Vector::new([0.0, 0.0, 1.0]));
        assert!((up_axis[2].acos() - max_tilt).abs() < 1e-9);
        // The lateral direction is preserved, not flipped to the far side.
        assert!(up_axis[0] > 0.0);
    }
}

#[test]
fn straight_down_descent_leans_toward_the_heading() {
    // No lateral push at all to take a direction from (horizontal == 0), just a straight-down
    // command steeper than free-fall. There is nothing to preserve the direction of, so the cap
    // leans toward desired_heading instead of leaving it undefined or dividing by zero.
    let max_tilt = 0.5;
    let heading = 0.9;
    let command = thrust_command_from_acceleration_with_tilt_limit(
        Vector::new([0.0, 0.0, -60.0]),
        heading,
        GRAVITY,
        max_tilt,
    )
    .unwrap();

    assert!(command.tilt_bound());
    let up_axis = command.attitude().act(Vector::new([0.0, 0.0, 1.0]));
    assert!((up_axis[2].acos() - max_tilt).abs() < 1e-9);
    assert!((up_axis[0] * heading.sin() - up_axis[1] * heading.cos()).abs() < 1e-9);
}

#[test]
fn rejects_invalid_tilt_limit() {
    let level = Vector::new([0.0, 0.0, 0.0]);
    let quarter_turn = core::f64::consts::FRAC_PI_2;
    for max_tilt in [0.0, -0.2, quarter_turn, quarter_turn + 0.1] {
        assert_eq!(
            thrust_command_from_acceleration_with_tilt_limit(level, 0.0, GRAVITY, max_tilt).err(),
            Some(ControlError::InvalidTiltLimit)
        );
    }
}

#[test]
fn rejects_non_finite_tilt_limit() {
    assert_eq!(
        thrust_command_from_acceleration_with_tilt_limit(
            Vector::new([0.0, 0.0, 0.0]),
            0.0,
            GRAVITY,
            f64::NAN
        )
        .err(),
        Some(ControlError::NonFinite)
    );
}
