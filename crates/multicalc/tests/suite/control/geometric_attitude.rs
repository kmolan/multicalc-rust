//! Geometric attitude tests: nothing to do at the target, the torque opposing a tilt, tracking a
//! turning target, a simulated body settling on the target, the half-turn balance point, and
//! constructor rejection.

use multicalc::SO3;
use multicalc::control::GeometricAttitudeController;
use multicalc::error::ControlError;
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Dual, Numeric};

/// A small quadrotor-sized body: heavier to spin about its up axis than about the other two.
fn controller<T: Numeric>() -> GeometricAttitudeController<T> {
    let inertia = Matrix::from_diagonal([T::from_f64(0.02), T::from_f64(0.02), T::from_f64(0.04)]);
    GeometricAttitudeController::new(T::from_f64(6.0), T::from_f64(1.2), inertia).unwrap()
}

#[test]
fn nothing_to_do_at_the_target() {
    let control = controller::<f64>();
    let level = SO3::<f64>::identity();
    let still = Vector::new([0.0, 0.0, 0.0]);
    let torque = control.torque(level, still, level, still, still);
    assert!(torque.norm() < 1e-15);
}

#[test]
fn torque_opposes_a_tilt() {
    let control = controller::<f64>();
    let still = Vector::new([0.0, 0.0, 0.0]);
    let tilted = SO3::exp(Vector::new([0.1, 0.0, 0.0]));
    let torque = control.torque(tilted, still, SO3::identity(), still, still);
    assert!(torque[0] < 0.0);
    assert!(torque[1].abs() < 1e-12);
    assert!(torque[2].abs() < 1e-12);
}

#[test]
fn torque_opposes_a_turn() {
    let control = controller::<f64>();
    let level = SO3::<f64>::identity();
    let still = Vector::new([0.0, 0.0, 0.0]);
    let body_rate = Vector::new([0.0, 0.3, 0.0]);
    let torque = control.torque(level, body_rate, level, still, still);
    // A turn about one of the body's own axes leaves no spin term, so only the rate gain acts.
    assert!((torque[1] + 0.36).abs() < 1e-12);
}

#[test]
fn attitude_error_is_zero_at_the_target() {
    let rotation = SO3::exp(Vector::new([0.3, -0.2, 0.7]));
    let error = GeometricAttitudeController::<f64>::attitude_error(rotation, rotation);
    assert!(error.norm() < 1e-15);
}

#[test]
fn attitude_error_matches_the_turn_the_body_is_off_by() {
    // For a small turn away from the target, the error reads back as that same turn.
    let axis = Vector::new([0.02, -0.01, 0.03]);
    let attitude = SO3::exp(axis);
    let error = GeometricAttitudeController::<f64>::attitude_error(attitude, SO3::identity());
    assert!((error - axis).norm() < 1e-4);
}

#[test]
fn attitude_error_vanishes_at_a_half_turn() {
    let attitude = SO3::exp(Vector::new([core::f64::consts::PI, 0.0, 0.0]));
    let error = GeometricAttitudeController::<f64>::attitude_error(attitude, SO3::identity());
    assert!(error.norm() < 1e-9, "error {}", error.norm());
}

#[test]
fn body_settles_on_the_target() {
    let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
    let inverse_inertia = Matrix::<3, 3>::from_diagonal([50.0, 50.0, 25.0]);
    let control = controller::<f64>();
    let target = SO3::<f64>::identity();
    let still = Vector::new([0.0, 0.0, 0.0]);

    let mut attitude = SO3::exp(Vector::new([0.6, -0.4, 0.9]));
    let mut body_rate = Vector::new([1.5, -1.0, 0.8]);
    let timestep = 0.002;
    for _ in 0..4000 {
        let torque = control.torque(attitude, body_rate, target, still, still);
        // What is left over after the body's own spin drives the change in turn rate.
        let rate_change = inverse_inertia * (torque - body_rate.cross(inertia * body_rate));
        body_rate += rate_change.scale(timestep);
        attitude = attitude
            .compose(SO3::exp(body_rate.scale(timestep)))
            .normalized();
    }
    let error = GeometricAttitudeController::<f64>::attitude_error(attitude, target);
    assert!(error.norm() < 1e-3, "error {}", error.norm());
    assert!(body_rate.norm() < 1e-2, "rate {}", body_rate.norm());
}

#[test]
fn tracks_a_turning_target() {
    let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
    let inverse_inertia = Matrix::<3, 3>::from_diagonal([50.0, 50.0, 25.0]);
    let control = controller::<f64>();
    let still = Vector::new([0.0, 0.0, 0.0]);
    let turn_rate = 0.5;
    let desired_body_rate = Vector::new([0.0, 0.0, turn_rate]);

    let mut target = SO3::<f64>::identity();
    let mut attitude = target;
    let mut body_rate = desired_body_rate;
    let timestep = 0.002;
    for _ in 0..4000 {
        let torque = control.torque(attitude, body_rate, target, desired_body_rate, still);
        let rate_change = inverse_inertia * (torque - body_rate.cross(inertia * body_rate));
        body_rate += rate_change.scale(timestep);
        attitude = attitude
            .compose(SO3::exp(body_rate.scale(timestep)))
            .normalized();
        target = target.compose(SO3::exp(Vector::new([0.0, 0.0, turn_rate * timestep])));

        let error = GeometricAttitudeController::<f64>::attitude_error(attitude, target);
        assert!(error.norm() < 1e-3, "error {}", error.norm());
    }
}

#[test]
fn works_at_f32() {
    let inertia = Matrix::<3, 3, f32>::from_diagonal([0.02, 0.02, 0.04]);
    let inverse_inertia = Matrix::<3, 3, f32>::from_diagonal([50.0, 50.0, 25.0]);
    let control = controller::<f32>();
    let target = SO3::<f32>::identity();
    let still = Vector::new([0.0_f32, 0.0, 0.0]);

    let mut attitude = SO3::exp(Vector::new([0.6_f32, -0.4, 0.9]));
    let mut body_rate = Vector::new([1.5_f32, -1.0, 0.8]);
    let timestep = 0.002_f32;
    for _ in 0..4000 {
        let torque = control.torque(attitude, body_rate, target, still, still);
        let rate_change = inverse_inertia * (torque - body_rate.cross(inertia * body_rate));
        body_rate += rate_change.scale(timestep);
        attitude = attitude
            .compose(SO3::exp(body_rate.scale(timestep)))
            .normalized();
    }
    let error = GeometricAttitudeController::<f32>::attitude_error(attitude, target);
    assert!(error.norm() < 1e-2_f32, "error {}", error.norm());
}

// The first torque component for a fixed tilt, as a function of the attitude gain.
fn torque_of_one_call<T: Numeric>(attitude_gain: T) -> T {
    let inertia = Matrix::from_diagonal([T::from_f64(0.02), T::from_f64(0.02), T::from_f64(0.04)]);
    let control =
        GeometricAttitudeController::new(attitude_gain, T::from_f64(1.2), inertia).unwrap();
    let tilted = SO3::exp(Vector::new([T::from_f64(0.1), T::from_f64(-0.2), T::ZERO]));
    let body_rate = Vector::new([T::from_f64(0.3), T::ZERO, T::from_f64(-0.1)]);
    let still = Vector::new([T::ZERO, T::ZERO, T::ZERO]);
    control.torque(tilted, body_rate, SO3::identity(), still, still)[0]
}

#[test]
fn torque_is_differentiable() {
    let attitude_gain = 6.0_f64;
    let autodiff = torque_of_one_call(Dual::variable(attitude_gain)).deriv;
    let step = 1e-6;
    let finite_difference = (torque_of_one_call(attitude_gain + step)
        - torque_of_one_call(attitude_gain - step))
        / (2.0 * step);
    assert!(
        (autodiff - finite_difference).abs() < 1e-6,
        "autodiff {autodiff}, finite difference {finite_difference}"
    );
}

#[test]
fn rejects_a_non_positive_gain() {
    assert_eq!(
        GeometricAttitudeController::new(0.0_f64, 1.0, Matrix::identity()).err(),
        Some(ControlError::NonPositiveGain)
    );
    assert_eq!(
        GeometricAttitudeController::new(1.0_f64, -1.0, Matrix::identity()).err(),
        Some(ControlError::NonPositiveGain)
    );
}

#[test]
fn rejects_a_lopsided_inertia() {
    let inertia = Matrix::<3, 3>::new([[1.0, 0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert_eq!(
        GeometricAttitudeController::new(1.0_f64, 1.0, inertia).err(),
        Some(ControlError::NotSymmetricInertia)
    );
}

#[test]
fn rejects_an_inertia_that_is_not_positive() {
    let inertia = Matrix::<3, 3>::from_diagonal([1.0, -1.0, 1.0]);
    assert_eq!(
        GeometricAttitudeController::new(1.0_f64, 1.0, inertia).err(),
        Some(ControlError::NonPositiveInertia)
    );
}

#[test]
fn rejects_non_finite() {
    assert_eq!(
        GeometricAttitudeController::new(f64::NAN, 1.0, Matrix::identity()).err(),
        Some(ControlError::NonFinite)
    );
}
