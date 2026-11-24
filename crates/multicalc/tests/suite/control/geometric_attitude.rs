//! Geometric attitude tests: nothing to do at the target, the torque opposing a tilt, tracking a
//! turning target, a simulated body settling on the target, the shorter path past a half turn, and
//! constructor rejection.

use multicalc::SO3;
use multicalc::control::{GeometricAttitudeController, LimitedOutput};
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
fn attitude_error_is_largest_at_a_half_turn() {
    let attitude = SO3::exp(Vector::new([core::f64::consts::PI, 0.0, 0.0]));
    let error = GeometricAttitudeController::<f64>::attitude_error(attitude, SO3::identity());
    assert!((error.norm() - 2.0).abs() < 1e-12, "error {}", error.norm());
}

#[test]
fn torque_takes_the_shorter_way_past_a_half_turn() {
    let control = controller::<f64>();
    let still = Vector::new([0.0, 0.0, 0.0]);
    let level = SO3::<f64>::identity();
    let before = SO3::exp(Vector::new([core::f64::consts::PI - 0.1, 0.0, 0.0]));
    let past = SO3::exp(Vector::new([core::f64::consts::PI + 0.1, 0.0, 0.0]));
    assert!(control.torque(before, still, level, still, still)[0] < 0.0);
    assert!(control.torque(past, still, level, still, still)[0] > 0.0);
}

fn error_about_x<T: Numeric>(angle: T) -> T {
    let attitude = SO3::exp(Vector::new([angle, T::ZERO, T::ZERO]));
    GeometricAttitudeController::<T>::attitude_error(attitude, SO3::identity())[0]
}

#[test]
fn attitude_error_is_differentiable_across_a_half_turn() {
    let step = 1e-6;
    for angle in [core::f64::consts::PI - 0.1, core::f64::consts::PI + 0.1] {
        let autodiff = error_about_x(Dual::variable(angle)).deriv;
        let finite_difference =
            (error_about_x(angle + step) - error_about_x(angle - step)) / (2.0 * step);
        assert!(
            (autodiff - finite_difference).abs() < 1e-6,
            "angle {angle}: autodiff {autodiff}, finite difference {finite_difference}"
        );
    }
    // d(2·sin(θ/2))/dθ = cos(θ/2) = 0 at θ = π.
    let at_half_turn = error_about_x(Dual::variable(core::f64::consts::PI)).deriv;
    assert!(at_half_turn.abs() < 1e-6, "derivative {at_half_turn}");
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

const DISTURBANCE: [f64; 3] = [0.02, -0.01, 0.005];

/// Runs `update` on a body with a constant disturbance torque. Returns the final attitude error,
/// the final body rate, each command, and the controller.
fn run_with_disturbance(
    mut control: GeometricAttitudeController<f64>,
    steps: usize,
) -> (
    f64,
    f64,
    Vec<LimitedOutput<3>>,
    GeometricAttitudeController<f64>,
) {
    let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
    let inverse_inertia = Matrix::<3, 3>::from_diagonal([50.0, 50.0, 25.0]);
    let disturbance = Vector::new(DISTURBANCE);
    let target = SO3::<f64>::identity();
    let still = Vector::new([0.0, 0.0, 0.0]);
    let timestep = 0.002;

    let mut attitude = SO3::exp(Vector::new([2.5, -0.4, 0.9]));
    let mut body_rate = Vector::new([0.0, 0.0, 0.0]);
    let mut commands = Vec::with_capacity(steps);
    for _ in 0..steps {
        let command = control.update(attitude, body_rate, target, still, still);
        commands.push(command);
        let rate_change = inverse_inertia
            * (command.output() + disturbance - body_rate.cross(inertia * body_rate));
        body_rate += rate_change.scale(timestep);
        attitude = attitude
            .compose(SO3::exp(body_rate.scale(timestep)))
            .normalized();
    }
    let error = GeometricAttitudeController::<f64>::attitude_error(attitude, target);
    (error.norm(), body_rate.norm(), commands, control)
}

fn with_integral() -> GeometricAttitudeController<f64> {
    controller::<f64>()
        .with_integral(4.0, Vector::new([0.05, 0.05, 0.05]), 0.002)
        .unwrap()
}

#[test]
fn proportional_derivative_law_leaves_a_steady_error() {
    let (error, _, _, _) = run_with_disturbance(controller::<f64>(), 20_000);
    assert!(error > 1e-3, "error {error}");
}

#[test]
fn integral_removes_a_constant_disturbance() {
    let (error, rate, _, control) = run_with_disturbance(with_integral(), 20_000);
    assert!(error < 1e-9, "error {error}");
    assert!(rate < 1e-9, "rate {rate}");
    // At rest, k_I·∫e = d.
    let expected = Vector::new(DISTURBANCE).scale(1.0 / 4.0);
    assert!((control.integral() - expected).norm() < 1e-9);
}

#[test]
fn torque_limits_hold_and_report() {
    let limits = [0.15, 0.15, 0.075];
    let control = with_integral()
        .with_torque_limits(Vector::new(limits))
        .unwrap();
    let (error, _, commands, _) = run_with_disturbance(control, 20_000);
    assert!(commands[0].any_limited());
    for command in &commands {
        for axis in 0..3 {
            assert!(command.output()[axis].abs() <= limits[axis]);
        }
    }
    assert!(error < 1e-9, "error {error}");
}

#[test]
fn update_without_builders_equals_torque() {
    let mut control = controller::<f64>();
    let attitude = SO3::exp(Vector::new([0.4, -0.3, 0.8]));
    let body_rate = Vector::new([0.9, -0.5, 0.35]);
    let target = SO3::exp(Vector::new([-0.2, 0.15, 0.05]));
    let target_rate = Vector::new([0.1, 0.2, -0.3]);
    let target_rate_change = Vector::new([0.05, -0.02, 0.01]);
    let torque = control.torque(attitude, body_rate, target, target_rate, target_rate_change);
    let command = control.update(attitude, body_rate, target, target_rate, target_rate_change);
    assert_eq!(command.output(), torque);
    assert!(!command.any_limited());
}

#[test]
fn integral_holds_while_the_torque_is_limited() {
    let mut control = with_integral()
        .with_torque_limits(Vector::new([0.01, 0.01, 0.01]))
        .unwrap();
    let tipped = SO3::exp(Vector::new([0.5, 0.0, 0.0]));
    let still = Vector::new([0.0, 0.0, 0.0]);
    for _ in 0..10 {
        let command = control.update(tipped, still, SO3::identity(), still, still);
        assert_eq!(command.limited(), [true, false, false]);
    }
    assert_eq!(control.integral()[0], 0.0);
}

#[test]
fn reset_clears_the_integral() {
    let (_, _, _, mut control) = run_with_disturbance(with_integral(), 100);
    assert!(control.integral().norm() > 0.0);
    control.reset();
    assert_eq!(control.integral(), Vector::new([0.0, 0.0, 0.0]));
}

#[test]
fn rejects_invalid_integral_and_torque_limits() {
    let limit = Vector::new([0.05, 0.05, 0.05]);
    let base = controller::<f64>();
    assert_eq!(
        base.with_integral(0.0, limit, 0.002).err(),
        Some(ControlError::NonPositiveGain)
    );
    assert_eq!(
        base.with_integral(-1.0, limit, 0.002).err(),
        Some(ControlError::NonPositiveGain)
    );
    assert_eq!(
        base.with_integral(f64::NAN, limit, 0.002).err(),
        Some(ControlError::NonFinite)
    );
    assert_eq!(
        base.with_integral(4.0, limit, 0.0).err(),
        Some(ControlError::NonPositiveTimestep)
    );
    assert_eq!(
        base.with_integral(4.0, Vector::new([-0.05, 0.05, 0.05]), 0.002)
            .err(),
        Some(ControlError::InvalidOutputLimits)
    );
    assert_eq!(
        base.with_torque_limits(Vector::new([f64::NAN, 0.1, 0.1]))
            .err(),
        Some(ControlError::NonFinite)
    );
    assert_eq!(
        base.with_torque_limits(Vector::new([-0.1, 0.1, 0.1])).err(),
        Some(ControlError::InvalidOutputLimits)
    );
}
