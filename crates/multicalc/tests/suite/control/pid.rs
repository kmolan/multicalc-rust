//! PID tests: setpoint tracking, output saturation, conditional-integration anti-windup, reset,
//! constructor rejection, and an autodiff-vs-finite-difference sanity check.

use multicalc::control::Pid;
use multicalc::error::ControlError;
use multicalc::scalar::{Dual, Numeric};

// A scalar first-order plant `x_next = x + dt * output` driven to a setpoint.
fn assert_drives_to_setpoint<T: Numeric>(tolerance: T) {
    let dt = T::from_f64(0.01);
    let mut controller = Pid::new(T::from_f64(2.0), T::from_f64(1.0), T::ZERO, dt).unwrap();
    let setpoint = T::ONE;
    let mut measurement = T::ZERO;
    for _ in 0..3000 {
        let output = controller.update(setpoint, measurement);
        measurement += dt * output;
    }
    assert!((measurement - setpoint).abs() < tolerance);
}

#[test]
fn drives_measurement_to_setpoint_f64() {
    assert_drives_to_setpoint(1e-3_f64);
}

#[test]
fn drives_measurement_to_setpoint_f32() {
    assert_drives_to_setpoint(1e-2_f32);
}

#[test]
fn output_never_exceeds_limits() {
    let dt = 0.01_f64;
    let mut controller = Pid::new(50.0, 10.0, 0.0, dt)
        .unwrap()
        .with_output_limits(-1.0, 1.0)
        .unwrap();
    let mut measurement = 0.0;
    for _ in 0..500 {
        let output = controller.update(5.0, measurement);
        assert!((-1.0..=1.0).contains(&output));
        measurement += dt * output;
    }
}

#[test]
fn conditional_integration_bounds_the_integral() {
    let dt = 0.01_f64;
    let mut controller = Pid::new(1.0, 5.0, 0.0, dt)
        .unwrap()
        .with_output_limits(-1.0, 1.0)
        .unwrap();
    // A setpoint the saturated plant can never reach. The naive integral would grow like
    // error * dt * steps ≈ 1e5; conditional integration freezes it while saturated instead.
    let mut measurement = 0.0;
    for _ in 0..10_000 {
        let output = controller.update(1000.0, measurement);
        measurement += dt * output;
    }
    assert!(controller.integral().abs() < 1.0);
}

#[test]
fn zero_gains_give_zero_command() {
    let mut controller = Pid::new(0.0_f64, 0.0, 0.0, 0.01).unwrap();
    for (setpoint, measurement) in [(1.0, 0.0), (5.0, 2.0), (-3.0, 1.0)] {
        assert_eq!(controller.update(setpoint, measurement), 0.0);
    }
}

#[test]
fn reset_zeroes_the_integral() {
    let mut controller = Pid::new(1.0_f64, 1.0, 0.0, 0.01).unwrap();
    for _ in 0..100 {
        controller.update(1.0, 0.0);
    }
    assert!(controller.integral() != 0.0);
    controller.reset();
    assert_eq!(controller.integral(), 0.0);
}

#[test]
fn new_rejects_non_finite_and_non_positive_timestep() {
    assert_eq!(
        Pid::new(f64::NAN, 0.0, 0.0, 0.01),
        Err(ControlError::NonFinite)
    );
    assert_eq!(
        Pid::new(1.0, 1.0, 1.0, 0.0),
        Err(ControlError::NonPositiveTimestep)
    );
    assert_eq!(
        Pid::new(1.0, 1.0, 1.0, -0.5),
        Err(ControlError::NonPositiveTimestep)
    );
}

#[test]
fn output_limits_reject_inverted_and_nan_but_allow_infinities() {
    assert_eq!(
        Pid::new(1.0_f64, 0.0, 0.0, 0.01)
            .unwrap()
            .with_output_limits(1.0, -1.0)
            .err(),
        Some(ControlError::InvalidOutputLimits)
    );
    assert_eq!(
        Pid::new(1.0_f64, 0.0, 0.0, 0.01)
            .unwrap()
            .with_output_limits(f64::NAN, 1.0)
            .err(),
        Some(ControlError::NonFinite)
    );
    assert!(
        Pid::new(1.0_f64, 0.0, 0.0, 0.01)
            .unwrap()
            .with_output_limits(f64::NEG_INFINITY, f64::INFINITY)
            .is_ok()
    );
}

// One `update` step (after seeding the derivative history), as a function of the setpoint.
fn output_of_one_step<T: Numeric>(setpoint: T) -> T {
    let dt = T::from_f64(0.01);
    let mut controller = Pid::new(T::from_f64(2.0), T::from_f64(0.5), T::from_f64(0.1), dt).unwrap();
    controller.update(setpoint, T::ZERO);
    controller.update(setpoint, T::from_f64(0.3))
}

#[test]
fn output_setpoint_derivative_matches_finite_difference() {
    let setpoint = 1.0_f64;
    let autodiff = output_of_one_step(Dual::variable(setpoint)).deriv;
    let h = 1e-6;
    let finite_difference =
        (output_of_one_step(setpoint + h) - output_of_one_step(setpoint - h)) / (2.0 * h);
    assert!(
        (autodiff - finite_difference).abs() < 1e-6,
        "autodiff {autodiff}, finite difference {finite_difference}"
    );
}
