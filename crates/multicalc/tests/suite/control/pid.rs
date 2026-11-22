//! PID tests: setpoint tracking, output saturation, conditional-integration anti-windup, the
//! derivative acting on the measurement, bumpless gain changes and handover, reset, constructor
//! rejection, and an autodiff-vs-finite-difference sanity check.

use multicalc::control::Pid;
use multicalc::error::ControlError;
use multicalc::scalar::{Dual, Numeric};

// A scalar first-order plant `x_next = x + timestep * output` driven to a setpoint.
fn assert_drives_to_setpoint<T: Numeric>(tolerance: T) {
    let proportional_gain = T::from_f64(2.0);
    let integral_gain = T::from_f64(1.0);
    let derivative_gain = T::ZERO;
    let timestep = T::from_f64(0.01);
    let mut controller =
        Pid::new(proportional_gain, integral_gain, derivative_gain, timestep).unwrap();
    let setpoint = T::ONE;
    let mut measurement = T::ZERO;
    for _ in 0..3000 {
        let output = controller.update(setpoint, measurement);
        measurement += timestep * output;
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
    let proportional_gain = 50.0_f64;
    let integral_gain = 10.0;
    let derivative_gain = 0.0;
    let timestep = 0.01;
    let minimum = -1.0;
    let maximum = 1.0;
    let mut controller = Pid::new(proportional_gain, integral_gain, derivative_gain, timestep)
        .unwrap()
        .with_output_limits(minimum, maximum)
        .unwrap();
    let mut measurement = 0.0;
    for _ in 0..500 {
        let output = controller.update(5.0, measurement);
        assert!((-1.0..=1.0).contains(&output));
        measurement += timestep * output;
    }
}

#[test]
fn conditional_integration_bounds_the_integral() {
    let proportional_gain = 1.0_f64;
    let integral_gain = 5.0;
    let derivative_gain = 0.0;
    let timestep = 0.01;
    let minimum = -1.0;
    let maximum = 1.0;
    let mut controller = Pid::new(proportional_gain, integral_gain, derivative_gain, timestep)
        .unwrap()
        .with_output_limits(minimum, maximum)
        .unwrap();
    // A setpoint the saturated plant can never reach. The naive integral would grow like
    // error * timestep * steps ≈ 1e5; conditional integration freezes it while saturated instead.
    let mut measurement = 0.0;
    for _ in 0..10_000 {
        let output = controller.update(1000.0, measurement);
        measurement += timestep * output;
    }
    assert!(controller.integral().abs() < 1.0);
}

#[test]
fn zero_gains_give_zero_command() {
    let proportional_gain = 0.0_f64;
    let integral_gain = 0.0;
    let derivative_gain = 0.0;
    let timestep = 0.01;
    let mut controller =
        Pid::new(proportional_gain, integral_gain, derivative_gain, timestep).unwrap();
    for (setpoint, measurement) in [(1.0, 0.0), (5.0, 2.0), (-3.0, 1.0)] {
        assert_eq!(controller.update(setpoint, measurement), 0.0);
    }
}

#[test]
fn reset_zeroes_the_integral() {
    let proportional_gain = 1.0_f64;
    let integral_gain = 1.0;
    let derivative_gain = 0.0;
    let timestep = 0.01;
    let mut controller =
        Pid::new(proportional_gain, integral_gain, derivative_gain, timestep).unwrap();
    for _ in 0..100 {
        let _ = controller.update(1.0, 0.0);
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
    let minimum = 1.0;
    let maximum = -1.0;
    assert_eq!(
        Pid::new(1.0_f64, 0.0, 0.0, 0.01)
            .unwrap()
            .with_output_limits(minimum, maximum)
            .err(),
        Some(ControlError::InvalidOutputLimits)
    );

    let minimum = f64::NAN;
    let maximum = 1.0;
    assert_eq!(
        Pid::new(1.0_f64, 0.0, 0.0, 0.01)
            .unwrap()
            .with_output_limits(minimum, maximum)
            .err(),
        Some(ControlError::NonFinite)
    );

    let minimum = f64::NEG_INFINITY;
    let maximum = f64::INFINITY;
    assert!(
        Pid::new(1.0_f64, 0.0, 0.0, 0.01)
            .unwrap()
            .with_output_limits(minimum, maximum)
            .is_ok()
    );
}

#[test]
fn setpoint_step_does_not_spike_the_derivative() {
    let mut controller = Pid::new(0.0_f64, 0.0, 1.0, 0.01).unwrap();
    // Settle the measurement history at zero.
    let _ = controller.update(0.0, 0.0);
    let _ = controller.update(0.0, 0.0);
    // The setpoint jumps by a whole unit while the measurement holds still.
    let output = controller.update(1.0, 0.0);
    assert_eq!(output, 0.0);
}

#[test]
fn derivative_opposes_a_rising_measurement() {
    let mut controller = Pid::new(0.0_f64, 0.0, 1.0, 0.01).unwrap();
    let _ = controller.update(0.0, 0.0);
    // The measurement rose by 0.05 over 0.01 s, so the term pushes back by 5.
    let output = controller.update(0.0, 0.05);
    assert!((output + 5.0).abs() < 1e-12);
}

#[test]
fn matches_derivative_on_error_when_the_setpoint_holds_still() {
    let proportional_gain = 1.0_f64;
    let derivative_gain = 0.2;
    let timestep = 0.01;
    let setpoint = 1.0;
    let mut controller = Pid::new(proportional_gain, 0.0, derivative_gain, timestep).unwrap();

    let mut previous_measurement = 0.0;
    let mut is_first = true;
    for measurement in [0.0, 0.1, 0.25, 0.4, 0.42] {
        let output = controller.update(setpoint, measurement);
        let rate = if is_first {
            0.0
        } else {
            (previous_measurement - measurement) / timestep
        };
        let expected = proportional_gain * (setpoint - measurement) + derivative_gain * rate;
        assert!(
            (output - expected).abs() < 1e-12,
            "output {output}, expected {expected}"
        );
        previous_measurement = measurement;
        is_first = false;
    }
}

#[test]
fn gain_change_does_not_step_the_output() {
    let timestep = 0.01;
    let mut controller = Pid::new(2.0_f64, 0.0, 0.1, timestep).unwrap();
    // The setpoint and the measurement climb together, so the error holds still and the
    // measurement moves at a steady rate. Both terms the gain change has to account for then read
    // the same on the step before the change as on the step after it.
    for step in 0..4 {
        let measurement = timestep * f64::from(step);
        let _ = controller.update(1.0 + measurement, measurement);
    }
    let next_measurement = timestep * 4.0;
    let next_setpoint = 1.0 + next_measurement;

    // What the old gains would have given on the next call.
    let mut reference = controller;
    let before = reference.update(next_setpoint, next_measurement);

    controller.set_gains(5.0, 0.0, 0.4).unwrap();
    let after = controller.update(next_setpoint, next_measurement);
    assert!(
        (after - before).abs() < 1e-12,
        "before {before}, after {after}"
    );
}

#[test]
fn handover_reproduces_the_command_it_took_over() {
    let mut controller = Pid::new(2.0_f64, 1.0, 0.1, 0.01).unwrap();
    let manual_command = 0.6;
    let setpoint = 1.0;
    let measurement = 0.35;
    controller
        .resume_from(manual_command, setpoint, measurement)
        .unwrap();
    let first = controller.update(setpoint, measurement);
    assert!((first - manual_command).abs() < 1e-12, "first {first}");
}

#[test]
fn handover_then_settles_to_the_setpoint() {
    let timestep = 0.01;
    let setpoint = 1.0;
    let mut controller = Pid::new(2.0_f64, 1.0, 0.1, timestep).unwrap();
    let mut measurement = 0.35;
    controller.resume_from(0.6, setpoint, measurement).unwrap();
    for _ in 0..3000 {
        let output = controller.update(setpoint, measurement);
        measurement += timestep * output;
    }
    assert!(
        (measurement - setpoint).abs() < 1e-3,
        "measurement {measurement}"
    );
}

#[test]
fn set_gains_rejects_non_finite() {
    let mut controller = Pid::new(1.0_f64, 1.0, 1.0, 0.01).unwrap();
    assert_eq!(
        controller.set_gains(f64::NAN, 1.0, 1.0),
        Err(ControlError::NonFinite)
    );
}

#[test]
fn resume_from_rejects_non_finite() {
    let mut controller = Pid::new(1.0_f64, 1.0, 1.0, 0.01).unwrap();
    assert_eq!(
        controller.resume_from(f64::INFINITY, 1.0, 0.0),
        Err(ControlError::NonFinite)
    );
}

#[test]
fn reset_clears_the_measurement_history() {
    let proportional_gain = 1.0_f64;
    let mut controller = Pid::new(proportional_gain, 0.0, 0.5, 0.01).unwrap();
    for measurement in [0.0, 0.1, 0.2] {
        let _ = controller.update(1.0, measurement);
    }
    controller.reset();
    // With nothing to compare against, the derivative contributes nothing on the first call back.
    let output = controller.update(1.0, 0.4);
    assert!(
        (output - proportional_gain * 0.6).abs() < 1e-12,
        "output {output}"
    );
}

// One `update` step (after seeding the derivative history), as a function of the setpoint.
fn output_of_one_step<T: Numeric>(setpoint: T) -> T {
    let proportional_gain = T::from_f64(2.0);
    let integral_gain = T::from_f64(0.5);
    let derivative_gain = T::from_f64(0.1);
    let timestep = T::from_f64(0.01);
    let mut controller =
        Pid::new(proportional_gain, integral_gain, derivative_gain, timestep).unwrap();
    let _ = controller.update(setpoint, T::ZERO);
    controller.update(setpoint, T::from_f64(0.3))
}

#[test]
fn output_setpoint_derivative_matches_finite_difference() {
    let setpoint = 1.0_f64;
    let autodiff = output_of_one_step(Dual::variable(setpoint)).deriv;
    let step = 1e-6;
    let finite_difference =
        (output_of_one_step(setpoint + step) - output_of_one_step(setpoint - step)) / (2.0 * step);
    assert!(
        (autodiff - finite_difference).abs() < 1e-6,
        "autodiff {autodiff}, finite difference {finite_difference}"
    );
}
