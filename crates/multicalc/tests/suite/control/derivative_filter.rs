//! One-pole low-pass filter tests: DC gain, pass-through, step attenuation, and constructor
//! rejection, at f32 and f64.

use multicalc::control::OnePoleLowPass;
use multicalc::error::ControlError;
use multicalc::scalar::Numeric;

// ---- steady state -----------------------------------------------------------

fn assert_dc_gain_is_one<T: Numeric>(tolerance: T) {
    let mut filter = OnePoleLowPass::new(T::from_f64(0.3)).unwrap();
    let input = T::from_f64(5.0);
    for _ in 0..500 {
        filter.filter(input);
    }
    assert!((filter.value() - input).abs() < tolerance);
}

#[test]
fn dc_gain_is_one_f64() {
    assert_dc_gain_is_one(1e-9_f64);
}

#[test]
fn dc_gain_is_one_f32() {
    assert_dc_gain_is_one(1e-3_f32);
}

// ---- pass-through -----------------------------------------------------------

fn assert_alpha_one_is_pass_through<T: Numeric>() {
    let mut filter = OnePoleLowPass::new(T::ONE).unwrap();
    for x in [1.0, -2.0, 3.5, 0.0] {
        let value = T::from_f64(x);
        assert_eq!(filter.filter(value), value);
    }
}

#[test]
fn alpha_one_is_pass_through_f64() {
    assert_alpha_one_is_pass_through::<f64>();
}

#[test]
fn alpha_one_is_pass_through_f32() {
    assert_alpha_one_is_pass_through::<f32>();
}

// ---- transient --------------------------------------------------------------

fn assert_step_attenuated_and_monotone<T: Numeric>(tolerance: T) {
    let mut filter = OnePoleLowPass::new(T::from_f64(0.25)).unwrap();
    filter.filter(T::ZERO); // seed at zero, then step the input to one
    let target = T::ONE;
    let mut previous = T::ZERO;
    // Strict checks only while the gap to the target stays above f32 resolution.
    for _ in 0..25 {
        let output = filter.filter(target);
        assert!(output < target); // a low-pass never overshoots a step
        assert!(output > previous); // and climbs monotonically toward it
        previous = output;
    }
    for _ in 0..200 {
        filter.filter(target);
    }
    assert!((filter.value() - target).abs() < tolerance);
}

#[test]
fn step_is_attenuated_and_monotone_f64() {
    assert_step_attenuated_and_monotone(1e-6_f64);
}

#[test]
fn step_is_attenuated_and_monotone_f32() {
    assert_step_attenuated_and_monotone(1e-3_f32);
}

// ---- construction -----------------------------------------------------------

#[test]
fn new_rejects_out_of_range_and_non_finite() {
    assert_eq!(
        OnePoleLowPass::<f64>::new(-0.1),
        Err(ControlError::FilterCoefficientOutOfRange)
    );
    assert_eq!(
        OnePoleLowPass::<f64>::new(1.5),
        Err(ControlError::FilterCoefficientOutOfRange)
    );
    assert_eq!(
        OnePoleLowPass::<f64>::new(f64::NAN),
        Err(ControlError::NonFinite)
    );
}

#[test]
fn from_cutoff_rejects_non_positive_timestep() {
    assert_eq!(
        OnePoleLowPass::<f64>::from_cutoff(5.0, 0.0),
        Err(ControlError::NonPositiveTimestep)
    );
    assert_eq!(
        OnePoleLowPass::<f64>::from_cutoff(5.0, -0.01),
        Err(ControlError::NonPositiveTimestep)
    );
}

#[test]
fn from_cutoff_rejects_negative_cutoff() {
    assert_eq!(
        OnePoleLowPass::<f64>::from_cutoff(-1.0, 0.01),
        Err(ControlError::FilterCoefficientOutOfRange)
    );
}

#[test]
fn reset_clears_state() {
    let mut filter = OnePoleLowPass::new(0.5_f64).unwrap();
    filter.filter(3.0);
    filter.reset();
    assert_eq!(filter.value(), 0.0);
    // The next sample seeds the filter again, so it passes straight through.
    assert_eq!(filter.filter(7.0), 7.0);
}
