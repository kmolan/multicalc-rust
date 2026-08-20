//! Curve-fit tests: exactness on a curve of the fitted order, the centered form's delay, noise
//! rejection against a plain difference, a line having no bend, and constructor rejection.

use multicalc::error::SignalError;
use multicalc::random::{Pcg32, RandomSource};
use multicalc::scalar::Numeric;
use multicalc::signal_processing::SavitzkyGolay;

const TIMESTEP: f64 = 0.001;
const SAMPLES: usize = 200;

/// The test curve `0.5·t² + 2·t + 1`, with its slope and bend.
fn curve(time: f64) -> f64 {
    0.5 * time * time + 2.0 * time + 1.0
}

fn slope(time: f64) -> f64 {
    time + 2.0
}

// ---- exactness on a curve of the fitted order -------------------------------

// A fit with three terms has to reproduce a curve with three terms exactly. This is the identity
// that pins the whole construction: the weights, the scaling by the timestep, and the order the
// window is walked in.
//
// The three quantities need three very different tolerances at f32. The second-derivative weights
// run to about 35000 while the answer is 1.0, so nearly five digits cancel away in the sum and f32
// only carries about seven. A tighter bound would be tuning against this one input rather than
// testing anything — take a bend at f64 if it has to be right.
fn assert_reproduces_a_curve_of_its_own_order<T: Numeric>(
    value_tolerance: T,
    slope_tolerance: T,
    bend_tolerance: T,
) {
    let mut fitted = SavitzkyGolay::<11, 3, T>::latest(T::from_f64(TIMESTEP)).unwrap();
    let mut time = 0.0;
    for sample in 0..SAMPLES {
        time = sample as f64 * TIMESTEP;
        let _ = fitted.filter(T::from_f64(curve(time)));
    }

    assert!((fitted.value() - T::from_f64(curve(time))).abs() < value_tolerance);
    assert!((fitted.first_derivative() - T::from_f64(slope(time))).abs() < slope_tolerance);
    assert!((fitted.second_derivative() - T::ONE).abs() < bend_tolerance);
}

#[test]
fn reproduces_a_curve_of_its_own_order_f64() {
    assert_reproduces_a_curve_of_its_own_order(1e-6_f64, 1e-6, 1e-6);
}

#[test]
fn reproduces_a_curve_of_its_own_order_f32() {
    assert_reproduces_a_curve_of_its_own_order(1e-4_f32, 1e-2, 0.5);
}

// The centered form reads its answer from the middle of the window, so it describes a sample from
// half a window ago. This is what makes that claim testable rather than documentation.
fn assert_centered_reproduces_the_curve_at_its_own_delay<T: Numeric>(tolerance: T) {
    let mut fitted = SavitzkyGolay::<11, 3, T>::centered(T::from_f64(TIMESTEP)).unwrap();
    let mut time = 0.0;
    for sample in 0..SAMPLES {
        time = sample as f64 * TIMESTEP;
        let _ = fitted.filter(T::from_f64(curve(time)));
    }

    // Half of an eleven-sample window at a millisecond a sample.
    let delay = 0.005;
    assert!((fitted.delay() - T::from_f64(delay)).abs() < tolerance);
    assert!((fitted.value() - T::from_f64(curve(time - delay))).abs() < tolerance);
}

#[test]
fn centered_reproduces_the_curve_at_its_own_delay_f64() {
    assert_centered_reproduces_the_curve_at_its_own_delay(1e-6_f64);
}

#[test]
fn centered_reproduces_the_curve_at_its_own_delay_f32() {
    assert_centered_reproduces_the_curve_at_its_own_delay(1e-4_f32);
}

// ---- noise ------------------------------------------------------------------

// Fitting a curve across eleven samples should recover a rate better than subtracting two noisy
// samples does. The wobble is drawn from a fixed seed, so the comparison is the same every run.
#[test]
fn smoothing_beats_a_plain_difference_on_noise() {
    let mut noise = Pcg32::<f64>::new(20260731);
    let mut fitted = SavitzkyGolay::<11, 3, f64>::latest(TIMESTEP).unwrap();

    let mut previous = 0.0;
    let mut worst_fitted = 0.0_f64;
    let mut worst_difference = 0.0_f64;

    for sample in 0..SAMPLES {
        let time = sample as f64 * TIMESTEP;
        let wobble = 0.002 * (noise.next_unit() - 0.5);
        let reading = (2.0 * core::f64::consts::PI * 2.0 * time).sin() + wobble;
        let _ = fitted.filter(reading);

        let true_slope =
            2.0 * core::f64::consts::PI * 2.0 * (2.0 * core::f64::consts::PI * 2.0 * time).cos();
        // Ignore the opening samples, where neither method has a full window yet.
        if sample >= 20 {
            worst_fitted = worst_fitted.max((fitted.first_derivative() - true_slope).abs());
            worst_difference =
                worst_difference.max(((reading - previous) / TIMESTEP - true_slope).abs());
        }
        previous = reading;
    }

    assert!(worst_fitted < worst_difference);
}

// ---- what the fitted order cannot express -----------------------------------

fn assert_a_line_has_no_bend<T: Numeric>() {
    let mut fitted = SavitzkyGolay::<9, 2, T>::latest(T::from_f64(TIMESTEP)).unwrap();
    for sample in 0..50 {
        let time = sample as f64 * TIMESTEP;
        let _ = fitted.filter(T::from_f64(3.0 * time + 1.0));
    }
    // Two terms describe a straight line, so there is no bend to report at all.
    assert_eq!(fitted.second_derivative(), T::ZERO);
}

#[test]
fn a_line_has_no_bend_f64() {
    assert_a_line_has_no_bend::<f64>();
}

#[test]
fn a_line_has_no_bend_f32() {
    assert_a_line_has_no_bend::<f32>();
}

// ---- construction -----------------------------------------------------------

#[test]
fn constructors_reject_unusable_arguments() {
    assert_eq!(
        SavitzkyGolay::<10, 3, f64>::centered(TIMESTEP),
        Err(SignalError::WindowEvenLength)
    );
    assert_eq!(
        SavitzkyGolay::<3, 5, f64>::latest(TIMESTEP),
        Err(SignalError::PolynomialOrderTooHigh)
    );
    assert_eq!(
        SavitzkyGolay::<11, 3, f64>::latest(0.0),
        Err(SignalError::NonPositiveTimestep)
    );
    assert_eq!(
        SavitzkyGolay::<11, 3, f64>::latest(f64::NAN),
        Err(SignalError::NonFinite)
    );
}
