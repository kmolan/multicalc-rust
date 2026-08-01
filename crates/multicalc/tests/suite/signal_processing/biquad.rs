//! Second-order filter tests: steady state, what each design removes, swapping weights mid-run,
//! settling, the response queries, stability, and constructor rejection, at f32 and f64.

use multicalc::error::SignalError;
use multicalc::scalar::Numeric;
use multicalc::signal_processing::{Biquad, BiquadCoefficients};

/// The sharpness that gives the flattest low-pass.
const FLATTEST: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// One sample of a sine at `frequency_hz`, sampled every millisecond.
fn sine_sample<T: Numeric>(frequency_hz: f64, sample: usize) -> T {
    T::from_f64((2.0 * core::f64::consts::PI * frequency_hz * sample as f64 / 1000.0).sin())
}

// ---- steady state -----------------------------------------------------------

fn assert_low_pass_passes_a_constant<T: Numeric>(tolerance: T) {
    let coefficients =
        BiquadCoefficients::low_pass(T::from_f64(50.0), T::from_f64(FLATTEST), T::from_f64(0.001))
            .unwrap();
    let mut filter = Biquad::new(coefficients);
    let input = T::from_f64(5.0);
    for _ in 0..500 {
        let _ = filter.filter(input);
    }
    assert!((filter.value() - input).abs() < tolerance);
}

#[test]
fn low_pass_passes_a_constant_f64() {
    assert_low_pass_passes_a_constant(1e-9_f64);
}

#[test]
fn low_pass_passes_a_constant_f32() {
    assert_low_pass_passes_a_constant(1e-3_f32);
}

fn assert_high_pass_blocks_a_constant<T: Numeric>(tolerance: T) {
    let coefficients =
        BiquadCoefficients::high_pass(T::from_f64(50.0), T::from_f64(FLATTEST), T::from_f64(0.001))
            .unwrap();
    let mut filter = Biquad::new(coefficients);
    for _ in 0..500 {
        let _ = filter.filter(T::from_f64(5.0));
    }
    assert!(filter.value().abs() < tolerance);
}

#[test]
fn high_pass_blocks_a_constant_f64() {
    assert_high_pass_blocks_a_constant(1e-9_f64);
}

#[test]
fn high_pass_blocks_a_constant_f32() {
    assert_high_pass_blocks_a_constant(1e-3_f32);
}

// ---- what a notch removes ---------------------------------------------------

fn assert_notch_removes_its_own_frequency<T: Numeric>() {
    let coefficients =
        BiquadCoefficients::notch(T::from_f64(180.0), T::from_f64(4.0), T::from_f64(0.001))
            .unwrap();
    let mut filter = Biquad::new(coefficients);
    for sample in 0..2000 {
        let output = filter.filter(sine_sample::<T>(180.0, sample));
        if sample >= 1500 {
            assert!(output.abs() < T::from_f64(0.05));
        }
    }
}

#[test]
fn notch_removes_its_own_frequency_f64() {
    assert_notch_removes_its_own_frequency::<f64>();
}

#[test]
fn notch_removes_its_own_frequency_f32() {
    assert_notch_removes_its_own_frequency::<f32>();
}

fn assert_notch_leaves_a_distant_frequency<T: Numeric>() {
    let coefficients =
        BiquadCoefficients::notch(T::from_f64(180.0), T::from_f64(4.0), T::from_f64(0.001))
            .unwrap();
    let mut filter = Biquad::new(coefficients);
    let mut largest = T::ZERO;
    for sample in 0..2000 {
        let output = filter.filter(sine_sample::<T>(20.0, sample));
        if sample >= 1500 && output.abs() > largest {
            largest = output.abs();
        }
    }
    assert!(largest > T::from_f64(0.9));
}

#[test]
fn notch_leaves_a_distant_frequency_f64() {
    assert_notch_leaves_a_distant_frequency::<f64>();
}

#[test]
fn notch_leaves_a_distant_frequency_f32() {
    assert_notch_leaves_a_distant_frequency::<f32>();
}

// ---- swapping weights mid-run -----------------------------------------------

// Driven by a constant rather than an oscillation on purpose. A 180 Hz sine sampled every
// millisecond is exactly zero at sample 1000, and a settled notch's output is near zero too, so an
// oscillating version of this test would pass whatever the swap did.
#[test]
fn swapping_weights_keeps_the_output_continuous() {
    let first = BiquadCoefficients::notch(180.0_f64, 4.0, 0.001).unwrap();
    let second = BiquadCoefficients::notch(210.0_f64, 4.0, 0.001).unwrap();

    // A notch passes a constant untouched, so this settles on 1.
    let mut running = Biquad::new(first);
    for _ in 0..1000 {
        let _ = running.filter(1.0);
    }
    running.set_coefficients(second);
    assert!((running.filter(1.0) - 1.0).abs() < 0.03);

    // The same filter built from scratch has no memory, which is what makes the check above mean
    // something.
    let mut fresh = Biquad::new(second);
    assert!((fresh.filter(1.0) - 1.0).abs() > 0.05);
}

// ---- settling ---------------------------------------------------------------

fn assert_settling_removes_the_transient<T: Numeric>(tolerance: T) {
    let coefficients =
        BiquadCoefficients::low_pass(T::from_f64(50.0), T::from_f64(FLATTEST), T::from_f64(0.001))
            .unwrap();
    let value = T::from_f64(3.0);

    let mut settled = Biquad::new(coefficients);
    settled.settle_to(value);
    assert!((settled.filter(value) - value).abs() < tolerance);

    // Starting at rest, the same filter is still well short of the input on its first sample.
    let mut cold = Biquad::new(coefficients);
    assert!(cold.filter(value) < T::ONE);
}

#[test]
fn settling_removes_the_transient_f64() {
    assert_settling_removes_the_transient(1e-12_f64);
}

#[test]
fn settling_removes_the_transient_f32() {
    assert_settling_removes_the_transient(1e-4_f32);
}

// ---- the response queries ---------------------------------------------------

// Ties the reported response to the filter that actually runs.
#[test]
fn reported_magnitude_matches_a_long_run() {
    let coefficients = BiquadCoefficients::low_pass(50.0_f64, FLATTEST, 0.001).unwrap();
    let mut filter = Biquad::new(coefficients);
    let mut largest = 0.0;
    for sample in 0..4000 {
        let output = filter.filter(sine_sample::<f64>(30.0, sample));
        if sample >= 3000 && output.abs() > largest {
            largest = output.abs();
        }
    }
    assert!((largest - coefficients.magnitude_at(30.0)).abs() < 0.02);
}

#[test]
fn stability_follows_the_feedback_weights() {
    // No feedback at all, so nothing can grow.
    assert!(
        BiquadCoefficients::new([1.0_f64, 0.0, 0.0], [0.0, 0.0], 0.001)
            .unwrap()
            .is_stable()
    );
    // More than the whole of the previous output fed back.
    assert!(
        !BiquadCoefficients::new([1.0_f64, 0.0, 0.0], [0.0, 1.5], 0.001)
            .unwrap()
            .is_stable()
    );
    // Heavy feedback that still settles, which is where a sharp filter sits.
    assert!(
        BiquadCoefficients::new([1.0_f64, 0.0, 0.0], [-1.98, 0.99], 0.001)
            .unwrap()
            .is_stable()
    );
    // Just past that, the weights hold a ringing that never dies away.
    assert!(
        !BiquadCoefficients::new([1.0_f64, 0.0, 0.0], [-1.99, 0.99], 0.001)
            .unwrap()
            .is_stable()
    );
}

// ---- construction -----------------------------------------------------------

#[test]
fn designs_reject_a_non_positive_timestep() {
    assert_eq!(
        BiquadCoefficients::low_pass(50.0_f64, 0.7, 0.0),
        Err(SignalError::NonPositiveTimestep)
    );
    assert_eq!(
        BiquadCoefficients::high_pass(50.0_f64, 0.7, 0.0),
        Err(SignalError::NonPositiveTimestep)
    );
    assert_eq!(
        BiquadCoefficients::band_pass(50.0_f64, 0.7, 0.0),
        Err(SignalError::NonPositiveTimestep)
    );
    assert_eq!(
        BiquadCoefficients::notch(50.0_f64, 0.7, 0.0),
        Err(SignalError::NonPositiveTimestep)
    );
}

#[test]
fn designs_reject_a_non_positive_quality_factor() {
    assert_eq!(
        BiquadCoefficients::low_pass(50.0_f64, 0.0, 0.001),
        Err(SignalError::NonPositiveQualityFactor)
    );
    assert_eq!(
        BiquadCoefficients::high_pass(50.0_f64, 0.0, 0.001),
        Err(SignalError::NonPositiveQualityFactor)
    );
    assert_eq!(
        BiquadCoefficients::band_pass(50.0_f64, 0.0, 0.001),
        Err(SignalError::NonPositiveQualityFactor)
    );
    assert_eq!(
        BiquadCoefficients::notch(50.0_f64, 0.0, 0.001),
        Err(SignalError::NonPositiveQualityFactor)
    );
}

#[test]
fn designs_reject_a_frequency_outside_the_usable_range() {
    for frequency_hz in [0.0_f64, 500.0] {
        assert_eq!(
            BiquadCoefficients::low_pass(frequency_hz, 0.7, 0.001),
            Err(SignalError::FrequencyOutOfRange)
        );
        assert_eq!(
            BiquadCoefficients::high_pass(frequency_hz, 0.7, 0.001),
            Err(SignalError::FrequencyOutOfRange)
        );
        assert_eq!(
            BiquadCoefficients::band_pass(frequency_hz, 0.7, 0.001),
            Err(SignalError::FrequencyOutOfRange)
        );
        assert_eq!(
            BiquadCoefficients::notch(frequency_hz, 0.7, 0.001),
            Err(SignalError::FrequencyOutOfRange)
        );
    }
}

#[test]
fn designs_reject_a_non_finite_frequency() {
    assert_eq!(
        BiquadCoefficients::low_pass(f64::NAN, 0.7, 0.001),
        Err(SignalError::NonFinite)
    );
    assert_eq!(
        BiquadCoefficients::high_pass(f64::NAN, 0.7, 0.001),
        Err(SignalError::NonFinite)
    );
    assert_eq!(
        BiquadCoefficients::band_pass(f64::NAN, 0.7, 0.001),
        Err(SignalError::NonFinite)
    );
    assert_eq!(
        BiquadCoefficients::notch(f64::NAN, 0.7, 0.001),
        Err(SignalError::NonFinite)
    );
}
