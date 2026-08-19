//! Window-filter tests: what a mean and a median do to a constant, a known window, a spike, and a
//! ramp, plus constructor rejection.

use multicalc::error::SignalError;
use multicalc::scalar::Numeric;
use multicalc::signal_processing::{MovingAverage, RunningMedian};

// ---- the moving average -----------------------------------------------------

fn assert_average_of_a_constant_is_that_constant<T: Numeric>(tolerance: T) {
    let mut average = MovingAverage::<8, T>::new().unwrap();
    let input = T::from_f64(3.0);
    for _ in 0..20 {
        let _ = average.filter(input);
    }
    assert!((average.value() - input).abs() < tolerance);
}

#[test]
fn average_of_a_constant_is_that_constant_f64() {
    assert_average_of_a_constant_is_that_constant(1e-12_f64);
}

#[test]
fn average_of_a_constant_is_that_constant_f32() {
    assert_average_of_a_constant_is_that_constant(1e-5_f32);
}

#[test]
fn average_is_exact_on_a_known_window() {
    let mut average = MovingAverage::<4, f64>::new().unwrap();
    let mut output = 0.0;
    for sample in [1.0, 2.0, 3.0, 4.0, 5.0] {
        output = average.filter(sample);
    }
    // The window holds the last four samples: 2, 3, 4, 5.
    assert!((output - 3.5).abs() < 1e-12);
}

// Adding the window up fresh each time passes this; keeping a running total does not, because the
// total loses the low bits of every value added into it.
#[test]
fn average_does_not_drift_at_single_precision() {
    let mut average = MovingAverage::<16, f32>::new().unwrap();
    for _ in 0..100_000 {
        let _ = average.filter(1.0e6);
    }
    for _ in 0..16 {
        let _ = average.filter(0.0);
    }
    assert!(average.value().abs() < 1.0);
}

// ---- the running median -----------------------------------------------------

fn assert_median_ignores_one_spike<T: Numeric>() {
    let mut median = RunningMedian::<5, T>::new().unwrap();
    let mut output = T::ZERO;
    for reading in [1.0, 1.1, 0.9, 1000.0, 1.05] {
        output = median.filter(T::from_f64(reading));
    }
    assert_eq!(output, T::from_f64(1.05));
}

#[test]
fn median_ignores_one_spike_f64() {
    assert_median_ignores_one_spike::<f64>();
}

#[test]
fn median_ignores_one_spike_f32() {
    assert_median_ignores_one_spike::<f32>();
}

fn assert_median_of_a_constant_is_that_constant<T: Numeric>() {
    let mut median = RunningMedian::<7, T>::new().unwrap();
    let input = T::from_f64(2.5);
    for _ in 0..10 {
        let _ = median.filter(input);
    }
    assert_eq!(median.value(), input);
}

#[test]
fn median_of_a_constant_is_that_constant_f64() {
    assert_median_of_a_constant_is_that_constant::<f64>();
}

#[test]
fn median_of_a_constant_is_that_constant_f32() {
    assert_median_of_a_constant_is_that_constant::<f32>();
}

// A median reports the middle of its window, so on a steadily climbing input it sits half a window
// behind. That lag is the price of shrugging off spikes.
#[test]
fn median_tracks_a_ramp_with_a_lag() {
    let mut median = RunningMedian::<5, f64>::new().unwrap();
    let mut output = 0.0;
    for sample in 0..20 {
        output = median.filter(f64::from(sample));
    }
    assert_eq!(output, 17.0);
}

// A NaN among otherwise-finite readings used to defeat the insertion sort outright, since every
// comparison against it comes back false and leaves the window unsorted around it. It should
// instead sort to one end, so the reported median is still the median of the finite readings.
//
// The finite readings are 1, 2, 2, 3, so their two middle values are both 2 — the median is 2
// however it is picked, sidestepping any ambiguity about which of an even count of finite
// readings counts as "the middle one".
#[test]
fn median_ignores_a_nan_among_finite_readings() {
    let mut median = RunningMedian::<5, f64>::new().unwrap();
    let mut output = 0.0;
    for reading in [1.0, 2.0, f64::NAN, 3.0, 2.0] {
        output = median.filter(reading);
    }
    assert_eq!(output, 2.0);
}

// ---- construction -----------------------------------------------------------

#[test]
fn constructors_reject_unusable_windows() {
    assert_eq!(
        MovingAverage::<0, f64>::new(),
        Err(SignalError::WindowTooShort)
    );
    assert_eq!(
        RunningMedian::<0, f64>::new(),
        Err(SignalError::WindowTooShort)
    );
    assert_eq!(
        RunningMedian::<6, f64>::new(),
        Err(SignalError::WindowEvenLength)
    );
}
