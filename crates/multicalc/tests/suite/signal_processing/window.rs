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

// ---- handling non-finite signal ---------------------------------------------
#[test]
fn non_finite_input_spoils_the_average_for_one_window() {
    // The sample is flushed out rather than latched, so the damage is bounded by the window
    let mut running = MovingAverage::<4, f64>::new().unwrap();

    let _ = running.filter(1.0);
    assert!(running.value().is_finite());

    assert!(running.filter(f64::NAN).is_nan());
    for call in 1..4 {
        assert!(
            running.filter(1.0).is_nan(),
            "call {call} after the bad sample"
        );
    }

    // The fourth sample pushes it out of the window
    assert!((running.filter(1.0) - 1.0).abs() < 1e-12);
}

#[test]
fn non_finite_first_sample_spoils_the_whole_window() {
    // The first sample fills every slot, and still takes exactly one window to flush out
    let mut running = MovingAverage::<4, f64>::new().unwrap();

    assert!(running.filter(f64::NAN).is_nan());
    for call in 1..4 {
        assert!(
            running.filter(1.0).is_nan(),
            "call {call} after the bad sample"
        );
    }

    assert!((running.filter(1.0) - 1.0).abs() < 1e-12);
}

#[test]
fn opposite_infinities_in_one_window_make_a_nan() {
    // Neither sample is a NaN, but the running total cancels them into one
    let mut running = MovingAverage::<4, f64>::new().unwrap();

    let _ = running.filter(1.0);
    assert!(running.filter(f64::INFINITY).is_infinite());
    assert!(running.filter(f64::NEG_INFINITY).is_nan());
}

#[test]
fn filter_checked_protects_the_average() {
    let mut running = MovingAverage::<4, f64>::new().unwrap();

    let _ = running.filter(1.0);
    let running_snapshot = running;
    assert!(running.value().is_finite());

    for signal in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let output = running.filter_checked(signal);

        assert_eq!(output, Err(SignalError::NonFinite));
        assert_eq!(running, running_snapshot);
    }
}

#[test]
fn filter_checked_protects_an_unseeded_average() {
    // Nothing has filled the window yet, so a rejected sample must not seed it either
    let mut running = MovingAverage::<4, f64>::new().unwrap();
    let running_snapshot = running;

    assert_eq!(
        running.filter_checked(f64::NAN),
        Err(SignalError::NonFinite)
    );
    assert_eq!(running, running_snapshot);

    // The next good sample still fills the whole window, so it comes straight back out
    assert!((running.filter_checked(2.0).unwrap() - 2.0).abs() < 1e-12);
}

// ---- the median and non-finite samples --------------------------------------
#[test]
fn a_nan_moves_the_median_without_showing_it() {
    // The sort cannot move elements past a NaN, so the middle slot stops holding the middle value
    let mut control = RunningMedian::<5, f64>::new().unwrap();
    for reading in [3.0, 50.0, 1.0, 2.0, 5.0] {
        let _ = control.filter(reading);
    }
    assert_eq!(control.value(), 3.0);

    let mut spoiled = RunningMedian::<5, f64>::new().unwrap();
    for reading in [3.0, f64::NAN, 1.0, 2.0, 5.0] {
        let _ = spoiled.filter(reading);
    }

    // The answer moved, and it is a perfectly ordinary finite number
    assert_eq!(spoiled.value(), 2.0);
    assert!(spoiled.value().is_finite());
    assert_ne!(spoiled.value(), control.value());
}

#[test]
fn the_median_flushes_a_nan_out_with_the_window() {
    // Like the average, the damage is bounded by the window rather than latched
    let mut spoiled = RunningMedian::<5, f64>::new().unwrap();
    let mut clean = RunningMedian::<5, f64>::new().unwrap();
    for reading in [3.0, 50.0, 1.0, 2.0, 5.0] {
        let _ = clean.filter(reading);
    }
    for reading in [3.0, f64::NAN, 1.0, 2.0, 5.0] {
        let _ = spoiled.filter(reading);
    }
    assert_ne!(spoiled.value(), clean.value());

    // Once the window has moved past the NaN the two agree again
    for reading in [7.0, 8.0, 9.0, 10.0, 11.0] {
        let _ = spoiled.filter(reading);
        let _ = clean.filter(reading);
    }
    assert_eq!(spoiled.value(), clean.value());
}

#[test]
fn the_median_rejects_an_infinity_as_an_ordinary_outlier() {
    // The sort is a total order over infinities, and the median does no arithmetic, so no NaN can
    // be manufactured the way the average manufactures one
    let mut one_high = RunningMedian::<5, f64>::new().unwrap();
    for reading in [1.0, 1.1, 0.9, f64::INFINITY, 1.05] {
        let _ = one_high.filter(reading);
    }
    assert_eq!(one_high.value(), 1.05);

    let mut one_low = RunningMedian::<5, f64>::new().unwrap();
    for reading in [1.0, 1.1, 0.9, f64::NEG_INFINITY, 1.05] {
        let _ = one_low.filter(reading);
    }
    assert_eq!(one_low.value(), 1.0);

    // Both at once still leaves a correct finite middle value
    let mut both = RunningMedian::<5, f64>::new().unwrap();
    for reading in [1.0, f64::INFINITY, 0.9, f64::NEG_INFINITY, 1.05] {
        let _ = both.filter(reading);
    }
    assert_eq!(both.value(), 1.0);
}

#[test]
fn a_majority_of_infinities_makes_the_median_infinite() {
    // Not corruption: with three of five readings infinite, the middle one really is infinite
    let mut running = RunningMedian::<5, f64>::new().unwrap();
    for reading in [1.0, f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.05] {
        let _ = running.filter(reading);
    }
    assert!(running.value().is_infinite());
}

#[test]
fn filter_checked_refuses_a_nan_but_admits_an_infinity() {
    let mut running = RunningMedian::<5, f64>::new().unwrap();

    let _ = running.filter(1.0);
    let running_snapshot = running;

    assert_eq!(
        running.filter_checked(f64::NAN),
        Err(SignalError::NonFinite)
    );
    assert_eq!(running, running_snapshot);

    // Infinities are admitted, and a single one does not move the answer
    for signal in [f64::INFINITY, f64::NEG_INFINITY] {
        assert!(running.filter_checked(signal).is_ok());
    }
    assert_eq!(running.value(), 1.0);
}

#[test]
fn filter_checked_protects_an_unseeded_median() {
    // A rejected sample must not seed the window either
    let mut running = RunningMedian::<5, f64>::new().unwrap();
    let running_snapshot = running;

    assert_eq!(
        running.filter_checked(f64::NAN),
        Err(SignalError::NonFinite)
    );
    assert_eq!(running, running_snapshot);

    // The next good sample still fills the whole window
    assert_eq!(running.filter_checked(2.0), Ok(2.0));
}
