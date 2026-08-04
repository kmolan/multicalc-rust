//! Conditioning-block tests. These blocks are exact maps with nothing to approximate, so they are
//! pinned by hand-computed cases and by properties that have to hold for any input at all, rather
//! than against an outside reference.

use multicalc::error::SignalError;
use multicalc::random::{Pcg32, RandomSource};
use multicalc::scalar::Numeric;
use multicalc::signal_processing::{Deadband, Hysteresis, SlewRateLimiter};

/// Inputs the deadband cases all walk, from inside the band out to either side.
const DEADBAND_INPUTS: [f64; 7] = [0.0, 0.05, 0.1, 0.100001, 0.5, -0.05, -0.5];

// ---- hand-computed cases ----------------------------------------------------

fn assert_plain_deadband_cases<T: Numeric>(tolerance: T) {
    // A value exactly on the threshold counts as inside the band, since the test is on "at most".
    let expected = [0.0, 0.0, 0.0, 0.100001, 0.5, 0.0, -0.5];
    let band = Deadband::plain(T::from_f64(0.1)).unwrap();
    for (input, want) in DEADBAND_INPUTS.iter().zip(expected) {
        let got = band.apply(T::from_f64(*input));
        assert!((got - T::from_f64(want)).abs() < tolerance, "input {input}");
    }
}

#[test]
fn plain_deadband_cases_f64() {
    assert_plain_deadband_cases(1e-12_f64);
}

#[test]
fn plain_deadband_cases_f32() {
    assert_plain_deadband_cases(1e-5_f32);
}

fn assert_recentered_deadband_cases<T: Numeric>(tolerance: T) {
    // Outside the band the value slides back toward zero by the threshold.
    let expected = [0.0, 0.0, 0.0, 0.000001, 0.4, 0.0, -0.4];
    let band = Deadband::recentered(T::from_f64(0.1)).unwrap();
    for (input, want) in DEADBAND_INPUTS.iter().zip(expected) {
        let got = band.apply(T::from_f64(*input));
        assert!((got - T::from_f64(want)).abs() < tolerance, "input {input}");
    }
}

#[test]
fn recentered_deadband_cases_f64() {
    assert_recentered_deadband_cases(1e-12_f64);
}

#[test]
fn recentered_deadband_cases_f32() {
    assert_recentered_deadband_cases(1e-5_f32);
}

fn assert_zero_threshold_passes_everything_through<T: Numeric>() {
    let band = Deadband::plain(T::ZERO).unwrap();
    // Zero itself takes the inside-the-band path, but that path returns zero too, so the answer is
    // the same either way.
    for input in [-2.0, 0.0, 2.0] {
        assert_eq!(band.apply(T::from_f64(input)), T::from_f64(input));
    }
}

#[test]
fn zero_threshold_passes_everything_through_f64() {
    assert_zero_threshold_passes_everything_through::<f64>();
}

#[test]
fn zero_threshold_passes_everything_through_f32() {
    assert_zero_threshold_passes_everything_through::<f32>();
}

fn assert_hysteresis_walk<T: Numeric>() {
    let walk = [0.5, 0.7, 0.65, 0.5, 0.45, 0.3, 0.5, 0.61];
    let expected = [false, true, true, true, true, false, false, true];
    let mut switch = Hysteresis::new(T::from_f64(0.4), T::from_f64(0.6)).unwrap();
    for (input, want) in walk.iter().zip(expected) {
        assert_eq!(switch.update(T::from_f64(*input)), want, "input {input}");
    }
}

#[test]
fn hysteresis_walk_f64() {
    assert_hysteresis_walk::<f64>();
}

#[test]
fn hysteresis_walk_f32() {
    assert_hysteresis_walk::<f32>();
}

fn assert_slew_rate_limiter_ramps<T: Numeric>(tolerance: T) {
    let mut limited =
        SlewRateLimiter::new(T::from_f64(1.0), T::from_f64(2.0), T::from_f64(0.1)).unwrap();

    // The first call goes straight to its target; after that it climbs a tenth per call.
    assert!(limited.filter(T::ZERO).abs() < tolerance);
    for step in 1..=10 {
        let got = limited.filter(T::from_f64(10.0));
        let want = T::from_f64(f64::from(step) * 0.1);
        assert!((got - want).abs() < tolerance, "climbing, step {step}");
    }

    // Turning around, it moves at the faster falling rate.
    for (step, want) in [(1, 0.8), (2, 0.6)] {
        let got = limited.filter(T::from_f64(-10.0));
        assert!(
            (got - T::from_f64(want)).abs() < tolerance,
            "falling, step {step}"
        );
    }
}

#[test]
fn slew_rate_limiter_ramps_f64() {
    assert_slew_rate_limiter_ramps(1e-12_f64);
}

#[test]
fn slew_rate_limiter_ramps_f32() {
    assert_slew_rate_limiter_ramps(1e-5_f32);
}

// ---- properties that hold for any input -------------------------------------

#[test]
fn rate_limited_output_never_moves_faster_than_its_limit() {
    let (rise, fall, dt) = (1.5, 0.5, 0.01);
    let mut limited = SlewRateLimiter::new(rise, fall, dt).unwrap();
    let mut targets = Pcg32::<f64>::new(20260731);

    let mut previous = limited.filter(0.0);
    for _ in 0..1000 {
        let target = 20.0 * targets.next_unit() - 10.0;
        let output = limited.filter(target);
        let step = output - previous;
        assert!(step <= rise * dt + 1e-12, "climbed by {step}");
        assert!(step >= -(fall * dt) - 1e-12, "fell by {step}");
        previous = output;
    }
}

#[test]
fn rate_limited_output_reaches_a_held_target() {
    let (rise, dt, target) = (1.5_f64, 0.01, 3.0);
    let mut limited = SlewRateLimiter::new(rise, 0.5, dt).unwrap();
    let _ = limited.filter(0.0);

    let steps = (target / (rise * dt)).ceil() as usize + 10;
    for _ in 0..steps {
        let output = limited.filter(target);
        assert!(output <= target + 1e-12, "overshot to {output}");
    }
    assert!((limited.value() - target).abs() < 1e-12);
}

#[test]
fn recentered_deadband_is_continuous_at_its_edge() {
    let threshold = 0.1;
    let just_outside = threshold + 1e-9;

    // Leaving the band produces a value near zero rather than a jump to the threshold.
    let recentered = Deadband::recentered(threshold).unwrap();
    assert!(recentered.apply(just_outside).abs() < 1e-8);
    assert!(recentered.apply(-just_outside).abs() < 1e-8);

    // The plain form does the opposite, which is the whole difference between the two.
    let plain = Deadband::plain(threshold).unwrap();
    assert!(plain.apply(just_outside) > 0.09);
}

#[test]
fn deadband_never_grows_a_value() {
    let plain = Deadband::plain(0.1).unwrap();
    let recentered = Deadband::recentered(0.1).unwrap();
    let mut inputs = Pcg32::<f64>::new(20260731);

    for _ in 0..1000 {
        let input = 20.0 * inputs.next_unit() - 10.0;
        assert!(plain.apply(input).abs() <= input.abs() + 1e-12);
        assert!(recentered.apply(input).abs() <= input.abs() + 1e-12);
    }
}

#[test]
fn switch_never_flips_while_inside_its_band() {
    let (lower, upper) = (0.4, 0.6);
    let mut switch = Hysteresis::new(lower, upper).unwrap();
    let mut inputs = Pcg32::<f64>::new(20260731);

    assert!(switch.update(0.7));
    for _ in 0..1000 {
        // Strictly inside the band, so no value here can change the answer.
        let input = lower + (upper - lower) * inputs.next_unit();
        assert!(switch.update(input));
        assert!(switch.is_high());
    }
}

// ---- construction -----------------------------------------------------------

#[test]
fn deadband_rejects_a_negative_or_non_finite_threshold() {
    assert_eq!(
        Deadband::plain(-0.1_f64),
        Err(SignalError::NegativeThreshold)
    );
    assert_eq!(
        Deadband::recentered(-0.1_f64),
        Err(SignalError::NegativeThreshold)
    );
    assert_eq!(Deadband::plain(f64::NAN), Err(SignalError::NonFinite));
}

#[test]
fn hysteresis_rejects_thresholds_out_of_order() {
    assert_eq!(
        Hysteresis::new(0.6_f64, 0.4),
        Err(SignalError::ThresholdsOutOfOrder)
    );
    assert_eq!(
        Hysteresis::new(0.5_f64, 0.5),
        Err(SignalError::ThresholdsOutOfOrder)
    );
}

#[test]
fn slew_rate_limiter_rejects_unusable_rates_and_timesteps() {
    assert_eq!(
        SlewRateLimiter::new(1.0_f64, 0.0, 0.1),
        Err(SignalError::NonPositiveRate)
    );
    assert_eq!(
        SlewRateLimiter::new(-1.0_f64, 1.0, 0.1),
        Err(SignalError::NonPositiveRate)
    );
    assert_eq!(
        SlewRateLimiter::new(1.0_f64, 1.0, 0.0),
        Err(SignalError::NonPositiveTimestep)
    );
    assert_eq!(
        SlewRateLimiter::new(1.0_f64, 1.0, f64::NAN),
        Err(SignalError::NonFinite)
    );
}
