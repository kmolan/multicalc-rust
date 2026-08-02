#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks whole filtering jobs against scipy.signal goldens: a drone's gyro conditioning chain,
//! vibration tracking, drift removal, an encoder read for rate and acceleration, and despiking a
//! range trace.

use multicalc::linear_algebra::Vector;
use multicalc::signal_processing::{
    Biquad, BiquadCascade, BiquadCoefficients, MovingAverage, RunningMedian, SavitzkyGolay,
    harmonic_notch_coefficients,
};
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

/// A magnitude this small is a null: the filter removes that frequency completely. The phase
/// there is whatever the leftover rounding in a near-zero response gives, so the two
/// implementations disagree wildly while both being right, and the phase comparison is skipped.
/// The magnitude is still compared, which checks the null is as deep as SciPy's.
const NULL_MAGNITUDE: f64 = 1e-6;

/// The difference between two phases, brought into a single turn.
///
/// A phase only means anything up to whole turns, and the two sides count them differently:
/// SciPy always reports within one turn of zero, while a chain here adds its sections up and runs
/// past that on purpose. Comparing the difference within a turn is the comparison that holds for
/// both.
#[must_use]
fn phase_difference(got: f64, want: f64) -> f64 {
    let turn = 2.0 * core::f64::consts::PI;
    let mut difference = (got - want) % turn;
    if difference > core::f64::consts::PI {
        difference -= turn;
    } else if difference < -core::f64::consts::PI {
        difference += turn;
    }
    difference
}

fn assert_response(magnitude_at: impl Fn(f64) -> f64, phase_at: impl Fn(f64) -> f64, fx: &Fixture) {
    let probes = fx.inputs["probe_hz"].as_vector();
    let magnitudes = fx.expected["magnitude"].as_vector();
    let phases = fx.expected["phase"].as_vector();
    let t = fx.tolerances.f64;
    let case = &fx.case;

    for (index, &frequency_hz) in probes.iter().enumerate() {
        let (want_magnitude, want_phase) = (magnitudes[index], phases[index]);
        let got_magnitude = magnitude_at(frequency_hz);
        assert!(
            close(got_magnitude, want_magnitude, t),
            "{case}: magnitude at {frequency_hz} Hz: got {got_magnitude}, want {want_magnitude}, tol {t:?}"
        );

        if want_magnitude < NULL_MAGNITUDE {
            continue;
        }
        let got_phase = phase_at(frequency_hz);
        let shifted = want_phase + phase_difference(got_phase, want_phase);
        assert!(
            close(shifted, want_phase, t),
            "{case}: phase at {frequency_hz} Hz: got {got_phase}, want {want_phase}, tol {t:?}"
        );
    }
}

fn assert_filtered_output(mut filter: impl FnMut(f64) -> f64, fx: &Fixture) {
    let input = fx.inputs["input"].as_vector();
    let want = fx.expected["output"].as_vector();
    let t = fx.tolerances.f64;
    let case = &fx.case;

    for (index, &sample) in input.iter().enumerate() {
        let got = filter(sample);
        assert!(
            close(got, want[index], t),
            "{case}: output[{index}]: got {got}, want {}, tol {t:?}",
            want[index]
        );
    }
}

/// A drone's gyro chain: notches on the motor's frequency and its first two harmonics, then a
/// fourth-order low-pass, all in one pass over the reading.
#[test]
fn gyro_conditioning_chain() {
    for fx in load_dir("signal_processing") {
        if fx.inputs["kind"].as_str() != "gyro_chain" {
            continue;
        }
        let dt = fx.inputs["dt"].as_scalar();
        let fundamental_hz = fx.inputs["fundamental_hz"].as_scalar();
        let quality_factor = fx.inputs["quality_factor"].as_scalar();
        let low_pass_hz = fx.inputs["low_pass_hz"].as_scalar();
        let low_pass_quality_factors = fx.inputs["low_pass_quality_factors"].as_vector();
        assert_eq!(fx.inputs["notch_sections"].as_int(), 3, "notch sections");
        let t = fx.tolerances.f64;

        let notches =
            harmonic_notch_coefficients::<3, f64>(fundamental_hz, quality_factor, dt).unwrap();
        let low_pass: Vec<_> = low_pass_quality_factors
            .iter()
            .map(|&section_quality| {
                BiquadCoefficients::low_pass(low_pass_hz, section_quality, dt).unwrap()
            })
            .collect();
        let chain = [notches[0], notches[1], notches[2], low_pass[0], low_pass[1]];

        // The notch weights are compared directly, since a notch's sharpness means something
        // different from a low-pass's and the two are built from different formulas.
        assert_vector(
            &Vector::new(notches[0].feed_forward()),
            &fx.expected["notch_feed_forward"],
            t,
            "notch_feed_forward",
        );
        assert_vector(
            &Vector::new(notches[0].feedback()),
            &fx.expected["notch_feedback"],
            t,
            "notch_feedback",
        );

        let cascade = BiquadCascade::new(chain);
        assert_response(
            |frequency_hz| cascade.magnitude_at(frequency_hz),
            |frequency_hz| cascade.phase_at(frequency_hz),
            &fx,
        );

        let mut running = BiquadCascade::new(chain);
        assert_filtered_output(|sample| running.filter(sample), &fx);
    }
}

/// One designed section doing its own job: isolating a vibration to track it, or taking a slow
/// bias out from under a reading.
#[test]
fn single_section_jobs() {
    for fx in load_dir("signal_processing") {
        if fx.inputs["kind"].as_str() != "biquad_section" {
            continue;
        }
        let frequency_hz = fx.inputs["frequency_hz"].as_scalar();
        let quality_factor = fx.inputs["quality_factor"].as_scalar();
        let dt = fx.inputs["dt"].as_scalar();
        let design = fx.inputs["design"].as_str();

        let coefficients = match design {
            "low_pass" => BiquadCoefficients::low_pass(frequency_hz, quality_factor, dt),
            "high_pass" => BiquadCoefficients::high_pass(frequency_hz, quality_factor, dt),
            "band_pass" => BiquadCoefficients::band_pass(frequency_hz, quality_factor, dt),
            "notch" => BiquadCoefficients::notch(frequency_hz, quality_factor, dt),
            design => panic!("unregistered design {design}"),
        }
        .unwrap();

        let t = fx.tolerances.f64;
        assert_vector(
            &Vector::new(coefficients.feed_forward()),
            &fx.expected["feed_forward"],
            t,
            "feed_forward",
        );
        assert_vector(
            &Vector::new(coefficients.feedback()),
            &fx.expected["feedback"],
            t,
            "feedback",
        );

        assert_response(
            |frequency_hz| coefficients.magnitude_at(frequency_hz),
            |frequency_hz| coefficients.phase_at(frequency_hz),
            &fx,
        );

        let mut filter = Biquad::new(coefficients);
        assert_filtered_output(|sample| filter.filter(sample), &fx);
    }
}

/// Recovers one of the curve fit's three weight rows by feeding it impulses.
///
/// The weights are private, and no accessor was added to the crate just for this: feeding a 1 at
/// each window position in turn and reading the output back gives the row directly, since the
/// output is the weighted sum of the window.
///
/// The leading `0.0` is required. The filter seeds its whole window from its first sample, so
/// starting straight in with the impulse at position 0 would fill every slot with 1 — not an
/// impulse at all — and only that one row would come out wrong. One zero first seeds the window to
/// zeros, and the `WINDOW` samples after it land in positions 0 through `WINDOW - 1` in order.
fn recover_weights<const WINDOW: usize, const TERMS: usize>(
    build: impl Fn() -> SavitzkyGolay<WINDOW, TERMS, f64>,
    read: impl Fn(&SavitzkyGolay<WINDOW, TERMS, f64>) -> f64,
) -> Vector<WINDOW> {
    Vector::from_fn(|position| {
        let mut filter = build();
        let _ = filter.filter(0.0);
        for sample in 0..WINDOW {
            let _ = filter.filter(if sample == position { 1.0 } else { 0.0 });
        }
        read(&filter)
    })
}

/// A noisy encoder giving position, rate, and acceleration — read at the newest sample for a live
/// loop, and at the middle of the window where the extra smoothing is worth the delay.
#[test]
fn encoder_curve_fit() {
    for fx in load_dir("signal_processing") {
        if fx.inputs["kind"].as_str() != "savitzky_golay" {
            continue;
        }
        assert_eq!(fx.inputs["window"].as_int(), 11, "window");
        assert_eq!(
            fx.inputs["polynomial_terms"].as_int(),
            3,
            "polynomial terms"
        );
        let dt = fx.inputs["dt"].as_scalar();
        let t = fx.tolerances.f64;

        for position in ["latest", "centered"] {
            let build = || match position {
                "latest" => SavitzkyGolay::<11, 3, f64>::latest(dt).unwrap(),
                _ => SavitzkyGolay::<11, 3, f64>::centered(dt).unwrap(),
            };

            for (row, read) in [
                ("smoothing_weights", SavitzkyGolay::value as fn(&_) -> f64),
                ("first_derivative_weights", SavitzkyGolay::first_derivative),
                (
                    "second_derivative_weights",
                    SavitzkyGolay::second_derivative,
                ),
            ] {
                let key = format!("{position}_{row}");
                assert_vector(&recover_weights(build, read), &fx.expected[&key], t, &key);
            }

            let mut filter = build();
            for sample in fx.inputs["input"].as_vector() {
                let _ = filter.filter(sample);
            }
            for (quantity, got) in [
                ("value", filter.value()),
                ("first_derivative", filter.first_derivative()),
                ("second_derivative", filter.second_derivative()),
            ] {
                let key = format!("{position}_{quantity}");
                assert_scalar(got, &fx.expected[&key], t, &key);
            }
        }
    }
}

/// A range trace with three bad returns, through a median and through a mean. The median drops
/// them outright; the mean carries a fifth of each into its output, which is the whole reason both
/// exist.
#[test]
fn lidar_despiking() {
    for fx in load_dir("signal_processing") {
        if fx.inputs["kind"].as_str() != "window_filters" {
            continue;
        }
        assert_eq!(fx.inputs["window"].as_int(), 5, "window");
        let input = fx.inputs["input"].as_vector();
        let first_compared_index = fx.inputs["first_compared_index"].as_int() as usize;
        let t = fx.tolerances.f64;
        let case = &fx.case;

        let mut median = RunningMedian::<5, f64>::new().unwrap();
        let mut mean = MovingAverage::<5, f64>::new().unwrap();
        let want_median = fx.expected["median_output"].as_vector();
        let want_mean = fx.expected["mean_output"].as_vector();

        for (index, &sample) in input.iter().enumerate() {
            let (got_median, got_mean) = (median.filter(sample), mean.filter(sample));
            // The reference centres its window on each sample while these filters cover the
            // samples up to and including it, so the opening samples have no counterpart.
            if index < first_compared_index {
                continue;
            }
            let compared = index - first_compared_index;
            assert!(
                close(got_median, want_median[compared], t),
                "{case}: median[{index}]: got {got_median}, want {}, tol {t:?}",
                want_median[compared]
            );
            assert!(
                close(got_mean, want_mean[compared], t),
                "{case}: mean[{index}]: got {got_mean}, want {}, tol {t:?}",
                want_mean[compared]
            );
        }
    }
}
