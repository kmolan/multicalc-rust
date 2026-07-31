#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks biquad design, frequency response, and filtered output against scipy.signal goldens.

use multicalc::linear_algebra::Vector;
use multicalc::signal_processing::{
    Biquad, BiquadCascade, BiquadCoefficients, harmonic_notch_coefficients,
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
/// SciPy always reports within one turn of zero, while a cascade here adds its sections up and
/// runs past that on purpose. Comparing the difference within a turn is the comparison that holds
/// for both.
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

#[test]
fn biquad_sections() {
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

/// A cascade built from the fixture's own inputs, checked by what the chain does rather than by
/// its section weights: section order is not part of the filter, and the weights themselves are
/// already pinned by the single-section cases.
fn run_cascade<const SECTIONS: usize>(
    coefficients: [BiquadCoefficients<f64>; SECTIONS],
    fx: &Fixture,
) {
    let cascade = BiquadCascade::new(coefficients);
    assert_response(
        |frequency_hz| cascade.magnitude_at(frequency_hz),
        |frequency_hz| cascade.phase_at(frequency_hz),
        fx,
    );

    let mut running = BiquadCascade::new(coefficients);
    assert_filtered_output(|sample| running.filter(sample), fx);
}

#[test]
fn biquad_cascades() {
    for fx in load_dir("signal_processing") {
        if fx.inputs["kind"].as_str() != "biquad_cascade" {
            continue;
        }
        let frequency_hz = fx.inputs["frequency_hz"].as_scalar();
        let dt = fx.inputs["dt"].as_scalar();
        let sections = fx.inputs["sections"].as_int();

        match (fx.inputs["design"].as_str(), sections) {
            ("low_pass", 2) => {
                let quality_factors = fx.inputs["quality_factors"].as_vector();
                let designed = [
                    BiquadCoefficients::low_pass(frequency_hz, quality_factors[0], dt).unwrap(),
                    BiquadCoefficients::low_pass(frequency_hz, quality_factors[1], dt).unwrap(),
                ];
                run_cascade::<2>(designed, &fx);
            }
            ("harmonic_notch", 3) => {
                let quality_factor = fx.inputs["quality_factor"].as_scalar();
                let designed =
                    harmonic_notch_coefficients::<3, f64>(frequency_hz, quality_factor, dt)
                        .unwrap();
                run_cascade::<3>(designed, &fx);
            }
            (design, sections) => panic!("unregistered cascade {design} with {sections} sections"),
        }
    }
}
