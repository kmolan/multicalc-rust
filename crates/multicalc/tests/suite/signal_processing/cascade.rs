//! Cascade and multi-channel tests: sharper roll-off, magnitude as a product, phase as a sum past
//! half a turn, harmonic notches, section indexing, and per-channel independence.

use multicalc::error::SignalError;
use multicalc::linear_algebra::Vector;
use multicalc::scalar::Numeric;
use multicalc::signal_processing::{
    Biquad, BiquadCascade, BiquadCoefficients, MultiChannelBiquad, harmonic_notch_coefficients,
};

/// The sharpness that gives the flattest low-pass.
const FLATTEST: f64 = core::f64::consts::FRAC_1_SQRT_2;

// ---- sections in series -----------------------------------------------------

fn assert_two_sections_cut_deeper_than_one<T: Numeric>(tolerance: T) {
    let section =
        BiquadCoefficients::low_pass(T::from_f64(50.0), T::from_f64(FLATTEST), T::from_f64(0.001))
            .unwrap();
    let cascade = BiquadCascade::new([section; 2]);

    assert!(
        cascade.magnitude_at(T::from_f64(200.0)).unwrap()
            < section.magnitude_at(T::from_f64(200.0)).unwrap()
    );

    // Well below the cutoff both leave the signal alone. A fiftieth of the cutoff still costs
    // about 1e-7 of the amplitude for one section and twice that for two, so the check is against
    // that rather than against nothing at all.
    assert!((section.magnitude_at(T::ONE).unwrap() - T::ONE).abs() < tolerance);
    assert!((cascade.magnitude_at(T::ONE).unwrap() - T::ONE).abs() < tolerance);
}

#[test]
fn two_sections_cut_deeper_than_one_f64() {
    assert_two_sections_cut_deeper_than_one(1e-6_f64);
}

#[test]
fn two_sections_cut_deeper_than_one_f32() {
    assert_two_sections_cut_deeper_than_one(1e-5_f32);
}

fn assert_cascade_magnitude_is_the_product<T: Numeric>(tolerance: T) {
    let first =
        BiquadCoefficients::low_pass(T::from_f64(50.0), T::from_f64(FLATTEST), T::from_f64(0.001))
            .unwrap();
    let second = BiquadCoefficients::low_pass(
        T::from_f64(120.0),
        T::from_f64(FLATTEST),
        T::from_f64(0.001),
    )
    .unwrap();
    let cascade = BiquadCascade::new([first, second]);

    let probe = T::from_f64(80.0);
    let separately = first.magnitude_at(probe).unwrap() * second.magnitude_at(probe).unwrap();
    assert!((cascade.magnitude_at(probe).unwrap() - separately).abs() < tolerance);
}

#[test]
fn cascade_magnitude_is_the_product_f64() {
    assert_cascade_magnitude_is_the_product(1e-12_f64);
}

#[test]
fn cascade_magnitude_is_the_product_f32() {
    assert_cascade_magnitude_is_the_product(1e-4_f32);
}

// Three sections shift a 300 Hz oscillation by more than half a turn. Working the total out from a
// combined transfer function would wrap it back into range and report the wrong number.
#[test]
fn cascade_phase_adds_up_past_half_a_turn() {
    let section = BiquadCoefficients::low_pass(40.0_f64, FLATTEST, 0.001).unwrap();
    let cascade = BiquadCascade::new([section; 3]);
    assert!(cascade.phase_at(300.0).unwrap() < -core::f64::consts::PI);
}

// ---- harmonic notches -------------------------------------------------------

#[test]
fn harmonic_notch_hits_every_multiple() {
    let sections = harmonic_notch_coefficients::<3, f64>(80.0, 4.0, 0.001).unwrap();
    let cascade = BiquadCascade::new(sections);

    for frequency_hz in [80.0, 160.0, 240.0] {
        assert!(cascade.magnitude_at(frequency_hz).unwrap() < 0.05);
    }

    // Below the first notch, between two of them, and above the last, the signal survives.
    assert!(cascade.magnitude_at(20.0).unwrap() > 0.9);
    assert!(cascade.magnitude_at(120.0).unwrap() > 0.7);
    assert!(cascade.magnitude_at(300.0).unwrap() > 0.7);
}

// ---- the response queries ---------------------------------------------------

#[test]
fn cascade_response_queries_reject_a_frequency_at_or_above_nyquist() {
    let section = BiquadCoefficients::low_pass(50.0_f64, FLATTEST, 0.001).unwrap();
    let cascade = BiquadCascade::new([section; 2]);
    for frequency_hz in [500.0_f64, 600.0] {
        assert_eq!(
            cascade.magnitude_at(frequency_hz),
            Err(SignalError::FrequencyOutOfRange)
        );
        assert_eq!(
            cascade.phase_at(frequency_hz),
            Err(SignalError::FrequencyOutOfRange)
        );
        assert_eq!(
            cascade.delay_at(frequency_hz),
            Err(SignalError::FrequencyOutOfRange)
        );
    }
}

#[test]
fn cascade_response_queries_reject_a_non_finite_frequency() {
    let section = BiquadCoefficients::low_pass(50.0_f64, FLATTEST, 0.001).unwrap();
    let cascade = BiquadCascade::new([section; 2]);
    assert_eq!(cascade.magnitude_at(f64::NAN), Err(SignalError::NonFinite));
    assert_eq!(cascade.phase_at(f64::NAN), Err(SignalError::NonFinite));
    assert_eq!(cascade.delay_at(f64::NAN), Err(SignalError::NonFinite));
}

#[test]
fn harmonic_notch_rejects_a_multiple_past_the_limit() {
    // The third section would sit at 540 Hz, above half of a 1 kHz sampling rate.
    assert_eq!(
        harmonic_notch_coefficients::<3, f64>(180.0, 4.0, 0.001),
        Err(SignalError::FrequencyOutOfRange)
    );
}

#[test]
fn set_section_checks_the_index() {
    let section = BiquadCoefficients::low_pass(50.0_f64, FLATTEST, 0.001).unwrap();
    let mut cascade = BiquadCascade::new([section; 2]);
    let replacement = BiquadCoefficients::low_pass(80.0_f64, FLATTEST, 0.001).unwrap();

    assert_eq!(cascade.set_section(0, replacement), Ok(()));
    assert_eq!(
        cascade.set_section(2, replacement),
        Err(SignalError::SectionIndexOutOfRange)
    );
}

// ---- one shape over several channels ----------------------------------------

fn assert_multi_channel_matches_separate_filters<T: Numeric>(tolerance: T) {
    let coefficients =
        BiquadCoefficients::low_pass(T::from_f64(50.0), T::from_f64(FLATTEST), T::from_f64(0.001))
            .unwrap();
    let scales = [T::from_f64(0.3), T::from_f64(-0.7), T::from_f64(1.1)];

    let mut together = MultiChannelBiquad::new(coefficients);
    let mut separately = [Biquad::new(coefficients); 3];

    for sample in 0..200 {
        let drive =
            T::from_f64((2.0 * core::f64::consts::PI * 30.0 * f64::from(sample) / 1000.0).sin());
        let combined = together.filter(Vector::new([
            scales[0] * drive,
            scales[1] * drive,
            scales[2] * drive,
        ]));
        for channel in 0..3 {
            let alone = separately[channel].filter(scales[channel] * drive);
            assert!((combined[channel] - alone).abs() < tolerance);
        }
    }
}

#[test]
fn multi_channel_matches_separate_filters_f64() {
    assert_multi_channel_matches_separate_filters(1e-12_f64);
}

#[test]
fn multi_channel_matches_separate_filters_f32() {
    assert_multi_channel_matches_separate_filters(1e-4_f32);
}
