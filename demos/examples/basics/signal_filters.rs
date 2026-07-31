//! Signal processing: a notch removing a single tone, and a low-pass letting the slow part through.
//!
//! Run with: `cargo run -p multicalc-demos --example signal_filters`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::signal_processing::{Biquad, BiquadCoefficients};

const SAMPLE_RATE_HZ: f64 = 1000.0;
const DT: f64 = 1.0 / SAMPLE_RATE_HZ;
const SAMPLES: usize = 4000;
/// The measurement ignores the first half, by which time the filters have settled.
const MEASURE_FROM: usize = SAMPLES / 2;

/// How much of `frequency_hz` a stretch of samples holds.
///
/// The samples are weighed against a sine and a cosine at that frequency; whatever oscillates
/// along with them adds up, and everything else averages away. The length of the resulting pair is
/// the amplitude.
fn amplitude_at(samples: &[f64], frequency_hz: f64) -> f64 {
    let mut with_sine = 0.0;
    let mut with_cosine = 0.0;
    for (offset, &sample) in samples.iter().enumerate() {
        let angle = 2.0 * core::f64::consts::PI * frequency_hz * offset as f64 * DT;
        with_sine += sample * angle.sin();
        with_cosine += sample * angle.cos();
    }
    2.0 * with_sine.hypot(with_cosine) / samples.len() as f64
}

fn run(mut filter: Biquad<f64>, signal: &[f64]) -> Vec<f64> {
    signal.iter().map(|&sample| filter.filter(sample)).collect()
}

fn main() {
    // A slow 5 Hz movement with a 180 Hz vibration riding on it, which is what a rate reading looks
    // like with a motor spinning nearby.
    let signal: Vec<f64> = (0..SAMPLES)
        .map(|sample| {
            let time = sample as f64 * DT;
            (2.0 * core::f64::consts::PI * 5.0 * time).sin()
                + 0.8 * (2.0 * core::f64::consts::PI * 180.0 * time).sin()
        })
        .collect();

    let before_slow = amplitude_at(&signal[MEASURE_FROM..], 5.0);
    let before_fast = amplitude_at(&signal[MEASURE_FROM..], 180.0);
    println!("Input, measured over the second half of 4000 samples at 1 kHz");
    println!("  5 Hz movement    = {before_slow:.5}");
    println!("  180 Hz vibration = {before_fast:.5}");

    // (1) A notch on the vibration, leaving everything else alone.
    let notch = BiquadCoefficients::notch(180.0, 4.0, DT).unwrap();
    let notched = run(Biquad::new(notch), &signal);
    let notched_slow = amplitude_at(&notched[MEASURE_FROM..], 5.0);
    let notched_fast = amplitude_at(&notched[MEASURE_FROM..], 180.0);
    let suppression_db = 20.0 * (before_fast / notched_fast).log10();

    println!("\nNotch at 180 Hz, sharpness 4");
    println!("  180 Hz vibration = {notched_fast:.3e}  ({suppression_db:.0} dB down)");
    println!("  5 Hz movement    = {notched_slow:.5}");
    assert!(suppression_db >= 20.0, "the notch barely touched 180 Hz");
    assert!(
        (notched_slow - before_slow).abs() < 0.05 * before_slow,
        "the notch should leave the slow movement alone"
    );

    // (2) A low-pass, which removes the vibration as well but delays everything it keeps.
    let low_pass =
        BiquadCoefficients::low_pass(50.0, core::f64::consts::FRAC_1_SQRT_2, DT).unwrap();
    let smoothed = run(Biquad::new(low_pass), &signal);
    let smoothed_fast = amplitude_at(&smoothed[MEASURE_FROM..], 180.0);
    let delay_seconds = low_pass.delay_at(5.0);

    println!("\nLow-pass at 50 Hz, sharpness 0.707");
    println!("  180 Hz vibration = {smoothed_fast:.5}");
    println!("  delay at 5 Hz    = {:.3} ms", delay_seconds * 1000.0);
    assert!(smoothed_fast < 0.08, "the fast part should be mostly gone");
    assert!(
        delay_seconds > 0.0 && delay_seconds < 0.01,
        "the delay should be positive and under ten milliseconds"
    );

    // (3) The same filter, describing itself: how much it keeps at three frequencies.
    println!("\nWhat the low-pass keeps");
    for frequency_hz in [5.0, 50.0, 180.0] {
        println!(
            "  {frequency_hz:>5.0} Hz          = {:>8.3} dB",
            low_pass.magnitude_in_decibels_at(frequency_hz)
        );
    }
    let at_cutoff = low_pass.magnitude_in_decibels_at(50.0);
    assert!(
        (at_cutoff + 3.0).abs() < 0.5,
        "a low-pass is about 3 dB down at its own cutoff"
    );
}
