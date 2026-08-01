//! Signal processing: a notch removing a single tone, and a low-pass letting the slow part through.
//!
//! Run with: `cargo run -p multicalc-demos --example signal_filters`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::random::{Pcg32, RandomSource};
use multicalc::signal_processing::{Biquad, BiquadCoefficients, RunningMedian, SavitzkyGolay};

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

    // (4) A rate and an acceleration recovered from a noisy position reading, by fitting a small
    // curve across the last eleven samples.
    //
    // At a hundred samples a second rather than a thousand. Working out a rate divides by the
    // timestep, so a shorter one magnifies the noise: the same wobble that costs 0.03 of the rate
    // here would cost 0.3 at 1 kHz, and the acceleration, which divides twice, is ten times worse
    // again. A rate wanted from a noisy reading needs a longer gap between samples, not a shorter
    // one.
    const SLOW_DT: f64 = 0.01;
    let mut wobble = Pcg32::new(20260731);
    let noisy_position: Vec<f64> = (0..500)
        .map(|sample| {
            let time = sample as f64 * SLOW_DT;
            0.5 * time * time + 0.002 * (wobble.next_unit_f64() - 0.5)
        })
        .collect();
    let last_time = 499.0 * SLOW_DT;

    let mut fitted = SavitzkyGolay::<11, 3, f64>::latest(SLOW_DT).unwrap();
    let mut worst_fitted = 0.0_f64;
    let mut worst_difference = 0.0_f64;
    for (sample, &reading) in noisy_position.iter().enumerate() {
        let _ = fitted.filter(reading);
        // What a plain subtraction of the last two readings makes of the same data.
        let true_rate = sample as f64 * SLOW_DT;
        if sample >= 20 {
            let plain_difference = (reading - noisy_position[sample - 1]) / SLOW_DT;
            worst_fitted = worst_fitted.max((fitted.first_derivative() - true_rate).abs());
            worst_difference = worst_difference.max((plain_difference - true_rate).abs());
        }
    }
    let fitted_error = (fitted.first_derivative() - last_time).abs();

    println!("\nRate and acceleration from a noisy position, curve fit over 11 samples");
    println!("  true rate        = {last_time:.5}");
    println!(
        "  fitted rate      = {:.5}   (off by {fitted_error:.5})",
        fitted.first_derivative()
    );
    println!(
        "  fitted bend      = {:.5}   (true 1.00000)",
        fitted.second_derivative()
    );
    // One sample is luck either way, so the comparison is over the whole run.
    println!(
        "  worst rate error over the run: curve fit {worst_fitted:.5}, two-sample {worst_difference:.5}"
    );
    assert!(fitted_error < 0.05, "the fitted rate should be close");
    assert!(
        worst_fitted < worst_difference,
        "fitting a curve should beat subtracting two noisy samples"
    );
    // A second derivative out of noisy data is genuinely loose, and the bound says so.
    assert!((fitted.second_derivative() - 1.0).abs() < 2.0);

    // (5) One wild reading dropped outright, which no amount of smoothing can do. The same filter
    // is run twice over the same data, once with a single reading replaced by a wild value, and the
    // two outputs are compared where the spike lands.
    let spike_at = 250;
    let mut with_spike = RunningMedian::<5, f64>::new().unwrap();
    let mut clean = RunningMedian::<5, f64>::new().unwrap();
    let (mut spiked_output, mut clean_output) = (0.0, 0.0);
    for (sample, &reading) in noisy_position.iter().enumerate() {
        let output = with_spike.filter(if sample == spike_at { 99.0 } else { reading });
        let untouched = clean.filter(reading);
        if sample == spike_at {
            spiked_output = output;
            clean_output = untouched;
        }
    }

    println!("\nA single wild reading of 99.0 through a 5-wide median");
    println!("  with the wild reading = {spiked_output:.5}");
    println!("  without it            = {clean_output:.5}");
    // One bad sample out of five never reaches the middle of the sorted window, so the two agree
    // exactly rather than merely closely.
    assert_eq!(
        spiked_output, clean_output,
        "the median should not notice one bad reading"
    );
}
