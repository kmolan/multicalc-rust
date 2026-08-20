# Signal processing

Filters for a stream of samples: give a filter a number, get the filtered number back. Fixed-size,
no allocation, no panics, and generic over the `Numeric` scalar, so the same code runs at `f32` on a
microcontroller.

Frequencies are in hertz and timesteps in seconds. Every filter is configured once, with the
configuration checked up front, and every call after that is total.

- `BiquadCoefficients`: the shape of a second-order filter, designed as a `low_pass`, `high_pass`,
  `band_pass`, or `notch` from a frequency, a sharpness, and the seconds between samples. It also
  reports on itself: `magnitude_at` and `magnitude_in_decibels_at` for how much of a frequency it
  keeps, `phase_at` and `delay_at` for how far it shifts one, and `is_stable` for whether weights
  handed in directly will settle.
- `Biquad`: the running filter. `set_coefficients` swaps the weights without disturbing the memory
  of recent samples, so a notch can follow a frequency that moves; `settle_to` starts it where a
  long run of a value would leave it, skipping the opening transient.
- `BiquadCascade`: several sections in series, for a sharper cut than one section gives.
  `harmonic_notch_coefficients` fills one with notches on a frequency and the whole-number multiples
  above it, which is the shape a spinning motor's vibration takes.
- `MultiChannelBiquad`: one shape over every component of a `Vector`, each channel keeping its own
  memory. One object and one call per tick for a three-axis rate sensor.
- `OnePoleLowPass`: the simplest low-pass, by smoothing weight (`new`) or by cutoff frequency
  (`from_cutoff`). This is the filter `Pid::with_derivative_filter` puts on the derivative term.
- `MovingAverage`: the average of the last few samples, added up fresh each time so it cannot drift.
- `RunningMedian`: their middle value instead, which drops a single bad reading outright.
- `SavitzkyGolay`: a small curve fitted across the window, reporting the smoothed value together
  with the slope and the bend — one noisy position reading gives a rate and an acceleration.
- `Deadband`: treats values near zero as zero, in a plain form and one that leaves the band
  smoothly rather than jumping.
- `Hysteresis`: a yes-or-no answer with a gap between its thresholds, so a signal sitting near the
  switching point does not chatter.
- `SlewRateLimiter`: follows a target without moving faster than its separate rising and falling
  limits, turning a step into a ramp.

```rust
use multicalc::{Biquad, BiquadCascade, BiquadCoefficients, MultiChannelBiquad};
use multicalc::signal_processing::harmonic_notch_coefficients;
use multicalc::Vector;

let timestep = 0.001_f64; // a 1 kHz loop

// A low-pass is about 3 dB down at its own cutoff, and lags by a few milliseconds below it.
let low_pass = BiquadCoefficients::low_pass(50.0, 0.70710678, timestep).unwrap();
assert!((low_pass.magnitude_in_decibels_at(50.0) + 3.0).abs() < 0.5);
assert!(low_pass.delay_at(5.0) < 0.01);

// A notch removes one frequency: 2000 samples of a 180 Hz oscillation come out flat.
let mut notch = Biquad::new(BiquadCoefficients::notch(180.0, 4.0, timestep).unwrap());
let mut last = 0.0;
for sample in 0..2000 {
    last = notch.filter((2.0 * core::f64::consts::PI * 180.0 * sample as f64 * timestep).sin());
}
assert!(last.abs() < 0.05);

// Retuning keeps the memory, so a tracked notch does not step as it moves.
notch.set_coefficients(BiquadCoefficients::notch(210.0, 4.0, timestep).unwrap());

// Notches on 80 Hz and its next two multiples, one section each. The fundamental is 80 Hz and not
// 180 Hz because a third section on 180 Hz would sit at 540 Hz, past half of a 1 kHz sampling rate.
let harmonics = harmonic_notch_coefficients::<3, f64>(80.0, 4.0, timestep).unwrap();
let motor_notch = BiquadCascade::new(harmonics);
for frequency_hz in [80.0, 160.0, 240.0] {
    assert!(motor_notch.magnitude_at(frequency_hz) < 0.05);
}

// Three axes through one filter, each keeping its own memory.
let mut rates = MultiChannelBiquad::new(low_pass);
let filtered = rates.filter(Vector::new([0.3, -0.7, 1.1]));
assert!(filtered[0] > 0.0);
```

The window filters work from the last few samples rather than from a recurrence:

```rust
use multicalc::{RunningMedian, SavitzkyGolay};

// A median drops one wild reading outright. An average would carry a fifth of it into the output.
let mut median = RunningMedian::<5, f64>::new().unwrap();
for reading in [1.0, 1.1, 0.9, 50.0, 1.05] {
    median.filter(reading);
}
assert_eq!(median.value(), 1.05);

// Three terms fit a curve exactly, so a curve comes back with its slope and bend.
let timestep = 0.001_f64;
let mut fitted = SavitzkyGolay::<11, 3, f64>::latest(timestep).unwrap();
let mut time = 0.0;
for sample in 0..200 {
    time = sample as f64 * timestep;
    fitted.filter(0.5 * time * time);
}
assert!((fitted.first_derivative() - time).abs() < 1e-6);
assert!((fitted.second_derivative() - 1.0).abs() < 1e-6);

// Reading at the newest sample costs no delay; reading at the middle costs half a window.
assert_eq!(fitted.delay(), 0.0);
assert!((SavitzkyGolay::<11, 3, f64>::centered(timestep).unwrap().delay() - 0.005).abs() < 1e-12);
```

Which of the three to reach for: a mean for gentle smoothing of a signal whose noise is small and
even; a median when single readings come back wrong, which is the one case no amount of smoothing
helps; and a curve fit when a rate or an acceleration is wanted out of the same noisy signal, since
subtracting two noisy samples multiplies the noise by the sampling rate. Note that
`SavitzkyGolay::centered` smooths considerably better than `latest` but describes a sample from half
a window ago — eleven samples at 1 kHz puts its answer five milliseconds behind, which `delay()`
reports so a loop can account for it.

The last three shape a signal without filtering it. They live here for the same reason the filters
do: numbers in, numbers out, nothing about robots.

```rust
use multicalc::{Deadband, Hysteresis, SlewRateLimiter};

// A stick that reads a little off centre at rest. The re-centered form leaves the band smoothly,
// so a small push gives a small command instead of jumping to a tenth.
let stick = Deadband::recentered(0.1_f64).unwrap();
assert_eq!(stick.apply(0.05), 0.0);
assert!((stick.apply(0.5) - 0.4).abs() < 1e-12);

// A switch that will not chatter: it turns on above 0.6 and off below 0.4, and holds in between.
let mut warning = Hysteresis::new(0.4_f64, 0.6).unwrap();
assert!(!warning.update(0.5));
assert!(warning.update(0.7));
assert!(warning.update(0.5));
assert!(!warning.update(0.3));

// A step in the target comes out as a ramp, climbing at one per second on a tenth-second tick.
let mut command = SlewRateLimiter::new(1.0_f64, 2.0, 0.1).unwrap();
command.filter(0.0);
assert!((command.filter(10.0) - 0.1).abs() < 1e-12);
```

The number worth knowing before picking a filter is `delay_at`. Every filter pays for what it
removes by putting its output behind its input, and inside a control loop that delay is what eats
the margin the loop has before it starts oscillating — so it is worth reading at the frequency the
loop works around, not just at the cutoff. A sharper filter or a lower cutoff buys a cleaner signal
with more delay, and there is no setting that avoids the trade.

Errors: every constructor returns [`SignalError`](error-handling.md). Full demo:
[signal_filters.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/signal_filters.rs).


---

[Back to the tutorial index](README.md)
