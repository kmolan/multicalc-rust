# Signal processing

Streaming filters: one sample in, one filtered sample out. Fixed-size, no allocation, no panics,
and generic over the `Numeric` scalar, so the same code runs at `f32` on a microcontroller.

Frequencies are in hertz and timesteps in seconds. Each filter checks its configuration once at
construction. After that, each per-sample call is infallible: no panic, no allocation, no error.
The per-sample calls do not inspect the sample, so a NaN or an infinity goes in. The exception is
`RunningMedian`, which rejects a NaN. Each filter also has a checked entry point; see
[Non-finite samples](#non-finite-samples).

| Type | Function |
|---|---|
| `BiquadCoefficients` | Second-order section design: `low_pass`, `high_pass`, `band_pass`, `notch` from a frequency, a quality factor and a timestep, or `new` from raw weights. Response queries: `magnitude_at`, `magnitude_in_decibels_at`, `phase_at`, `delay_at` (phase delay, in seconds). `is_stable` checks raw weights. |
| `Biquad` | The running section. `set_coefficients` changes the weights and keeps the state, so a notch can track a moving frequency. `settle_to` sets the steady state for a constant input and removes the start-up transient. |
| `BiquadCascade` | Sections in series, for a steeper roll-off. `harmonic_notch_coefficients` designs notches at a fundamental and its integer harmonics, for motor vibration. |
| `MultiChannelBiquad` | One design on each component of a `Vector`, with separate state per channel. One call per tick for a three-axis gyro. |
| `OnePoleLowPass` | First-order low-pass, from a smoothing weight (`new`) or a cutoff (`from_cutoff`). `Pid::with_derivative_filter` puts this filter on the derivative term. |
| `MovingAverage` | Mean of the window, summed from the whole window each sample, so it has no running-total drift. |
| `RunningMedian` | Median of the window. It removes a single outlier and rejects a NaN. |
| `SavitzkyGolay` | Least-squares polynomial fit across the window. It gives the value, the first derivative and the second derivative. |
| `Deadband` | Maps values near zero to zero. `plain` passes values outside the band unchanged; `recentered` subtracts the threshold, so the output is continuous. |
| `Hysteresis` | A boolean output with separate on and off thresholds, so a signal near the threshold does not chatter. |
| `SlewRateLimiter` | Follows a target with separate rise and fall rate limits, so a step becomes a ramp. `symmetric` uses one rate. |

```rust
use multicalc::{Biquad, BiquadCascade, BiquadCoefficients, MultiChannelBiquad};
use multicalc::signal_processing::harmonic_notch_coefficients;
use multicalc::Vector;

let timestep = 0.001_f64; // a 1 kHz loop

// Butterworth low-pass (Q = 1/√2): -3 dB at the cutoff, under 10 ms of phase delay at 5 Hz.
let low_pass = BiquadCoefficients::low_pass(50.0, 0.70710678, timestep).unwrap();
assert!((low_pass.magnitude_in_decibels_at(50.0).unwrap() + 3.0).abs() < 0.5);
assert!(low_pass.delay_at(5.0).unwrap() < 0.01);

// A notch at 180 Hz: after 2000 samples, a 180 Hz sine is gone.
let mut notch = Biquad::new(BiquadCoefficients::notch(180.0, 4.0, timestep).unwrap());
let mut last = 0.0;
for sample in 0..2000 {
    last = notch.filter((2.0 * core::f64::consts::PI * 180.0 * sample as f64 * timestep).sin());
}
assert!(last.abs() < 0.05);

// Retune: the state is kept, so the output has no step.
notch.set_coefficients(BiquadCoefficients::notch(210.0, 4.0, timestep).unwrap());

// Notches at 80, 160 and 240 Hz. A 180 Hz fundamental would put the third at 540 Hz, above the
// 500 Hz Nyquist frequency.
let harmonics = harmonic_notch_coefficients::<3, f64>(80.0, 4.0, timestep).unwrap();
let motor_notch = BiquadCascade::new(harmonics);
for frequency_hz in [80.0, 160.0, 240.0] {
    assert!(motor_notch.magnitude_at(frequency_hz).unwrap() < 0.05);
}

// Three axes, one design, separate state per axis.
let mut rates = MultiChannelBiquad::new(low_pass);
let filtered = rates.filter(Vector::new([0.3, -0.7, 1.1]));
assert!(filtered[0] > 0.0);
```

The window filters use the last `WINDOW` samples, not a recurrence:

```rust
use multicalc::{RunningMedian, SavitzkyGolay};

// The median removes one outlier. A 5-sample mean would pass a fifth of it.
let mut median = RunningMedian::<5, f64>::new().unwrap();
for reading in [1.0, 1.1, 0.9, 50.0, 1.05] {
    median.filter(reading);
}
assert_eq!(median.value(), 1.05);

// A quadratic fit (3 terms) is exact on a parabola, so both derivatives are exact.
let timestep = 0.001_f64;
let mut fitted = SavitzkyGolay::<11, 3, f64>::latest(timestep).unwrap();
let mut time = 0.0;
for sample in 0..200 {
    time = sample as f64 * timestep;
    fitted.filter(0.5 * time * time);
}
assert!((fitted.first_derivative() - time).abs() < 1e-6);
assert!((fitted.second_derivative() - 1.0).abs() < 1e-6);

// `latest` evaluates at the newest sample: no delay. `centered` evaluates mid-window.
assert_eq!(fitted.delay(), 0.0);
assert!((SavitzkyGolay::<11, 3, f64>::centered(timestep).unwrap().delay() - 0.005).abs() < 1e-12);
```

| Filter | Use |
|---|---|
| `MovingAverage` | Light smoothing of small, uniform noise. |
| `RunningMedian` | Single wrong readings; no linear smoother removes them. |
| `SavitzkyGolay` | A rate or an acceleration from a noisy signal. A finite difference multiplies the noise by the sample rate. |

`SavitzkyGolay::centered` smooths better than `latest`, but its output is half a window old. At
eleven samples and 1 kHz that is 5 ms. `delay()` returns this value, so a loop can compensate.

`Deadband`, `Hysteresis` and `SlewRateLimiter` condition a signal; they are not filters. They are
in this module because they map numbers to numbers with no robot model.

```rust
use multicalc::{Deadband, Hysteresis, SlewRateLimiter};

// Joystick deadband. `recentered` is continuous at the band edge: 0.5 gives 0.4, not 0.5.
let stick = Deadband::recentered(0.1_f64).unwrap();
assert_eq!(stick.apply(0.05), 0.0);
assert!((stick.apply(0.5) - 0.4).abs() < 1e-12);

// On above 0.6, off below 0.4, hold in between.
let mut warning = Hysteresis::new(0.4_f64, 0.6).unwrap();
assert!(!warning.update(0.5));
assert!(warning.update(0.7));
assert!(warning.update(0.5));
assert!(!warning.update(0.3));

// Rise limit 1 per second, 0.1 s tick: a step becomes a ramp of 0.1 per tick.
let mut command = SlewRateLimiter::new(1.0_f64, 2.0, 0.1).unwrap();
command.filter(0.0);
assert!((command.filter(10.0) - 0.1).abs() < 1e-12);
```

Read `delay_at` before you select a filter. Each filter delays its output to remove noise. In a
control loop, that delay reduces the phase margin. Evaluate it near the loop crossover frequency,
not only at the cutoff. A steeper filter or a lower cutoff gives a cleaner signal and more delay.

## Non-finite samples

Sensor dropouts, upstream division by zero and GPS loss without a fix all produce NaN. Use the
unchecked call for samples already validated: it has no branch, for a 1 kHz loop. Use the checked
call for all other samples. A checked call returns `Err(SignalError::NonFinite)` before it changes
any state.

| Filter | Unchecked | Effect of a non-finite sample | Checked |
|---|---|---|---|
| `OnePoleLowPass` | `filter` | Latches until `reset`. | `filter_checked` |
| `Biquad` | `filter` | Latches: the value enters the feedback path. | `filter_checked` |
| `BiquadCascade` | `filter` | Latches in each section. | `filter_checked` |
| `MultiChannelBiquad` | `filter` | Latches in that channel only. | `filter_checked` |
| `MovingAverage` | `filter` | Spoils one window, then clears. | `filter_checked` |
| `RunningMedian` | `filter` | A NaN is rejected and the median holds. | `filter_checked` |
| `SavitzkyGolay` | `filter` | Spoils value, slope and curvature for one window. | `filter_checked` |
| `Deadband` | `apply` | Passes through; no state. | `apply_checked` |
| `Hysteresis` | `update` | A NaN is ignored and the output holds. | `update_checked` |
| `SlewRateLimiter` | `filter` | A NaN latches; an infinity latches only as the first target. | `filter_checked` |

Two checked calls accept more than the name suggests:

- `RunningMedian` rejects only NaN. The sort orders an infinity correctly, so the median treats it
  as an outlier.
- `SlewRateLimiter` rejects an infinity only before the first sample. After that, the rate clamp
  bounds it.

`settle_to` is not checked. `RunningMedian::filter` pays one branch for its NaN check.

```rust
use multicalc::error::SignalError;
use multicalc::{Biquad, BiquadCoefficients, RunningMedian};

let shape = BiquadCoefficients::low_pass(50.0, 0.70710678, 0.001_f64).unwrap();

// Unchecked: the NaN enters the feedback path, and each later output is NaN.
let mut spoiled = Biquad::new(shape);
let _ = spoiled.filter(1.0);
let _ = spoiled.filter(f64::NAN);
assert!(spoiled.filter(1.0).is_nan());

// Checked: the sample is rejected and the state does not change.
let mut guarded = Biquad::new(shape);
let _ = guarded.filter(1.0);
let before = guarded.value();

assert_eq!(guarded.filter_checked(f64::NAN), Err(SignalError::NonFinite));
assert_eq!(guarded.value(), before);

// `RunningMedian` accepts an infinity as an outlier. Both entry points reject a NaN; only the
// checked one reports it.
let mut median = RunningMedian::<5, f64>::new().unwrap();
for reading in [1.0, 1.1, 0.9, f64::INFINITY, 1.05] {
    let _ = median.filter_checked(reading);
}
assert_eq!(median.value(), 1.05);
assert_eq!(median.filter_checked(f64::NAN), Err(SignalError::NonFinite));

// Unchecked: the NaN is rejected and the median holds.
let mut unchecked = RunningMedian::<5, f64>::new().unwrap();
for reading in [1.0, 1.1, 0.9, 50.0, 1.05] {
    let _ = unchecked.filter(reading);
}
let held = unchecked.value();
assert_eq!(held, 1.05);
assert_eq!(unchecked.filter(f64::NAN), held);
assert_eq!(unchecked.value(), held);
```

Use one entry point per filter. A checked call protects the filter from that sample only. It does
not guarantee a finite output, and it cannot repair a filter that an unchecked call already
spoiled. Use `reset` to recover.

Caution: a NaN into `Hysteresis::update` is silent. The switch holds its last output, so a dead
sensor looks the same as a signal inside the hysteresis band. `RunningMedian` also holds, but at a
value from accepted readings only, and `filter_checked` reports each rejected NaN.

## Errors

These calls return [`SignalError`](error-handling.md):

- Each design and filter constructor. `Biquad::new`, `BiquadCascade::new` and
  `MultiChannelBiquad::new` take a checked design and are infallible.
- `harmonic_notch_coefficients`.
- The response queries `magnitude_at`, `magnitude_in_decibels_at`, `phase_at` and `delay_at`.
- `BiquadCascade::set_section` (`SectionIndexOutOfRange`).
- The checked entry points above.

Demo:
[signal_filters.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/signal_filters.rs).

---

[Back to the tutorial index](README.md)
