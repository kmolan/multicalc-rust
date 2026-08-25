//! Signal processing: filters, smoothers, and signal conditioning.
//!
//! - [`OnePoleLowPass`] — the simplest low-pass, by smoothing weight or by cutoff frequency.
//! - [`BiquadCoefficients`] — the shape of a second-order filter, as a low-pass, high-pass,
//!   band-pass, or notch.
//! - [`Biquad`] — a second-order filter running on a stream of samples.
//! - [`BiquadCascade`] — several of those in series, with
//!   [`harmonic_notch_coefficients`] for a frequency and the multiples above it.
//! - [`MultiChannelBiquad`] — one shape filtering every component of a vector, such as the three
//!   axes of a rate sensor.
//! - [`MovingAverage`] — the average of the last few samples.
//! - [`RunningMedian`] — their middle value, which drops a single wild reading outright.
//! - [`SavitzkyGolay`] — a small curve fitted across the window, reporting its value, slope, and
//!   bend.
//! - [`Deadband`] — treats values near zero as zero.
//! - [`Hysteresis`] — a yes-or-no answer with a gap, so it does not chatter.
//! - [`SlewRateLimiter`] — follows a target without moving faster than a given rate.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), runs on a fixed
//! timestep in seconds, and takes frequencies in hertz. A filter is configured once, with the
//! configuration checked up front, and every call after that is infallible: it never panics, never
//! allocates, and never returns an error.
//!
//! # Non-finite samples
//!
//! Infallible is not the same as meaningful. The per-sample calls listed below **do not look at the
//! sample they are handed**, so a NaN or an infinity goes into the filter and whatever falls out is
//! the answer. What that costs depends entirely on the filter, which is why it is worth reading the
//! row for the one you are using rather than assuming they behave alike:
//!
//! | Filter | Unchecked call | What a non-finite sample costs | Checked call |
//! |---|---|---|---|
//! | [`OnePoleLowPass`] | `filter` | Latches. Every later output is NaN, whatever the input. | `filter_checked` |
//! | [`Biquad`] | `filter` | Latches: the value enters the feedback path. An infinity degrades to NaN. | `filter_checked` |
//! | [`BiquadCascade`] | `filter` | Latches, in every section at once. | `filter_checked` |
//! | [`MultiChannelBiquad`] | `filter` | Latches, but only in the channel it lands on. | `filter_checked` |
//! | [`MovingAverage`] | `filter` | Spoils the output for one window, then clears on its own. Opposite infinities in one window cancel into a NaN. | `filter_checked` |
//! | [`RunningMedian`] | `filter` | A NaN silently shifts the answer to a **wrong finite number** for one window. Infinities are handled correctly. | `filter_checked` |
//! | [`SavitzkyGolay`] | `filter` | Spoils the value, slope and bend for one window. The signed weights can flip an infinity or turn it into a NaN. | `filter_checked` |
//! | [`Deadband`] | `apply` | Passes straight through. No state, so it costs exactly one sample. | `apply_checked` |
//! | [`Hysteresis`] | `update` | A NaN is ignored and the answer silently **holds**. Infinities switch it normally. | `update_checked` |
//! | [`SlewRateLimiter`] | `filter` | A NaN latches. An infinity is clamped harmlessly unless it is the first target, which it becomes. | `filter_checked` |
//!
//! Every checked call returns [`Err(SignalError::NonFinite)`](crate::error::SignalError::NonFinite)
//! for a sample it refuses, and refuses it **before touching any state** — so a rejected sample
//! leaves the filter exactly as it was and the caller is free to drop it and carry on.
//!
//! Three things do not follow that rule, and are worth knowing before you rely on it:
//!
//! - [`RunningMedian`] admits infinities, because sorting handles them correctly and the filter does
//!   no arithmetic that could manufacture a NaN. It refuses only NaN, and its checked call can
//!   therefore return `Ok` wrapping an infinity, when most of the window is infinite.
//! - [`SlewRateLimiter`] refuses an infinity only while it is unseeded. Once a first target has
//!   arrived the rate clamp makes an infinite target harmless, so it is accepted.
//! - `settle_to`, on [`Biquad`], [`BiquadCascade`] and [`MultiChannelBiquad`], writes state straight
//!   from the value it is given and is not checked at all. It is the one piece of configuration that
//!   is not screened up front.
//!
//! Finally, a checked call promises that *this sample* will not spoil the filter. It does not
//! promise a finite result: a filter already spoiled through its unchecked entry point keeps
//! returning non-finite values from checked calls too, since none of them inspect the state they
//! start from. Pick one entry point per filter and use it consistently, and call `reset` if a filter
//! has been spoiled.

mod biquad;
mod cascade;
mod conditioning;
mod one_pole;
mod savitzky_golay;
mod window;

pub use biquad::{Biquad, BiquadCoefficients};
pub use cascade::{BiquadCascade, MultiChannelBiquad, harmonic_notch_coefficients};
pub use conditioning::{Deadband, Hysteresis, SlewRateLimiter};
pub use one_pole::OnePoleLowPass;
pub use savitzky_golay::SavitzkyGolay;
pub use window::{MovingAverage, RunningMedian};
