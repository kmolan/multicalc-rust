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
//! Infallible is not the same as meaningful: the per-sample calls do not look at the sample they
//! are handed, so a NaN or an infinity goes in and whatever falls out is the answer. What that
//! costs varies by filter:
//!
//! | Filter | Unchecked | Cost of a non-finite sample | Checked |
//! |---|---|---|---|
//! | [`OnePoleLowPass`] | `filter` | Latches until `reset`. | `filter_checked` |
//! | [`Biquad`] | `filter` | Latches: the value enters the feedback path. | `filter_checked` |
//! | [`BiquadCascade`] | `filter` | Latches, in every section at once. | `filter_checked` |
//! | [`MultiChannelBiquad`] | `filter` | Latches, in the channel it lands on only. | `filter_checked` |
//! | [`MovingAverage`] | `filter` | Spoils one window, then clears on its own. | `filter_checked` |
//! | [`RunningMedian`] | `filter` | A NaN silently shifts the answer to a **wrong finite number**. | `filter_checked` |
//! | [`SavitzkyGolay`] | `filter` | Spoils value, slope and bend for one window. | `filter_checked` |
//! | [`Deadband`] | `apply` | Passes straight through; no state to spoil. | `apply_checked` |
//! | [`Hysteresis`] | `update` | A NaN is ignored and the answer silently **holds**. | `update_checked` |
//! | [`SlewRateLimiter`] | `filter` | A NaN latches; an infinity only as the first target. | `filter_checked` |
//!
//! A checked call returns [`Err(SignalError::NonFinite)`](crate::error::SignalError::NonFinite)
//! before touching any state, so a refused sample leaves the filter exactly as it was. Two admit
//! more than the name suggests: [`RunningMedian`] refuses only NaN, since sorting handles
//! infinities correctly, and [`SlewRateLimiter`] refuses an infinity only while unseeded, since the
//! rate clamp then bounds it. `settle_to` is not checked at all.
//!
//! A checked call protects the filter from *this* sample; it does not promise a finite result, and
//! cannot repair a filter already spoiled through the unchecked call. Use one entry point per
//! filter, and `reset` to recover.

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
