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
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), runs on a fixed
//! timestep in seconds, and takes frequencies in hertz. A filter is configured once, with the
//! configuration checked up front, and every call after that is total.

mod biquad;
mod cascade;
mod one_pole;
mod savitzky_golay;
mod window;

pub use biquad::{Biquad, BiquadCoefficients};
pub use cascade::{BiquadCascade, MultiChannelBiquad, harmonic_notch_coefficients};
pub use one_pole::OnePoleLowPass;
pub use savitzky_golay::SavitzkyGolay;
pub use window::{MovingAverage, RunningMedian};
