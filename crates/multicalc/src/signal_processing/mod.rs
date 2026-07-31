//! Signal processing: filters, smoothers, and signal conditioning.
//!
//! - [`OnePoleLowPass`] — the simplest low-pass, by smoothing weight or by cutoff frequency.
//! - [`BiquadCoefficients`] — the shape of a second-order filter, as a low-pass, high-pass,
//!   band-pass, or notch.
//! - [`Biquad`] — a second-order filter running on a stream of samples.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), runs on a fixed
//! timestep in seconds, and takes frequencies in hertz. A filter is configured once, with the
//! configuration checked up front, and every call after that is total.

mod biquad;
mod one_pole;

pub use biquad::{Biquad, BiquadCoefficients};
pub use one_pole::OnePoleLowPass;
