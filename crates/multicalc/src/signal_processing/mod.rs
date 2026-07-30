//! Signal processing: filters, smoothers, and signal conditioning.
//!
//! - [`OnePoleLowPass`] — the simplest low-pass, by smoothing weight or by cutoff frequency.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), runs on a fixed
//! timestep in seconds, and takes frequencies in hertz. A filter is configured once, with the
//! configuration checked up front, and every call after that is total.

mod one_pole;

pub use one_pole::OnePoleLowPass;
