//! Control: feedback controllers, signal filters, and path-following laws.
//!
//! Everything here is generic over [`Numeric`](crate::Numeric), so the same code runs at `f32` or
//! `f64` or through an autodiff scalar. Units are SI, angles are radians, and controllers operate on
//! a fixed timestep `dt`.

mod derivative_filter;

pub use derivative_filter::OnePoleLowPass;
