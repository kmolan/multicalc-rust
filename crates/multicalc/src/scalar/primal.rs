//! One-way scalar-to-f64 projection for plotting.
//!
//! `multicalc::scalar::Numeric` provides `from_f64` but no `to_f64`, so this crate defines its
//! own projection. It covers the float scalars and the autodiff scalars, projecting each to its
//! primal (value) part so a differentiated quantity plots as its underlying value. The autodiff
//! impls delegate through the primal, so nested scalars such as `Dual<HyperDual<f64>>` work.use crate::{Dual, HyperDual, Jet, Numeric};

use crate::{Dual, HyperDual, Jet, Numeric};

/// A scalar that can be projected to `f64` and `f32`.
pub trait Primal {
    /// Returns the value as an `f64` (the primal part, for autodiff scalars).
    fn to_f64(&self) -> f64;

    /// Returns the value as an `f32` (the primal part, for autodiff scalars).
    fn to_f32(&self) -> f32;
}

impl Primal for f64 {
    fn to_f64(&self) -> f64 {
        *self
    }

    fn to_f32(&self) -> f32 {
        *self as f32
    }
}

impl Primal for f32 {
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    fn to_f32(&self) -> f32 {
        *self
    }
}

impl<T: Numeric + Primal> Primal for Dual<T> {
    fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }

    fn to_f32(&self) -> f32 {
        self.value.to_f32()
    }
}

impl<T: Numeric + Primal> Primal for HyperDual<T> {
    fn to_f64(&self) -> f64 {
        self.real.to_f64()
    }

    fn to_f32(&self) -> f32 {
        self.real.to_f32()
    }
}

impl<T: Numeric + Primal, const N: usize> Primal for Jet<T, N> {
    fn to_f64(&self) -> f64 {
        self.value().to_f64()
    }

    fn to_f32(&self) -> f32 {
        self.value().to_f32()
    }
}
