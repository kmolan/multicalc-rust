//! One-way scalar-to-f64 projection for plotting.
//!
//! `multicalc::scalar::Numeric` provides `from_f64` but no `to_f64`, so this crate defines its
//! own projection. It covers the float scalars and the autodiff scalars, projecting each to its
//! primal (value) part so a differentiated quantity plots as its underlying value. The autodiff
//! impls delegate through the primal, so nested scalars such as `Dual<HyperDual<f64>>` work.

use multicalc::scalar::{Dual, HyperDual, Jet, Numeric};

/// A scalar that can be projected to `f64` for plotting.
pub trait Plottable: Copy {
    /// Returns the value as an `f64` (the primal part, for autodiff scalars).
    fn to_plot_f64(self) -> f64;
}

impl Plottable for f64 {
    fn to_plot_f64(self) -> f64 {
        self
    }
}

impl Plottable for f32 {
    fn to_plot_f64(self) -> f64 {
        f64::from(self)
    }
}

impl<T: Numeric + Plottable> Plottable for Dual<T> {
    fn to_plot_f64(self) -> f64 {
        self.value.to_plot_f64()
    }
}

impl<T: Numeric + Plottable> Plottable for HyperDual<T> {
    fn to_plot_f64(self) -> f64 {
        self.real.to_plot_f64()
    }
}

impl<T: Numeric + Plottable, const N: usize> Plottable for Jet<T, N> {
    fn to_plot_f64(self) -> f64 {
        self.value().to_plot_f64()
    }
}
