//! Spatial math: rotations, Lie groups and spatial-algebra types.

use crate::scalar::Numeric;

pub mod lie;
pub mod quaternion;

pub use lie::{SE2, SE3, SO2, SO3};
pub use quaternion::Quaternion;

/// The angle threshold below which trig ratios switch to their Taylor series, keeping values
/// finite and derivatives continuous. A fixed absolute cutoff (not `EPSILON`-relative); correct for
/// both f32 and f64.
#[inline]
pub(crate) fn small_angle<T: Numeric>() -> T {
    T::from_f64(1e-6)
}

/// The squared small-angle threshold, for branches taken on a squared magnitude (θ², ‖v‖²) before
/// any `sqrt`. Branching pre-`sqrt` keeps the AD derivative finite at exactly zero, where `sqrt`'s
/// derivative is NaN.
#[inline]
pub(crate) fn small_angle_sq<T: Numeric>() -> T {
    T::from_f64(1e-12)
}
