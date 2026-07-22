//! Spatial math: rotations, Lie groups, and spatial-algebra types.
//!
//! - [`Quaternion`] — unit-quaternion rotations.
//! - [`SO2`] / [`SO3`] / [`SE2`] / [`SE3`] — 2D/3D rotation and rigid-transform Lie groups.
//! - [`Twist`] / [`Wrench`] — spatial velocity and force in `[v; ω]` / `[force; torque]` ordering.

use crate::scalar::Numeric;

pub mod lie;
pub mod quaternion;
pub mod twist;
pub mod wrench;

pub use lie::{SE2, SE3, SO2, SO3};
pub use quaternion::Quaternion;
pub use twist::Twist;
pub use wrench::Wrench;

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
