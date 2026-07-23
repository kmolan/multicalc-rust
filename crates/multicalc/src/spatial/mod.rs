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

/// Angle threshold below which trig ratios switch to their Taylor series.
/// Scaled as `30 · EPSILON` so each scalar type gets a type-appropriate cutoff.
#[inline]
pub(crate) fn small_angle<T: Numeric>() -> T {
    T::EPSILON_X30
}

/// Squared small-angle threshold, also `30 · EPSILON`, for branches on θ² / ‖v‖² before
/// `sqrt`, and for proximity-to-±1 checks (e.g. Euler gimbal lock).
#[inline]
pub(crate) fn small_angle_sq<T: Numeric>() -> T {
    T::EPSILON_X30
}
