//! Spatial math: rotations, Lie groups, and spatial-algebra types.
//!
//! - [`Quaternion`] — unit-quaternion rotations.
//! - [`SO2`] / [`SO3`] / [`SE2`] / [`SE3`] — 2D/3D rotation and rigid-transform Lie groups.
//! - [`Twist`] / [`Wrench`] — spatial velocity and force in `[v; ω]` / `[force; torque]` ordering.
//! - [`SpatialInertia`] — a body's mass, centre of mass, and rotational inertia.
//! - [`FreeJointState`] — the pose and velocity of a body free to move in all six directions.

use crate::scalar::Numeric;

mod free_joint;
mod inertia;
mod lie;
mod quaternion;
mod twist;
mod wrench;

pub use free_joint::FreeJointState;
pub use inertia::SpatialInertia;
pub use lie::{SE2, SE3, SO2, SO3};
pub use quaternion::Quaternion;
pub use twist::Twist;
pub use wrench::Wrench;

/// Angle threshold below which trig ratios switch to their Taylor series.
/// Also used for eps-scale proximity checks (e.g. Euler gimbal lock vs ±1).
/// Scaled as `30 · EPSILON` so each scalar type gets a type-appropriate cutoff.
#[inline]
#[must_use]
pub(crate) fn small_angle<T: Numeric>() -> T {
    T::EPSILON_X30
}

/// Squared small-angle threshold, `(30 · EPSILON)²`, for branches on θ² / ‖v‖² before
/// `sqrt`.
#[inline]
#[must_use]
pub(crate) fn small_angle_sq<T: Numeric>() -> T {
    small_angle::<T>() * small_angle::<T>()
}

/// Small-angle thresholds for left jacobian so3
/// thresh1 = (360*epsilon)^(1/6)
/// thresh2 = (2520*epsilon)^(1/6)
#[inline]
#[must_use]
pub(crate) fn small_angle_so3_sq<T: Numeric>() -> (T, T) {
    let thresh1 = (T::from_f64(360.0) * T::EPSILON).cbrt();
    let thresh2 = (T::from_f64(2520.0) * T::EPSILON).cbrt();
    (thresh1, thresh2)
}

/// Small-angle threshold for inverse left jacobian so3
/// t = (16*945*epsilon)^(1/6)
#[inline]
#[must_use]
pub(crate) fn small_angle_inverse_so3_sq<T: Numeric>() -> T {
    (T::from_f64(15_120.0) * T::EPSILON).cbrt()
}

/// Small-angle threshold for q matrix se3
/// The function has two thresholds, take the largest
/// thresh2 = (2520*espilon)^(1/6)
/// thresh3 = (12*1680*espilon)^(1/8)
/// thresh5 = (0.5*9!*epsilon)^(1/8)
#[inline]
#[must_use]
pub(crate) fn small_angle_se3_sq<T: Numeric>() -> (T, T, T) {
    let thresh2 = (T::from_f64(2520.0) * T::EPSILON).cbrt();
    let thresh3 = (T::from_f64(20_160.0) * T::EPSILON).sqrt().sqrt();
    let thresh5 = (T::from_f64(181_440.0) * T::EPSILON).sqrt().sqrt();
    (thresh2, thresh3, thresh5)
}
