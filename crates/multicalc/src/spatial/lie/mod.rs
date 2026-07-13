//! Lie groups for 2D and 3D rotations and rigid-body transforms.
//!
//! [`SO2`]/[`SO3`] are rotations; [`SE2`]/[`SE3`] are rigid transforms (rotation + translation).
//! Each provides `identity`, `compose` (also `*`), `inverse`, `act` on a point, `exp`/`log`,
//! `hat`/`vee`, `adjoint`, geodesic `interpolate`, and matrix conversions.
//!
//! Conventions: the tangent ordering is `[v; ω]` (linear part first) for `SE2`/`SE3`; the retract
//! is right-perturbation `X · exp(ξ)`, so `interpolate(a, b, t) = a · exp(t · log(a⁻¹·b))`. Angles
//! are radians. `SO3` wraps a unit [`Quaternion`](crate::spatial::Quaternion) and carries the
//! unit-rotation invariant.

mod se2;
mod se3;
mod so2;
mod so3;

pub use se2::SE2;
pub use se3::SE3;
pub use so2::SO2;
pub use so3::SO3;

use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::spatial::small_angle_sq;

/// The 3×3 skew-symmetric matrix `[v]×`, so that `[v]× · p = v × p`.
#[inline]
pub(crate) fn skew3<T: Numeric>(v: Vector<3, T>) -> Matrix<3, 3, T> {
    Matrix::new([
        [T::ZERO, -v[2], v[1]],
        [v[2], T::ZERO, -v[0]],
        [-v[1], v[0], T::ZERO],
    ])
}

/// The SO(3) left Jacobian `J_l(φ) = I + c1·[φ]× + c2·[φ]×²`. Near θ = 0 the coefficients use a
/// Taylor series in θ², so the value and its derivative stay finite at φ = 0. Finite at θ = π.
#[inline]
pub(crate) fn left_jacobian_so3<T: Numeric>(phi: Vector<3, T>) -> Matrix<3, 3, T> {
    let theta_sq = phi.dot(phi);
    let s = skew3(phi);
    let s2 = s * s;
    let (c1, c2) = if theta_sq < small_angle_sq::<T>() {
        (
            T::HALF - theta_sq / T::from_f64(24.0),
            T::ONE / T::from_f64(6.0) - theta_sq / T::from_f64(120.0),
        )
    } else {
        let theta = theta_sq.sqrt();
        (
            (T::ONE - theta.cos()) / theta_sq,
            (theta - theta.sin()) / (theta_sq * theta),
        )
    };
    Matrix::identity() + s.scale(c1) + s2.scale(c2)
}

/// The inverse SO(3) left Jacobian `J_l⁻¹(φ) = I − ½·[φ]× + c3·[φ]×²`. The `cot(θ/2)` coefficient
/// is finite for θ ∈ (0, π], so only θ = 0 needs the Taylor series (θ = π needs no special case).
#[inline]
pub(crate) fn inverse_left_jacobian_so3<T: Numeric>(phi: Vector<3, T>) -> Matrix<3, 3, T> {
    let theta_sq = phi.dot(phi);
    let s = skew3(phi);
    let s2 = s * s;
    let c3 = if theta_sq < small_angle_sq::<T>() {
        T::ONE / T::from_f64(12.0) + theta_sq / T::from_f64(720.0)
    } else {
        let theta = theta_sq.sqrt();
        let half = theta * T::HALF;
        (T::ONE - half * (half.cos() / half.sin())) / theta_sq
    };
    Matrix::identity() - s.scale(T::HALF) + s2.scale(c3)
}
