//! Lie groups for 2D and 3D rotations and rigid-body transforms.
//!
//! - [`SO2`] / [`SO3`] — rotations.
//! - [`SE2`] / [`SE3`] — rigid transforms (rotation + translation).
//!
//! Each provides `identity`, `compose` (also `*`), `inverse`, `act` on a point, `exp`/`log`,
//! `hat`/`vee`, `adjoint`, geodesic `interpolate`, and matrix conversions.
//!
//! Conventions: the tangent ordering is `[v; ω]` (linear first) for `SE2`/`SE3`; the retract is
//! right-perturbation `X · exp(ξ)`, so `interpolate(a, b, t) = a · exp(t · log(a⁻¹·b))`. Angles are
//! radians. `SO3` wraps a unit [`Quaternion`](crate::spatial::Quaternion).

mod se2;
mod se3;
mod so2;
mod so3;

pub use se2::SE2;
pub use se3::SE3;
pub use so2::SO2;
pub use so3::SO3;

use crate::linear_algebra::{Matrix, Matrix3D, Matrix6D, Vector, Vector3D, Vector6D};
use crate::scalar::Numeric;
use crate::spatial::{small_angle_inverse_so3_sq, small_angle_se3_sq, small_angle_so3_sq};

/// The 3×3 skew-symmetric matrix `[v]×`, so that `[v]× · p = v × p`.
#[inline]
pub(crate) fn skew3<T: Numeric>(v: Vector3D<T>) -> Matrix3D<T> {
    let [x, y, z] = *v.as_array();
    Matrix::new([[T::ZERO, -z, y], [z, T::ZERO, -x], [-y, x, T::ZERO]])
}

/// The SO(3) left Jacobian `J_l(φ) = I + c1·[φ]× + c2·[φ]×²`. Near θ = 0 the coefficients use a
/// Taylor series in θ², so the value and its derivative stay finite at φ = 0. Finite at θ = π.
#[inline]
pub(crate) fn left_jacobian_so3<T: Numeric>(phi: Vector3D<T>) -> Matrix3D<T> {
    let theta_sq = phi.dot(phi);
    let s = skew3(phi);
    let s2 = s * s;
    let (t1, t2) = small_angle_so3_sq::<T>();
    let theta = theta_sq.sqrt();

    let c1 = if theta_sq < t1 {
        T::HALF - theta_sq / T::from_f64(24.0)
    } else {
        (T::ONE - theta.cos()) / theta_sq
    };

    let c2 = if theta_sq < t2 {
        T::ONE / T::from_f64(6.0) - theta_sq / T::from_f64(120.0)
    } else {
        (theta - theta.sin()) / (theta_sq * theta)
    };

    Matrix::identity() + s.scale(c1) + s2.scale(c2)
}

/// The inverse SO(3) left Jacobian `J_l⁻¹(φ) = I − ½·[φ]× + c3·[φ]×²`. The `cot(θ/2)` coefficient
/// is finite for θ ∈ (0, π], so only θ = 0 needs the Taylor series (θ = π needs no special case).
#[inline]
pub(crate) fn inverse_left_jacobian_so3<T: Numeric>(phi: Vector3D<T>) -> Matrix3D<T> {
    let theta_sq = phi.dot(phi);
    let s = skew3(phi);
    let s2 = s * s;
    let c3 = if theta_sq < small_angle_inverse_so3_sq::<T>() {
        T::ONE / T::from_f64(12.0) + theta_sq / T::from_f64(720.0)
    } else {
        let theta = theta_sq.sqrt();
        let half = theta * T::HALF;
        (T::ONE - half * (half.cos() / half.sin())) / theta_sq
    };
    Matrix::identity() - s.scale(T::HALF) + s2.scale(c3)
}

/// The Barfoot SE(3) `Q(ρ, φ)` block (Eq. 7.86) used by the 6×6 left Jacobian. Near θ = 0 the
/// coefficients use a Taylor series in θ², keeping the value and its derivative finite at φ = 0.
#[inline]
pub(crate) fn q_matrix_se3<T: Numeric>(rho: Vector3D<T>, phi: Vector3D<T>) -> Matrix3D<T> {
    let theta_sq = phi.dot(phi);
    let p = skew3(rho);
    let ph = skew3(phi);
    // Each ratio cancels at its own angle, so each gets its own cutoff. Sharing one would force
    // the earliest-switching ratio onto its series far past its crossover, and truncation grows
    // as θ⁴.
    let (t_c2, t_c3, t_c5) = small_angle_se3_sq::<T>();
    let theta = theta_sq.sqrt();
    let theta3 = theta_sq * theta;
    let theta4 = theta_sq * theta_sq;
    let theta5 = theta4 * theta;

    let c2 = if theta_sq < t_c2 {
        T::ONE / T::from_f64(6.0) - theta_sq / T::from_f64(120.0)
    } else {
        (theta - theta.sin()) / theta3
    };

    let c3 = if theta_sq < t_c3 {
        -T::ONE / T::from_f64(24.0) + theta_sq / T::from_f64(720.0)
    } else {
        (T::ONE - theta_sq * T::HALF - theta.cos()) / theta4
    };

    let c5 = if theta_sq < t_c5 {
        -T::ONE / T::from_f64(120.0) + theta_sq / T::from_f64(5040.0)
    } else {
        (theta - theta.sin() - theta3 / T::from_f64(6.0)) / theta5
    };

    let c4 = (c3 - T::from_f64(3.0) * c5) * T::HALF;
    let phph = ph * p * ph; // Φ P Φ, reused in two terms
    let t2 = ph * p + p * ph + phph; // ΦP + PΦ + ΦPΦ
    let t3 = ph * ph * p + p * ph * ph - phph.scale(T::from_f64(3.0)); // Φ²P + PΦ² − 3ΦPΦ
    let t4 = ph * p * ph * ph + ph * ph * p * ph; // ΦPΦ² + Φ²PΦ
    p.scale(T::HALF) + t2.scale(c2) - t3.scale(c3) - t4.scale(c4)
}

/// The SE(3) left Jacobian `J_l(ξ) = [[J, Q], [0, J]]` for the `[v; ω]` ordering, with `J` the
/// SO(3) left Jacobian of the rotation part and `Q` the Barfoot block.
#[inline]
pub(crate) fn left_jacobian_se3<T: Numeric>(xi: Vector6D<T>) -> Matrix6D<T> {
    let [rx, ry, rz, px, py, pz] = *xi.as_array();
    let rho = Vector::new([rx, ry, rz]);
    let phi = Vector::new([px, py, pz]);
    let j = left_jacobian_so3(phi);
    let q = q_matrix_se3(rho, phi);
    Matrix::from_fn(|i, k| {
        if i < 3 && k < 3 {
            j[(i, k)]
        } else if i < 3 {
            q[(i, k - 3)]
        } else if k >= 3 {
            j[(i - 3, k - 3)]
        } else {
            T::ZERO
        }
    })
}

/// The inverse SE(3) left Jacobian `J_l⁻¹(ξ) = [[Jᵢ, −Jᵢ·Q·Jᵢ], [0, Jᵢ]]`.
#[inline]
pub(crate) fn inverse_left_jacobian_se3<T: Numeric>(xi: Vector6D<T>) -> Matrix6D<T> {
    let [rx, ry, rz, px, py, pz] = *xi.as_array();
    let rho = Vector::new([rx, ry, rz]);
    let phi = Vector::new([px, py, pz]);
    let ji = inverse_left_jacobian_so3(phi);
    let q = q_matrix_se3(rho, phi);
    let top_right = -(ji * q * ji);
    Matrix::from_fn(|i, k| {
        if i < 3 && k < 3 {
            ji[(i, k)]
        } else if i < 3 {
            top_right[(i, k - 3)]
        } else if k >= 3 {
            ji[(i - 3, k - 3)]
        } else {
            T::ZERO
        }
    })
}
