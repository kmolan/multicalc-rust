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
pub(crate) fn skew3<T: Numeric>(vector: Vector3D<T>) -> Matrix3D<T> {
    let [x, y, z] = *vector.as_array();
    Matrix::new([[T::ZERO, -z, y], [z, T::ZERO, -x], [-y, x, T::ZERO]])
}

/// The SO(3) left Jacobian `J_l(φ) = I + c1·[φ]× + c2·[φ]×²`. Near θ = 0 the coefficients use a
/// Taylor series in θ², so the value and its derivative stay finite at φ = 0. Finite at θ = π.
#[inline]
pub(crate) fn left_jacobian_so3<T: Numeric>(phi: Vector3D<T>) -> Matrix3D<T> {
    let theta_sq = phi.dot(phi);
    let skew = skew3(phi);
    let skew_sq = skew * skew;
    let (thresh1, thresh2) = small_angle_so3_sq::<T>();
    let theta = theta_sq.sqrt();

    let coeff1 = if theta_sq < thresh1 {
        T::HALF - theta_sq / T::from_f64(24.0)
    } else {
        (T::ONE - theta.cos()) / theta_sq
    };

    let coeff2 = if theta_sq < thresh2 {
        T::ONE / T::from_f64(6.0) - theta_sq / T::from_f64(120.0)
    } else {
        (theta - theta.sin()) / (theta_sq * theta)
    };

    Matrix::identity() + skew.scale(coeff1) + skew_sq.scale(coeff2)
}

/// The inverse SO(3) left Jacobian `J_l⁻¹(φ) = I − ½·[φ]× + c3·[φ]×²`. The `cot(θ/2)` coefficient
/// is finite for θ ∈ (0, π], so only θ = 0 needs the Taylor series (θ = π needs no special case).
#[inline]
pub(crate) fn inverse_left_jacobian_so3<T: Numeric>(phi: Vector3D<T>) -> Matrix3D<T> {
    let theta_sq = phi.dot(phi);
    let skew = skew3(phi);
    let skew_sq = skew * skew;
    let coeff3 = if theta_sq < small_angle_inverse_so3_sq::<T>() {
        T::ONE / T::from_f64(12.0) + theta_sq / T::from_f64(720.0)
    } else {
        let theta = theta_sq.sqrt();
        let half = theta * T::HALF;
        (T::ONE - half * (half.cos() / half.sin())) / theta_sq
    };
    Matrix::identity() - skew.scale(T::HALF) + skew_sq.scale(coeff3)
}

/// The Barfoot SE(3) `Q(ρ, φ)` block (Eq. 7.86) used by the 6×6 left Jacobian. Near θ = 0 the
/// coefficients use a Taylor series in θ², keeping the value and its derivative finite at φ = 0.
#[inline]
pub(crate) fn q_matrix_se3<T: Numeric>(rho: Vector3D<T>, phi: Vector3D<T>) -> Matrix3D<T> {
    let theta_sq = phi.dot(phi);
    let rho_skew = skew3(rho);
    let phi_skew = skew3(phi);
    // Each ratio cancels at its own angle, so each gets its own cutoff. Sharing one would force
    // the earliest-switching ratio onto its series far past its crossover, and truncation grows
    // as θ⁴.
    let (thresh_c2, thresh_c3, thresh_c5) = small_angle_se3_sq::<T>();
    let theta = theta_sq.sqrt();
    let theta3 = theta_sq * theta;
    let theta4 = theta_sq * theta_sq;
    let theta5 = theta4 * theta;

    let coeff2 = if theta_sq < thresh_c2 {
        T::ONE / T::from_f64(6.0) - theta_sq / T::from_f64(120.0)
    } else {
        (theta - theta.sin()) / theta3
    };

    let coeff3 = if theta_sq < thresh_c3 {
        -T::ONE / T::from_f64(24.0) + theta_sq / T::from_f64(720.0)
    } else {
        (T::ONE - theta_sq * T::HALF - theta.cos()) / theta4
    };

    let coeff5 = if theta_sq < thresh_c5 {
        -T::ONE / T::from_f64(120.0) + theta_sq / T::from_f64(5040.0)
    } else {
        (theta - theta.sin() - theta3 / T::from_f64(6.0)) / theta5
    };

    let coeff4 = (coeff3 - T::from_f64(3.0) * coeff5) * T::HALF;
    let phi_rho_phi = phi_skew * rho_skew * phi_skew; // Φ P Φ, reused in two terms
    let term2 = phi_skew * rho_skew + rho_skew * phi_skew + phi_rho_phi; // ΦP + PΦ + ΦPΦ
    let term3 = phi_skew * phi_skew * rho_skew + rho_skew * phi_skew * phi_skew
        - phi_rho_phi.scale(T::from_f64(3.0)); // Φ²P + PΦ² − 3ΦPΦ
    let term4 =
        phi_skew * rho_skew * phi_skew * phi_skew + phi_skew * phi_skew * rho_skew * phi_skew; // ΦPΦ² + Φ²PΦ
    rho_skew.scale(T::HALF) + term2.scale(coeff2) - term3.scale(coeff3) - term4.scale(coeff4)
}

/// The SE(3) left Jacobian `J_l(ξ) = [[J, Q], [0, J]]` for the `[v; ω]` ordering, with `J` the
/// SO(3) left Jacobian of the rotation part and `Q` the Barfoot block.
#[inline]
pub(crate) fn left_jacobian_se3<T: Numeric>(twist: Vector6D<T>) -> Matrix6D<T> {
    let [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z] = *twist.as_array();
    let rho = Vector::new([lin_x, lin_y, lin_z]);
    let phi = Vector::new([ang_x, ang_y, ang_z]);
    let j = left_jacobian_so3(phi);
    let q_block = q_matrix_se3(rho, phi);
    Matrix::from_fn(|i, k| {
        if i < 3 && k < 3 {
            j[(i, k)]
        } else if i < 3 {
            q_block[(i, k - 3)]
        } else if k >= 3 {
            j[(i - 3, k - 3)]
        } else {
            T::ZERO
        }
    })
}

/// The inverse SE(3) left Jacobian `J_l⁻¹(ξ) = [[Jᵢ, −Jᵢ·Q·Jᵢ], [0, Jᵢ]]`.
#[inline]
pub(crate) fn inverse_left_jacobian_se3<T: Numeric>(twist: Vector6D<T>) -> Matrix6D<T> {
    let [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z] = *twist.as_array();
    let rho = Vector::new([lin_x, lin_y, lin_z]);
    let phi = Vector::new([ang_x, ang_y, ang_z]);
    let j_inv = inverse_left_jacobian_so3(phi);
    let q_block = q_matrix_se3(rho, phi);
    let top_right = -(j_inv * q_block * j_inv);
    Matrix::from_fn(|i, k| {
        if i < 3 && k < 3 {
            j_inv[(i, k)]
        } else if i < 3 {
            top_right[(i, k - 3)]
        } else if k >= 3 {
            j_inv[(i - 3, k - 3)]
        } else {
            T::ZERO
        }
    })
}
