//! The 2D rigid-body transform group SE(2).

use core::ops::Mul;

use crate::linear_algebra::{Matrix, Matrix3D, Vector, Vector2D, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::lie::SO2;
use crate::spatial::small_angle_sq;

/// A 2D rigid-body transform: a rotation and a translation. The tangent is `[vx, vy, ω]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SE2<T: Numeric = f64> {
    rotation: SO2<T>,
    translation: Vector2D<T>,
}

impl<T: Numeric> SE2<T> {
    /// The identity transform.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        SE2 {
            rotation: SO2::identity(),
            translation: Vector::zeros(),
        }
    }

    /// A transform from a rotation and translation.
    #[inline]
    #[must_use]
    pub fn from_parts(rotation: SO2<T>, translation: Vector2D<T>) -> Self {
        SE2 {
            rotation,
            translation,
        }
    }

    /// The rotation part.
    #[inline]
    #[must_use]
    pub fn rotation(self) -> SO2<T> {
        self.rotation
    }

    /// The translation part.
    #[inline]
    pub fn translation(self) -> Vector2D<T> {
        self.translation
    }

    /// Composition (also available as `*`).
    #[inline]
    #[must_use]
    pub fn compose(self, rhs: Self) -> Self {
        SE2 {
            rotation: self.rotation.compose(rhs.rotation),
            translation: self.rotation.act(rhs.translation) + self.translation,
        }
    }

    /// The inverse transform.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        let r_inv = self.rotation.inverse();
        SE2 {
            rotation: r_inv,
            translation: -r_inv.act(self.translation),
        }
    }

    /// Applies the transform to a 2D point.
    #[inline]
    pub fn act(self, point: Vector2D<T>) -> Vector2D<T> {
        self.rotation.act(point) + self.translation
    }

    /// The exponential map from a `[vx, vy, ω]` twist. Near ω = 0 the `V(θ)` block uses a Taylor
    /// series, keeping the value and its derivative finite.
    #[inline]
    #[must_use]
    pub fn exp(twist: Vector3D<T>) -> Self {
        let [lin_x, lin_y, omega] = *twist.as_array();
        let theta_sq = omega * omega;
        let (a, b) = if theta_sq < small_angle_sq::<T>() {
            (
                T::ONE - theta_sq / T::from_f64(6.0),
                omega * (T::HALF - theta_sq / T::from_f64(24.0)),
            )
        } else {
            (omega.sin() / omega, (T::ONE - omega.cos()) / omega)
        };
        let translation = Vector::new([a * lin_x - b * lin_y, b * lin_x + a * lin_y]);
        SE2 {
            rotation: SO2::exp(omega),
            translation,
        }
    }

    /// The logarithm, the inverse of [`SE2::exp`], returning `[vx, vy, ω]`.
    #[inline]
    pub fn log(self) -> Vector3D<T> {
        let omega = self.rotation.log();
        let theta_sq = omega * omega;
        let (alpha, beta) = if theta_sq < small_angle_sq::<T>() {
            (T::ONE - theta_sq / T::from_f64(12.0), omega * T::HALF)
        } else {
            let half = omega * T::HALF;
            (half * (half.cos() / half.sin()), half)
        };
        let [pos_x, pos_y] = *self.translation.as_array();
        Vector::new([
            alpha * pos_x + beta * pos_y,
            -beta * pos_x + alpha * pos_y,
            omega,
        ])
    }

    /// The 3×3 adjoint for the `[v; ω]` ordering.
    #[inline]
    pub fn adjoint(self) -> Matrix3D<T> {
        let (cos, sin) = self.rotation.cos_sin();
        let [pos_x, pos_y] = *self.translation.as_array();
        Matrix::new([
            [cos, -sin, pos_y],
            [sin, cos, -pos_x],
            [T::ZERO, T::ZERO, T::ONE],
        ])
    }

    /// The Lie-algebra element for a `[vx, vy, ω]` twist.
    #[inline]
    pub fn hat(twist: Vector3D<T>) -> Matrix3D<T> {
        let [lin_x, lin_y, omega] = *twist.as_array();
        Matrix::new([
            [T::ZERO, -omega, lin_x],
            [omega, T::ZERO, lin_y],
            [T::ZERO, T::ZERO, T::ZERO],
        ])
    }

    /// The inverse of [`SE2::hat`].
    #[inline]
    pub fn vee(matrix: Matrix3D<T>) -> Vector3D<T> {
        let [[_, _, m02], [m10, _, m12], _] = matrix.into_array();
        Vector::new([m02, m12, m10])
    }

    /// The 3×3 homogeneous transform matrix.
    #[inline]
    pub fn to_matrix(self) -> Matrix3D<T> {
        let (cos, sin) = self.rotation.cos_sin();
        let [pos_x, pos_y] = *self.translation.as_array();
        Matrix::new([
            [cos, -sin, pos_x],
            [sin, cos, pos_y],
            [T::ZERO, T::ZERO, T::ONE],
        ])
    }

    /// Builds a transform from a finite 3×3 homogeneous matrix; `None` if the rotation block is not
    /// a proper unit rotation or the bottom row differs from `[0, 0, 1]` by more than scalar
    /// round-off.
    #[inline]
    #[must_use]
    pub fn try_from_matrix(matrix: Matrix3D<T>) -> Option<Self> {
        let [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] = matrix.into_array();
        if !m02.is_finite()
            || !m12.is_finite()
            || !m20.is_finite()
            || !m21.is_finite()
            || !m22.is_finite()
            || m20.abs() > T::EPSILON_X30
            || m21.abs() > T::EPSILON_X30
            || (m22 - T::ONE).abs() > T::EPSILON_X30
        {
            return None;
        }

        let rotation = SO2::try_from_matrix(Matrix::new([[m00, m01], [m10, m11]]))?;
        Some(SE2 {
            rotation,
            translation: Vector::new([m02, m12]),
        })
    }

    /// Geodesic interpolation; `t = 0` gives `self`, `t = 1` gives `other`.
    #[inline]
    #[must_use]
    pub fn interpolate(self, other: Self, amount: T) -> Self {
        self.compose(Self::exp(self.inverse().compose(other).log() * amount))
    }

    /// The SE(2) left Jacobian `J_l(ξ) = [[V(θ), q], [0, 1]]` for the `[vx, vy, ω]` ordering. The
    /// coupling column `q` comes from the se(2) adjoint series; `p` and `r` use a Taylor series in
    /// θ² near ω = 0 so the value and its derivative stay finite.
    ///
    /// ```
    /// use multicalc::spatial::SE2;
    /// use multicalc::linear_algebra::Vector;
    /// let twist = Vector::new([0.4_f64, -0.2, 0.3]);
    /// let prod = (SE2::left_jacobian(twist) * SE2::left_jacobian_inverse(twist));
    /// for i in 0..3 { assert!((prod[(i, i)] - 1.0).abs() < 1e-12); }
    /// ```
    #[inline]
    pub fn left_jacobian(twist: Vector3D<T>) -> Matrix3D<T> {
        let [rho_x, rho_y, omega] = *twist.as_array();
        let theta_sq = omega * omega;
        // a = sinθ/θ, b = (1−cosθ)/θ (the V(θ) block); p = (1−cosθ)/θ², r = (θ−sinθ)/θ² (q).
        let (a, b, p_coeff, r_coeff) = if theta_sq < small_angle_sq::<T>() {
            (
                T::ONE - theta_sq / T::from_f64(6.0),
                omega * (T::HALF - theta_sq / T::from_f64(24.0)),
                T::HALF - theta_sq / T::from_f64(24.0),
                omega * (T::ONE / T::from_f64(6.0) - theta_sq / T::from_f64(120.0)),
            )
        } else {
            let (sin, cos) = (omega.sin(), omega.cos());
            (
                sin / omega,
                (T::ONE - cos) / omega,
                (T::ONE - cos) / theta_sq,
                (omega - sin) / theta_sq,
            )
        };
        let coupling_x = p_coeff * rho_y + r_coeff * rho_x;
        let coupling_y = r_coeff * rho_y - p_coeff * rho_x;
        Matrix::new([
            [a, -b, coupling_x],
            [b, a, coupling_y],
            [T::ZERO, T::ZERO, T::ONE],
        ])
    }

    /// The SE(2) right Jacobian `J_r(ξ) = J_l(−ξ)`.
    #[inline]
    pub fn right_jacobian(twist: Vector3D<T>) -> Matrix3D<T> {
        Self::left_jacobian(-twist)
    }

    /// The inverse SE(2) left Jacobian `J_l⁻¹(ξ) = [[V⁻¹, −V⁻¹·q], [0, 1]]`, with `q` the same
    /// coupling column as [`SE2::left_jacobian`] and `V⁻¹` the `alpha, beta` block from [`SE2::log`].
    #[inline]
    pub fn left_jacobian_inverse(twist: Vector3D<T>) -> Matrix3D<T> {
        let [rho_x, rho_y, omega] = *twist.as_array();
        let theta_sq = omega * omega;
        // p = (1−cosθ)/θ², r = (θ−sinθ)/θ²: the coupling coefficients of the forward Jacobian.
        let (p_coeff, r_coeff) = if theta_sq < small_angle_sq::<T>() {
            (
                T::HALF - theta_sq / T::from_f64(24.0),
                omega * (T::ONE / T::from_f64(6.0) - theta_sq / T::from_f64(120.0)),
            )
        } else {
            let (sin, cos) = (omega.sin(), omega.cos());
            ((T::ONE - cos) / theta_sq, (omega - sin) / theta_sq)
        };
        let coupling_x = p_coeff * rho_y + r_coeff * rho_x;
        let coupling_y = r_coeff * rho_y - p_coeff * rho_x;
        let (alpha, beta) = if theta_sq < small_angle_sq::<T>() {
            (T::ONE - theta_sq / T::from_f64(12.0), omega * T::HALF)
        } else {
            let half = omega * T::HALF;
            (half * (half.cos() / half.sin()), half)
        };
        let corr_x = -(alpha * coupling_x + beta * coupling_y);
        let corr_y = -(-beta * coupling_x + alpha * coupling_y);
        Matrix::new([
            [alpha, beta, corr_x],
            [-beta, alpha, corr_y],
            [T::ZERO, T::ZERO, T::ONE],
        ])
    }

    /// The inverse SE(2) right Jacobian `J_r⁻¹(ξ) = J_l⁻¹(−ξ)`.
    #[inline]
    pub fn right_jacobian_inverse(twist: Vector3D<T>) -> Matrix3D<T> {
        Self::left_jacobian_inverse(-twist)
    }
}

impl<T: Numeric> Mul for SE2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.compose(rhs)
    }
}

impl<T: Numeric> Default for SE2<T> {
    /// Returns the identity as default.
    ///
    /// ```
    /// use multicalc::{SE2, SO2, Vector2D};
    ///
    /// let default_se2 = SE2::default();
    /// let se2 = SE2::from_parts(
    ///     SO2::from_angle(0.3), Vector2D::new([1.0, 2.0])
    /// );
    ///
    /// assert_eq!(default_se2 * se2, se2);
    /// ```
    fn default() -> Self {
        Self::identity()
    }
}
