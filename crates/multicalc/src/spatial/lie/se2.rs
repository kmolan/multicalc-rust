//! The 2D rigid-body transform group SE(2).

use core::ops::Mul;

use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::spatial::lie::SO2;
use crate::spatial::small_angle_sq;

/// A 2D rigid-body transform: a rotation and a translation. The tangent is `[vx, vy, ω]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SE2<T: Numeric> {
    rotation: SO2<T>,
    translation: Vector<2, T>,
}

impl<T: Numeric> SE2<T> {
    /// The identity transform.
    #[inline]
    pub fn identity() -> Self {
        SE2 {
            rotation: SO2::identity(),
            translation: Vector::zeros(),
        }
    }

    /// A transform from a rotation and translation.
    #[inline]
    pub fn from_parts(rotation: SO2<T>, translation: Vector<2, T>) -> Self {
        SE2 {
            rotation,
            translation,
        }
    }

    /// The rotation part.
    #[inline]
    pub fn rotation(self) -> SO2<T> {
        self.rotation
    }

    /// The translation part.
    #[inline]
    pub fn translation(self) -> Vector<2, T> {
        self.translation
    }

    /// Composition (also available as `*`).
    #[inline]
    pub fn compose(self, rhs: Self) -> Self {
        SE2 {
            rotation: self.rotation.compose(rhs.rotation),
            translation: self.rotation.act(rhs.translation) + self.translation,
        }
    }

    /// The inverse transform.
    #[inline]
    pub fn inverse(self) -> Self {
        let r_inv = self.rotation.inverse();
        SE2 {
            rotation: r_inv,
            translation: -r_inv.act(self.translation),
        }
    }

    /// Applies the transform to a 2D point.
    #[inline]
    pub fn act(self, p: Vector<2, T>) -> Vector<2, T> {
        self.rotation.act(p) + self.translation
    }

    /// The exponential map from a `[vx, vy, ω]` twist. Near ω = 0 the `V(θ)` block uses a Taylor
    /// series, keeping the value and its derivative finite.
    #[inline]
    pub fn exp(xi: Vector<3, T>) -> Self {
        let [vx, vy, omega] = *xi.as_array();
        let theta_sq = omega * omega;
        let (a, b) = if theta_sq < small_angle_sq::<T>() {
            (
                T::ONE - theta_sq / T::from_f64(6.0),
                omega * (T::HALF - theta_sq / T::from_f64(24.0)),
            )
        } else {
            (omega.sin() / omega, (T::ONE - omega.cos()) / omega)
        };
        let translation = Vector::new([a * vx - b * vy, b * vx + a * vy]);
        SE2 {
            rotation: SO2::exp(omega),
            translation,
        }
    }

    /// The logarithm, the inverse of [`SE2::exp`], returning `[vx, vy, ω]`.
    #[inline]
    pub fn log(self) -> Vector<3, T> {
        let omega = self.rotation.log();
        let theta_sq = omega * omega;
        let (alpha, beta) = if theta_sq < small_angle_sq::<T>() {
            (T::ONE - theta_sq / T::from_f64(12.0), omega * T::HALF)
        } else {
            let half = omega * T::HALF;
            (half * (half.cos() / half.sin()), half)
        };
        let [tx, ty] = *self.translation.as_array();
        Vector::new([alpha * tx + beta * ty, -beta * tx + alpha * ty, omega])
    }

    /// The 3×3 adjoint for the `[v; ω]` ordering.
    #[inline]
    pub fn adjoint(self) -> Matrix<3, 3, T> {
        let (c, s) = self.rotation.cos_sin();
        let [tx, ty] = *self.translation.as_array();
        Matrix::new([[c, -s, ty], [s, c, -tx], [T::ZERO, T::ZERO, T::ONE]])
    }

    /// The Lie-algebra element for a `[vx, vy, ω]` twist.
    #[inline]
    pub fn hat(xi: Vector<3, T>) -> Matrix<3, 3, T> {
        let [vx, vy, omega] = *xi.as_array();
        Matrix::new([
            [T::ZERO, -omega, vx],
            [omega, T::ZERO, vy],
            [T::ZERO, T::ZERO, T::ZERO],
        ])
    }

    /// The inverse of [`SE2::hat`].
    #[inline]
    pub fn vee(m: Matrix<3, 3, T>) -> Vector<3, T> {
        let [[_, _, m02], [m10, _, m12], _] = m.into_array();
        Vector::new([m02, m12, m10])
    }

    /// The 3×3 homogeneous transform matrix.
    #[inline]
    pub fn to_matrix(self) -> Matrix<3, 3, T> {
        let (c, s) = self.rotation.cos_sin();
        let [tx, ty] = *self.translation.as_array();
        Matrix::new([[c, -s, tx], [s, c, ty], [T::ZERO, T::ZERO, T::ONE]])
    }

    /// Builds a transform from a 3×3 homogeneous matrix; `None` if the rotation block is degenerate.
    #[inline]
    pub fn try_from_matrix(m: Matrix<3, 3, T>) -> Option<Self> {
        let [[m00, m01, m02], [m10, m11, m12], _] = m.into_array();
        let r = SO2::try_from_matrix(Matrix::new([[m00, m01], [m10, m11]]))?;
        Some(SE2 {
            rotation: r,
            translation: Vector::new([m02, m12]),
        })
    }

    /// Geodesic interpolation; `t = 0` gives `self`, `t = 1` gives `other`.
    #[inline]
    pub fn interpolate(self, other: Self, t: T) -> Self {
        self.compose(Self::exp(self.inverse().compose(other).log() * t))
    }

    /// The SE(2) left Jacobian `J_l(ξ) = [[V(θ), q], [0, 1]]` for the `[vx, vy, ω]` ordering. The
    /// coupling column `q` comes from the se(2) adjoint series; `p` and `r` use a Taylor series in
    /// θ² near ω = 0 so the value and its derivative stay finite.
    ///
    /// ```
    /// use multicalc::spatial::SE2;
    /// use multicalc::linear_algebra::Vector;
    /// let xi = Vector::new([0.4_f64, -0.2, 0.3]);
    /// let prod = (SE2::left_jacobian(xi) * SE2::left_jacobian_inverse(xi)).into_array();
    /// for i in 0..3 { assert!((prod[i][i] - 1.0).abs() < 1e-12); }
    /// ```
    #[inline]
    pub fn left_jacobian(xi: Vector<3, T>) -> Matrix<3, 3, T> {
        let [rx, ry, omega] = *xi.as_array();
        let theta_sq = omega * omega;
        // a = sinθ/θ, b = (1−cosθ)/θ (the V(θ) block); p = (1−cosθ)/θ², r = (θ−sinθ)/θ² (q).
        let (a, b, p, r) = if theta_sq < small_angle_sq::<T>() {
            (
                T::ONE - theta_sq / T::from_f64(6.0),
                omega * (T::HALF - theta_sq / T::from_f64(24.0)),
                T::HALF - theta_sq / T::from_f64(24.0),
                omega * (T::ONE / T::from_f64(6.0) - theta_sq / T::from_f64(120.0)),
            )
        } else {
            let (s, c) = (omega.sin(), omega.cos());
            (
                s / omega,
                (T::ONE - c) / omega,
                (T::ONE - c) / theta_sq,
                (omega - s) / theta_sq,
            )
        };
        let qx = p * ry + r * rx;
        let qy = r * ry - p * rx;
        Matrix::new([[a, -b, qx], [b, a, qy], [T::ZERO, T::ZERO, T::ONE]])
    }

    /// The SE(2) right Jacobian `J_r(ξ) = J_l(−ξ)`.
    #[inline]
    pub fn right_jacobian(xi: Vector<3, T>) -> Matrix<3, 3, T> {
        Self::left_jacobian(-xi)
    }

    /// The inverse SE(2) left Jacobian `J_l⁻¹(ξ) = [[V⁻¹, −V⁻¹·q], [0, 1]]`, with `q` the same
    /// coupling column as [`SE2::left_jacobian`] and `V⁻¹` the `alpha, beta` block from [`SE2::log`].
    #[inline]
    pub fn left_jacobian_inverse(xi: Vector<3, T>) -> Matrix<3, 3, T> {
        let [rx, ry, omega] = *xi.as_array();
        let theta_sq = omega * omega;
        // p = (1−cosθ)/θ², r = (θ−sinθ)/θ²: the coupling coefficients of the forward Jacobian.
        let (p, r) = if theta_sq < small_angle_sq::<T>() {
            (
                T::HALF - theta_sq / T::from_f64(24.0),
                omega * (T::ONE / T::from_f64(6.0) - theta_sq / T::from_f64(120.0)),
            )
        } else {
            let (s, c) = (omega.sin(), omega.cos());
            ((T::ONE - c) / theta_sq, (omega - s) / theta_sq)
        };
        let qx = p * ry + r * rx;
        let qy = r * ry - p * rx;
        let (alpha, beta) = if theta_sq < small_angle_sq::<T>() {
            (T::ONE - theta_sq / T::from_f64(12.0), omega * T::HALF)
        } else {
            let half = omega * T::HALF;
            (half * (half.cos() / half.sin()), half)
        };
        let cx = -(alpha * qx + beta * qy);
        let cy = -(-beta * qx + alpha * qy);
        Matrix::new([
            [alpha, beta, cx],
            [-beta, alpha, cy],
            [T::ZERO, T::ZERO, T::ONE],
        ])
    }

    /// The inverse SE(2) right Jacobian `J_r⁻¹(ξ) = J_l⁻¹(−ξ)`.
    #[inline]
    pub fn right_jacobian_inverse(xi: Vector<3, T>) -> Matrix<3, 3, T> {
        Self::left_jacobian_inverse(-xi)
    }
}

impl<T: Numeric> Mul for SE2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.compose(rhs)
    }
}
