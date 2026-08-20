//! The 2D rotation group SO(2).

use core::ops::Mul;

use crate::linear_algebra::{Matrix, Matrix2D, Vector, Vector2D};
use crate::scalar::Numeric;

/// A 2D rotation, stored as a unit complex number `(cosθ, sinθ)`. Composition is a complex product,
/// so it takes no trigonometry. The group is abelian; `exp`/`log` are exact and need no fallback.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SO2<T: Numeric = f64> {
    cos: T,
    sin: T,
}

impl<T: Numeric> SO2<T> {
    /// The zero rotation.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        SO2 {
            cos: T::ONE,
            sin: T::ZERO,
        }
    }

    /// The rotation by `theta` radians.
    #[inline]
    #[must_use]
    pub fn from_angle(theta: T) -> Self {
        SO2 {
            cos: theta.cos(),
            sin: theta.sin(),
        }
    }

    /// The `(cos, sin)` components.
    #[inline]
    #[must_use]
    pub fn cos_sin(self) -> (T, T) {
        (self.cos, self.sin)
    }

    /// Composition (also available as `*`).
    #[inline]
    #[must_use]
    pub fn compose(self, rhs: Self) -> Self {
        SO2 {
            cos: self.cos * rhs.cos - self.sin * rhs.sin,
            sin: self.cos * rhs.sin + self.sin * rhs.cos,
        }
    }

    /// The inverse rotation.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        SO2 {
            cos: self.cos,
            sin: -self.sin,
        }
    }

    /// Rotates a 2D point.
    #[inline]
    pub fn act(self, point: Vector2D<T>) -> Vector2D<T> {
        let [point_x, point_y] = *point.as_array();
        Vector::new([
            self.cos * point_x - self.sin * point_y,
            self.sin * point_x + self.cos * point_y,
        ])
    }

    /// The exponential map from the tangent angle.
    #[inline]
    #[must_use]
    pub fn exp(theta: T) -> Self {
        Self::from_angle(theta)
    }

    /// The logarithm, the tangent angle in `(−π, π]`.
    #[inline]
    #[must_use]
    pub fn log(self) -> T {
        self.sin.atan2(self.cos)
    }

    /// The Lie-algebra element `[[0, −θ], [θ, 0]]`.
    #[inline]
    pub fn hat(theta: T) -> Matrix2D<T> {
        Matrix::new([[T::ZERO, -theta], [theta, T::ZERO]])
    }

    /// The inverse of [`SO2::hat`].
    #[inline]
    #[must_use]
    pub fn vee(matrix: Matrix2D<T>) -> T {
        let [[_, _], [m10, _]] = matrix.into_array();
        m10
    }

    /// The adjoint, which is `1` (SO(2) is abelian).
    #[inline]
    #[must_use]
    pub fn adjoint(self) -> T {
        T::ONE
    }

    /// The 2×2 rotation matrix.
    #[inline]
    pub fn to_matrix(self) -> Matrix2D<T> {
        Matrix::new([[self.cos, -self.sin], [self.sin, self.cos]])
    }

    /// Builds a rotation from a finite 2×2 matrix sufficiently close to a proper unit rotation,
    /// removing small round-off drift; `None` otherwise.
    #[inline]
    #[must_use]
    pub fn try_from_matrix(matrix: Matrix2D<T>) -> Option<Self> {
        let [[cos, m01], [sin, m11]] = matrix.into_array();
        let n = cos.hypot(sin);
        if !n.is_finite() || n <= T::EPSILON || (n - T::ONE).abs() > T::EPSILON_X30 {
            return None;
        }

        let cos = cos / n;
        let sin = sin / n;
        let second_column_error = (m01 + sin).hypot(m11 - cos);
        if !second_column_error.is_finite() || second_column_error > T::EPSILON_X30 {
            return None;
        }

        Some(SO2 { cos, sin })
    }

    /// Geodesic interpolation; `t = 0` gives `self`, `t = 1` gives `other`.
    #[inline]
    #[must_use]
    pub fn interpolate(self, other: Self, amount: T) -> Self {
        self.compose(Self::exp(self.inverse().compose(other).log() * amount))
    }

    /// The squared norm `c² + s²`.
    #[inline]
    #[must_use]
    fn norm_squared(self) -> T {
        self.cos * self.cos + self.sin * self.sin
    }

    /// The Euclidean norm.
    ///
    /// ```
    /// use multicalc::spatial::SO2;
    ///
    /// let rotation = SO2::<f64>::from_angle(0.3);
    ///
    /// assert!((rotation.norm() - 1.0).abs() <= <f64 as multicalc::Numeric>::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// This rotation renormalized, removing drift accumulated over long composition chains.
    ///
    /// ```
    /// use multicalc::spatial::SO2;
    ///
    /// let rotation = SO2::<f64>::from_angle(0.3);
    /// let normalized = rotation.normalized();
    /// let (cos, sin) = normalized.cos_sin();
    ///
    /// assert!((normalized.norm() - 1.0).abs() <= <f64 as multicalc::Numeric>::EPSILON);
    /// ```
    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        let scale = self.norm().recip();
        SO2 {
            cos: self.cos * scale,
            sin: self.sin * scale,
        }
    }

    /// The SO(2) left Jacobian, which is `1` (SO(2) is abelian).
    #[inline]
    #[must_use]
    pub fn left_jacobian(_theta: T) -> T {
        T::ONE
    }

    /// The SO(2) right Jacobian, which is `1` (SO(2) is abelian).
    #[inline]
    #[must_use]
    pub fn right_jacobian(_theta: T) -> T {
        T::ONE
    }

    /// The inverse SO(2) left Jacobian, which is `1` (SO(2) is abelian).
    #[inline]
    #[must_use]
    pub fn left_jacobian_inverse(_theta: T) -> T {
        T::ONE
    }

    /// The inverse SO(2) right Jacobian, which is `1` (SO(2) is abelian).
    #[inline]
    #[must_use]
    pub fn right_jacobian_inverse(_theta: T) -> T {
        T::ONE
    }
}

impl<T: Numeric> Mul for SO2<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.compose(rhs)
    }
}
impl<T: Numeric> Default for SO2<T> {
    /// Returns the identity as default.
    ///
    /// ```
    /// use multicalc::SO2;
    ///
    /// let default_so2 = SO2::default();
    /// let so2 = SO2::<f64>::from_angle(0.3);
    ///
    /// assert_eq!(so2 * default_so2, so2);
    /// ```
    fn default() -> Self {
        Self::identity()
    }
}
