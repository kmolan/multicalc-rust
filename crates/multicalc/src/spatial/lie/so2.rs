//! The 2D rotation group SO(2).

use core::ops::Mul;

use crate::linear_algebra::{Matrix, Matrix2D, Vector, Vector2D};
use crate::scalar::Numeric;

/// A 2D rotation, stored as a unit complex number `(cosθ, sinθ)`. Composition is a complex product,
/// so it takes no trigonometry. The group is abelian; `exp`/`log` are exact and need no fallback.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SO2<T: Numeric = f64> {
    c: T,
    s: T,
}

impl<T: Numeric> SO2<T> {
    /// The zero rotation.
    #[inline]
    pub fn identity() -> Self {
        SO2 {
            c: T::ONE,
            s: T::ZERO,
        }
    }

    /// The rotation by `theta` radians.
    #[inline]
    pub fn from_angle(theta: T) -> Self {
        SO2 {
            c: theta.cos(),
            s: theta.sin(),
        }
    }

    /// The `(cos, sin)` components.
    #[inline]
    pub fn cos_sin(self) -> (T, T) {
        (self.c, self.s)
    }

    /// Composition (also available as `*`).
    #[inline]
    pub fn compose(self, rhs: Self) -> Self {
        SO2 {
            c: self.c * rhs.c - self.s * rhs.s,
            s: self.c * rhs.s + self.s * rhs.c,
        }
    }

    /// The inverse rotation.
    #[inline]
    pub fn inverse(self) -> Self {
        SO2 {
            c: self.c,
            s: -self.s,
        }
    }

    /// Rotates a 2D point.
    #[inline]
    pub fn act(self, p: Vector2D<T>) -> Vector2D<T> {
        let [px, py] = *p.as_array();
        Vector::new([self.c * px - self.s * py, self.s * px + self.c * py])
    }

    /// The exponential map from the tangent angle.
    #[inline]
    pub fn exp(theta: T) -> Self {
        Self::from_angle(theta)
    }

    /// The logarithm, the tangent angle in `(−π, π]`.
    #[inline]
    pub fn log(self) -> T {
        self.s.atan2(self.c)
    }

    /// The Lie-algebra element `[[0, −θ], [θ, 0]]`.
    #[inline]
    pub fn hat(theta: T) -> Matrix2D<T> {
        Matrix::new([[T::ZERO, -theta], [theta, T::ZERO]])
    }

    /// The inverse of [`SO2::hat`].
    #[inline]
    pub fn vee(m: Matrix2D<T>) -> T {
        let [[_, _], [m10, _]] = m.into_array();
        m10
    }

    /// The adjoint, which is `1` (SO(2) is abelian).
    #[inline]
    pub fn adjoint(self) -> T {
        T::ONE
    }

    /// The 2×2 rotation matrix.
    #[inline]
    pub fn to_matrix(self) -> Matrix2D<T> {
        Matrix::new([[self.c, -self.s], [self.s, self.c]])
    }

    /// Builds a rotation from a 2×2 matrix, normalizing its first column; `None` if that column is
    /// non-finite or degenerate.
    #[inline]
    pub fn try_from_matrix(m: Matrix2D<T>) -> Option<Self> {
        let [[c, _], [s, _]] = m.into_array();
        let n = (c * c + s * s).sqrt();
        if !n.is_finite() || n <= T::EPSILON {
            None
        } else {
            Some(SO2 { c: c / n, s: s / n })
        }
    }

    /// Geodesic interpolation; `t = 0` gives `self`, `t = 1` gives `other`.
    #[inline]
    pub fn interpolate(self, other: Self, t: T) -> Self {
        self.compose(Self::exp(self.inverse().compose(other).log() * t))
    }

    /// The squared norm `c² + s²`.
    #[inline]
    fn norm_squared(self) -> T {
        self.c * self.c + self.s * self.s
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
    /// let (c, s) = normalized.cos_sin();
    ///
    /// assert!((normalized.norm() - 1.0).abs() <= <f64 as multicalc::Numeric>::EPSILON);
    /// ```
    #[inline]
    pub fn normalized(self) -> Self {
        let scale = self.norm().recip();
        SO2 {
            c: self.c * scale,
            s: self.s * scale,
        }
    }

    /// The SO(2) left Jacobian, which is `1` (SO(2) is abelian).
    #[inline]
    pub fn left_jacobian(_theta: T) -> T {
        T::ONE
    }

    /// The SO(2) right Jacobian, which is `1` (SO(2) is abelian).
    #[inline]
    pub fn right_jacobian(_theta: T) -> T {
        T::ONE
    }

    /// The inverse SO(2) left Jacobian, which is `1` (SO(2) is abelian).
    #[inline]
    pub fn left_jacobian_inverse(_theta: T) -> T {
        T::ONE
    }

    /// The inverse SO(2) right Jacobian, which is `1` (SO(2) is abelian).
    #[inline]
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
