//! The 2D rotation group SO(2).

use core::ops::Mul;

use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;

/// A 2D rotation, stored as a unit complex number `(cosθ, sinθ)`. Composition is a complex product,
/// so it takes no trigonometry. The group is abelian; `exp`/`log` are exact and need no fallback.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SO2<T: Numeric> {
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
    pub fn act(self, p: Vector<2, T>) -> Vector<2, T> {
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
    pub fn hat(theta: T) -> Matrix<2, 2, T> {
        Matrix::new([[T::ZERO, -theta], [theta, T::ZERO]])
    }

    /// The inverse of [`SO2::hat`].
    #[inline]
    pub fn vee(m: Matrix<2, 2, T>) -> T {
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
    pub fn to_matrix(self) -> Matrix<2, 2, T> {
        Matrix::new([[self.c, -self.s], [self.s, self.c]])
    }

    /// Builds a rotation from a 2×2 matrix, normalizing its first column; `None` if that column is
    /// non-finite or degenerate.
    #[inline]
    pub fn try_from_matrix(m: Matrix<2, 2, T>) -> Option<Self> {
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
