//! The 3D rigid-body transform group SE(3).

use core::ops::Mul;

use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::spatial::lie::{
    SO3, inverse_left_jacobian_se3, inverse_left_jacobian_so3, left_jacobian_se3,
    left_jacobian_so3, skew3,
};

/// A 3D rigid-body transform: a rotation and a translation. The tangent is `[vx, vy, vz, ωx, ωy, ωz]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SE3<T: Numeric> {
    rotation: SO3<T>,
    translation: Vector<3, T>,
}

impl<T: Numeric> SE3<T> {
    /// The identity transform.
    #[inline]
    pub fn identity() -> Self {
        SE3 {
            rotation: SO3::identity(),
            translation: Vector::zeros(),
        }
    }

    /// A transform from a rotation and translation.
    #[inline]
    pub fn from_parts(rotation: SO3<T>, translation: Vector<3, T>) -> Self {
        SE3 {
            rotation,
            translation,
        }
    }

    /// The rotation part.
    #[inline]
    pub fn rotation(self) -> SO3<T> {
        self.rotation
    }

    /// The translation part.
    #[inline]
    pub fn translation(self) -> Vector<3, T> {
        self.translation
    }

    /// Composition (also available as `*`).
    #[inline]
    pub fn compose(self, rhs: Self) -> Self {
        SE3 {
            rotation: self.rotation.compose(rhs.rotation),
            translation: self.rotation.act(rhs.translation) + self.translation,
        }
    }

    /// The inverse transform.
    #[inline]
    pub fn inverse(self) -> Self {
        let r_inv = self.rotation.inverse();
        SE3 {
            rotation: r_inv,
            translation: -r_inv.act(self.translation),
        }
    }

    /// Applies the transform to a 3D point.
    #[inline]
    pub fn act(self, p: Vector<3, T>) -> Vector<3, T> {
        self.rotation.act(p) + self.translation
    }

    /// The exponential map from a `[v; ω]` twist. Near θ = 0 the SO(3) left Jacobian uses a Taylor
    /// series, keeping the value and its derivative finite.
    #[inline]
    pub fn exp(xi: Vector<6, T>) -> Self {
        let v = Vector::new([xi[0], xi[1], xi[2]]);
        let phi = Vector::new([xi[3], xi[4], xi[5]]);
        SE3 {
            rotation: SO3::exp(phi),
            translation: left_jacobian_so3(phi) * v,
        }
    }

    /// The logarithm, the inverse of [`SE3::exp`], returning a `[v; ω]` twist.
    #[inline]
    pub fn log(self) -> Vector<6, T> {
        let phi = self.rotation.log();
        let v = inverse_left_jacobian_so3(phi) * self.translation;
        Vector::new([v[0], v[1], v[2], phi[0], phi[1], phi[2]])
    }

    /// The 6×6 adjoint `[[R, [t]×·R], [0, R]]` for the `[v; ω]` ordering.
    #[inline]
    pub fn adjoint(self) -> Matrix<6, 6, T> {
        let r = self.rotation.to_matrix();
        let tr = skew3(self.translation) * r;
        let mut ad = Matrix::zeros();
        for i in 0..3 {
            for j in 0..3 {
                ad[(i, j)] = r[(i, j)];
                ad[(i, j + 3)] = tr[(i, j)];
                ad[(i + 3, j + 3)] = r[(i, j)];
            }
        }
        ad
    }

    /// The 4×4 Lie-algebra element for a `[v; ω]` twist.
    #[inline]
    pub fn hat(xi: Vector<6, T>) -> Matrix<4, 4, T> {
        Matrix::new([
            [T::ZERO, -xi[5], xi[4], xi[0]],
            [xi[5], T::ZERO, -xi[3], xi[1]],
            [-xi[4], xi[3], T::ZERO, xi[2]],
            [T::ZERO, T::ZERO, T::ZERO, T::ZERO],
        ])
    }

    /// The inverse of [`SE3::hat`].
    #[inline]
    pub fn vee(m: Matrix<4, 4, T>) -> Vector<6, T> {
        Vector::new([
            m[(0, 3)],
            m[(1, 3)],
            m[(2, 3)],
            m[(2, 1)],
            m[(0, 2)],
            m[(1, 0)],
        ])
    }

    /// The 4×4 homogeneous transform matrix.
    #[inline]
    pub fn to_matrix(self) -> Matrix<4, 4, T> {
        let r = self.rotation.to_matrix();
        let t = self.translation;
        let mut m = Matrix::zeros();
        for i in 0..3 {
            for j in 0..3 {
                m[(i, j)] = r[(i, j)];
            }
            m[(i, 3)] = t[i];
        }
        m[(3, 3)] = T::ONE;
        m
    }

    /// Builds a transform from a 4×4 homogeneous matrix; `None` if the rotation block is degenerate.
    #[inline]
    pub fn try_from_matrix(m: Matrix<4, 4, T>) -> Option<Self> {
        let mut r = Matrix::zeros();
        for i in 0..3 {
            for j in 0..3 {
                r[(i, j)] = m[(i, j)];
            }
        }
        let rotation = SO3::try_from_matrix(r)?;
        Some(SE3 {
            rotation,
            translation: Vector::new([m[(0, 3)], m[(1, 3)], m[(2, 3)]]),
        })
    }

    /// Geodesic (screw-motion) interpolation; `t = 0` gives `self`, `t = 1` gives `other`.
    #[inline]
    pub fn interpolate(self, other: Self, t: T) -> Self {
        self.compose(Self::exp(self.inverse().compose(other).log() * t))
    }

    /// The SE(3) left Jacobian `J_l(ξ)` for the `[v; ω]` twist ordering.
    ///
    /// ```
    /// use multicalc::spatial::SE3;
    /// use multicalc::linear_algebra::Vector;
    /// let xi = Vector::new([0.1_f64, -0.2, 0.3, 0.2, -0.1, 0.4]);
    /// let prod = SE3::left_jacobian(xi) * SE3::left_jacobian_inverse(xi);
    /// for i in 0..6 { assert!((prod[(i, i)] - 1.0).abs() < 1e-10); }
    /// ```
    #[inline]
    pub fn left_jacobian(xi: Vector<6, T>) -> Matrix<6, 6, T> {
        left_jacobian_se3(xi)
    }

    /// The SE(3) right Jacobian `J_r(ξ) = J_l(−ξ)`.
    #[inline]
    pub fn right_jacobian(xi: Vector<6, T>) -> Matrix<6, 6, T> {
        left_jacobian_se3(-xi)
    }

    /// The inverse SE(3) left Jacobian `J_l⁻¹(ξ)`.
    #[inline]
    pub fn left_jacobian_inverse(xi: Vector<6, T>) -> Matrix<6, 6, T> {
        inverse_left_jacobian_se3(xi)
    }

    /// The inverse SE(3) right Jacobian `J_r⁻¹(ξ) = J_l⁻¹(−ξ)`.
    #[inline]
    pub fn right_jacobian_inverse(xi: Vector<6, T>) -> Matrix<6, 6, T> {
        inverse_left_jacobian_se3(-xi)
    }
}

impl<T: Numeric> Mul for SE3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.compose(rhs)
    }
}
