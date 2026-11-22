//! The 3D rigid-body transform group SE(3).

use core::ops::Mul;

use crate::linear_algebra::{Matrix, Matrix4D, Matrix6D, Vector, Vector3D, Vector6D};
use crate::scalar::Numeric;
use crate::spatial::lie::{
    SO3, inverse_left_jacobian_se3, inverse_left_jacobian_so3, left_jacobian_se3,
    left_jacobian_so3, skew3,
};
use crate::spatial::{SpatialInertia, Twist, Wrench};

/// A 3D rigid-body transform: a rotation and a translation. The tangent is `[vx, vy, vz, ωx, ωy, ωz]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SE3<T: Numeric = f64> {
    rotation: SO3<T>,
    translation: Vector3D<T>,
}

impl<T: Numeric> SE3<T> {
    /// The identity transform.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        SE3 {
            rotation: SO3::identity(),
            translation: Vector::zeros(),
        }
    }

    /// A transform from a rotation and translation.
    #[inline]
    #[must_use]
    pub fn from_parts(rotation: SO3<T>, translation: Vector3D<T>) -> Self {
        SE3 {
            rotation,
            translation,
        }
    }

    /// The rotation part.
    #[inline]
    #[must_use]
    pub fn rotation(self) -> SO3<T> {
        self.rotation
    }

    /// The translation part.
    #[inline]
    pub fn translation(self) -> Vector3D<T> {
        self.translation
    }

    /// Composition (also available as `*`).
    #[inline]
    #[must_use]
    pub fn compose(self, rhs: Self) -> Self {
        SE3 {
            rotation: self.rotation.compose(rhs.rotation),
            translation: self.rotation.act(rhs.translation) + self.translation,
        }
    }

    /// The inverse transform.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        let r_inv = self.rotation.inverse();
        SE3 {
            rotation: r_inv,
            translation: -r_inv.act(self.translation),
        }
    }

    /// Applies the transform to a 3D point.
    #[inline]
    pub fn act(self, point: Vector3D<T>) -> Vector3D<T> {
        self.rotation.act(point) + self.translation
    }

    /// The exponential map from a `[v; ω]` twist. Near θ = 0 the SO(3) left Jacobian uses a Taylor
    /// series, keeping the value and its derivative finite.
    #[inline]
    #[must_use]
    pub fn exp(twist: Vector6D<T>) -> Self {
        let [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z] = *twist.as_array();
        let linear = Vector::new([lin_x, lin_y, lin_z]);
        let phi = Vector::new([ang_x, ang_y, ang_z]);
        SE3 {
            rotation: SO3::exp(phi),
            translation: left_jacobian_so3(phi) * linear,
        }
    }

    /// The logarithm, the inverse of [`SE3::exp`], returning a `[v; ω]` twist.
    #[inline]
    pub fn log(self) -> Vector6D<T> {
        let phi = self.rotation.log();
        let linear = inverse_left_jacobian_so3(phi) * self.translation;
        let [lin_x, lin_y, lin_z] = *linear.as_array();
        let [ang_x, ang_y, ang_z] = *phi.as_array();
        Vector::new([lin_x, lin_y, lin_z, ang_x, ang_y, ang_z])
    }

    /// The 6×6 adjoint `[[R, [t]×·R], [0, R]]` for the `[v; ω]` ordering.
    #[inline]
    pub fn adjoint(self) -> Matrix6D<T> {
        let rotation = self.rotation.to_matrix();
        let skew_t_r = skew3(self.translation) * rotation;
        let mut adjoint = Matrix::zeros();
        for i in 0..3 {
            for j in 0..3 {
                let rot_entry = rotation[(i, j)];
                let skew_entry = skew_t_r[(i, j)];
                adjoint[(i, j)] = rot_entry;
                adjoint[(i, j + 3)] = skew_entry;
                adjoint[(i + 3, j + 3)] = rot_entry;
            }
        }
        adjoint
    }

    /// The 6×6 force adjoint `Ad⁻ᵀ = [[R, 0], [[t]×·R, R]]` for the `[f; τ]` ordering.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::SE3;
    /// let pose = SE3::exp(Vector::new([0.4_f64, -0.2, 0.7, 0.3, 0.9, -0.5]));
    /// // Adᵀ · Ad⁻ᵀ = I₆.
    /// let product = pose.adjoint().transpose() * pose.force_adjoint();
    /// for index in 0..6 {
    ///     assert!((product[(index, index)] - 1.0).abs() < 1e-12);
    /// }
    /// ```
    #[inline]
    pub fn force_adjoint(self) -> Matrix6D<T> {
        let rotation = self.rotation.to_matrix();
        let skew_t_r = skew3(self.translation) * rotation;
        let mut adjoint = Matrix::zeros();
        for i in 0..3 {
            for j in 0..3 {
                let rot_entry = rotation[(i, j)];
                let skew_entry = skew_t_r[(i, j)];
                adjoint[(i, j)] = rot_entry;
                adjoint[(i + 3, j)] = skew_entry;
                adjoint[(i + 3, j + 3)] = rot_entry;
            }
        }
        adjoint
    }

    /// Carries a twist through this transform: `ω' = R·ω`, `v' = R·v + t×(R·ω)`.
    ///
    /// Block-wise; equals [`adjoint`](SE3::adjoint) applied to the flat `[v; ω]` vector.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3, Twist};
    /// // Quarter turn about z, offset 1 m along x.
    /// let pose = SE3::from_parts(
    ///     SO3::exp(Vector::new([0.0_f64, 0.0, core::f64::consts::FRAC_PI_2])),
    ///     Vector::new([1.0, 0.0, 0.0]),
    /// );
    /// let spin = Twist::new(Vector::zeros(), Vector::new([0.0_f64, 0.0, 1.0]));
    /// let moved = pose.act_twist(spin);
    /// // t × ω picks up a linear part.
    /// assert!((moved.linear() - Vector::new([0.0, -1.0, 0.0])).norm() < 1e-12);
    /// assert!((moved.angular() - Vector::new([0.0, 0.0, 1.0])).norm() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn act_twist(self, twist: Twist<T>) -> Twist<T> {
        let angular = self.rotation.act(twist.angular());
        let linear = self.rotation.act(twist.linear()) + self.translation.cross(angular);
        Twist::new(linear, angular)
    }

    /// Carries a wrench through this transform: `f' = R·f`, `τ' = R·τ + t×(R·f)`.
    ///
    /// Block-wise; equals [`force_adjoint`](SE3::force_adjoint) applied to the flat `[f; τ]` vector.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, Wrench};
    /// let pose = SE3::exp(Vector::new([0.4_f64, -0.2, 0.7, 0.3, 0.9, -0.5]));
    /// let push = Wrench::from_array([2.0_f64, -1.5, 0.7, 0.9, -0.2, 1.3]);
    /// let matrix_form = pose.force_adjoint() * push.to_vector();
    /// assert!((pose.act_wrench(push).to_vector() - matrix_form).norm() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn act_wrench(self, wrench: Wrench<T>) -> Wrench<T> {
        let force = self.rotation.act(wrench.force());
        let torque = self.rotation.act(wrench.torque()) + self.translation.cross(force);
        Wrench::new(force, torque)
    }

    /// The inverse twist action: `ω' = Rᵀ·ω`, `v' = Rᵀ·(v − t×ω)`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, Twist};
    /// let pose = SE3::exp(Vector::new([0.4_f64, -0.2, 0.7, 0.3, 0.9, -0.5]));
    /// let motion = Twist::from_array([0.3_f64, -0.7, 1.1, 0.5, 0.2, -0.9]);
    /// let round_trip = pose.inverse_act_twist(pose.act_twist(motion));
    /// assert!((round_trip - motion).to_vector().norm() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_act_twist(self, twist: Twist<T>) -> Twist<T> {
        let inverse_rotation = self.rotation.inverse();
        let angular = inverse_rotation.act(twist.angular());
        let linear = inverse_rotation.act(twist.linear() - self.translation.cross(twist.angular()));
        Twist::new(linear, angular)
    }

    /// The inverse wrench action: `f' = Rᵀ·f`, `τ' = Rᵀ·(τ − t×f)`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, Wrench};
    /// let pose = SE3::exp(Vector::new([0.4_f64, -0.2, 0.7, 0.3, 0.9, -0.5]));
    /// let push = Wrench::from_array([2.0_f64, -1.5, 0.7, 0.9, -0.2, 1.3]);
    /// let round_trip = pose.inverse_act_wrench(pose.act_wrench(push));
    /// assert!((round_trip - push).to_vector().norm() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_act_wrench(self, wrench: Wrench<T>) -> Wrench<T> {
        let inverse_rotation = self.rotation.inverse();
        let force = inverse_rotation.act(wrench.force());
        let torque = inverse_rotation.act(wrench.torque() - self.translation.cross(wrench.force()));
        Wrench::new(force, torque)
    }

    /// Carries a spatial inertia through this transform: `m` unchanged, `c' = R·c + t`,
    /// `I_c' = R·I_c·Rᵀ`.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::{SE3, SO3, SpatialInertia};
    /// let body = SpatialInertia::new(
    ///     2.0_f64,
    ///     Vector::new([0.0, 0.0, 0.0]),
    ///     Matrix::from_diagonal([1.0, 1.0, 1.0]),
    /// )
    /// .unwrap();
    /// // Pure translation along x.
    /// let pose = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
    /// let moved = pose.act_inertia(body);
    /// assert_eq!(moved.mass(), 2.0);
    /// assert!((moved.center_of_mass() - Vector::new([1.0, 0.0, 0.0])).norm() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn act_inertia(self, inertia: SpatialInertia<T>) -> SpatialInertia<T> {
        let rotation = self.rotation.to_matrix();
        SpatialInertia::from_parts(
            inertia.mass(),
            self.act(inertia.center_of_mass()),
            rotation * inertia.rotational_inertia() * rotation.transpose(),
        )
    }

    /// The inverse inertia action, via [`inverse`](SE3::inverse).
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::spatial::{SE3, SpatialInertia};
    /// let body = SpatialInertia::new(
    ///     2.0_f64,
    ///     Vector::new([0.1, -0.2, 0.3]),
    ///     Matrix::from_diagonal([1.0, 2.0, 3.0]),
    /// )
    /// .unwrap();
    /// let pose = SE3::exp(Vector::new([0.4_f64, -0.2, 0.7, 0.3, 0.9, -0.5]));
    /// let round_trip = pose.inverse_act_inertia(pose.act_inertia(body));
    /// assert!((round_trip.center_of_mass() - body.center_of_mass()).norm() < 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverse_act_inertia(self, inertia: SpatialInertia<T>) -> SpatialInertia<T> {
        self.inverse().act_inertia(inertia)
    }

    /// The 4×4 Lie-algebra element for a `[v; ω]` twist.
    #[inline]
    pub fn hat(twist: Vector6D<T>) -> Matrix4D<T> {
        let [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z] = *twist.as_array();
        Matrix::new([
            [T::ZERO, -ang_z, ang_y, lin_x],
            [ang_z, T::ZERO, -ang_x, lin_y],
            [-ang_y, ang_x, T::ZERO, lin_z],
            [T::ZERO, T::ZERO, T::ZERO, T::ZERO],
        ])
    }

    /// The inverse of [`SE3::hat`].
    #[inline]
    pub fn vee(matrix: Matrix4D<T>) -> Vector6D<T> {
        let [[_, _, m02, m03], [m10, _, _, m13], [_, m21, _, m23], _] = matrix.into_array();
        Vector::new([m03, m13, m23, m21, m02, m10])
    }

    /// The 4×4 homogeneous transform matrix.
    #[inline]
    pub fn to_matrix(self) -> Matrix4D<T> {
        let rotation = self.rotation.to_matrix();
        let translation = self.translation;
        let mut matrix = Matrix::zeros();
        for i in 0..3 {
            for j in 0..3 {
                matrix[(i, j)] = rotation[(i, j)];
            }
            matrix[(i, 3)] = translation[i];
        }
        matrix[(3, 3)] = T::ONE;
        matrix
    }

    /// Builds a transform from a finite 4×4 homogeneous matrix; `None` if the rotation block is not
    /// a proper unit rotation or the bottom row differs from `[0, 0, 0, 1]` by more than scalar
    /// round-off.
    #[inline]
    #[must_use]
    pub fn try_from_matrix(matrix: Matrix4D<T>) -> Option<Self> {
        let [
            [m00, m01, m02, m03],
            [m10, m11, m12, m13],
            [m20, m21, m22, m23],
            [m30, m31, m32, m33],
        ] = matrix.into_array();
        if !m03.is_finite()
            || !m13.is_finite()
            || !m23.is_finite()
            || !m30.is_finite()
            || !m31.is_finite()
            || !m32.is_finite()
            || !m33.is_finite()
            || m30.abs() > T::EPSILON_X30
            || m31.abs() > T::EPSILON_X30
            || m32.abs() > T::EPSILON_X30
            || (m33 - T::ONE).abs() > T::EPSILON_X30
        {
            return None;
        }

        let rotation_block = Matrix::new([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]]);
        let rotation = SO3::try_from_matrix(rotation_block)?;
        Some(SE3 {
            rotation,
            translation: Vector::new([m03, m13, m23]),
        })
    }

    /// Geodesic (screw-motion) interpolation; `t = 0` gives `self`, `t = 1` gives `other`.
    #[inline]
    #[must_use]
    pub fn interpolate(self, other: Self, amount: T) -> Self {
        self.compose(Self::exp(self.inverse().compose(other).log() * amount))
    }

    /// The SE(3) left Jacobian `J_l(ξ)` for the `[v; ω]` twist ordering.
    ///
    /// ```
    /// use multicalc::spatial::SE3;
    /// use multicalc::linear_algebra::Vector;
    /// let twist = Vector::new([0.1_f64, -0.2, 0.3, 0.2, -0.1, 0.4]);
    /// let prod = (SE3::left_jacobian(twist) * SE3::left_jacobian_inverse(twist));
    /// for i in 0..6 { assert!((prod[(i, i)] - 1.0).abs() < 1e-10); }
    /// ```
    #[inline]
    pub fn left_jacobian(twist: Vector6D<T>) -> Matrix6D<T> {
        left_jacobian_se3(twist)
    }

    /// The SE(3) right Jacobian `J_r(ξ) = J_l(−ξ)`.
    #[inline]
    pub fn right_jacobian(twist: Vector6D<T>) -> Matrix6D<T> {
        left_jacobian_se3(-twist)
    }

    /// The inverse SE(3) left Jacobian `J_l⁻¹(ξ)`.
    #[inline]
    pub fn left_jacobian_inverse(twist: Vector6D<T>) -> Matrix6D<T> {
        inverse_left_jacobian_se3(twist)
    }

    /// The inverse SE(3) right Jacobian `J_r⁻¹(ξ) = J_l⁻¹(−ξ)`.
    #[inline]
    pub fn right_jacobian_inverse(twist: Vector6D<T>) -> Matrix6D<T> {
        inverse_left_jacobian_se3(-twist)
    }
}

impl<T: Numeric> Mul for SE3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.compose(rhs)
    }
}

impl<T: Numeric> Default for SE3<T> {
    /// Returns the identity as default.
    ///
    /// ```
    /// use multicalc::{SE3, SO3, Vector3D};
    ///
    /// let default_se3 = SE3::default();
    /// let se3 = SE3::<f64>::from_parts(
    ///     SO3::from_quaternion(multicalc::Quaternion::from_array([1.0, 2.0, 3.0, 4.0])),
    ///     Vector3D::new([1.0, 2.0, 3.0])
    /// );
    ///
    /// assert_eq!(se3 * default_se3, se3);
    /// ```
    fn default() -> Self {
        Self::identity()
    }
}
