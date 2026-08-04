//! The 3D rotation group SO(3), wrapping a unit quaternion.

use core::ops::Mul;

use crate::linear_algebra::{Matrix3D, Vector, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::Quaternion;
use crate::spatial::lie::{inverse_left_jacobian_so3, left_jacobian_so3, skew3};

/// A 3D rotation. Wraps a unit [`Quaternion`] and carries the unit-rotation invariant. Composition
/// uses the Hamilton product; call [`SO3::normalized`] to remove drift after long chains. The
/// tangent is a rotation vector `φ = θ·n̂` in radians, and the retract is `R · exp(φ)`.
///
/// ```
/// use multicalc::spatial::SO3;
/// use multicalc::linear_algebra::Vector;
/// let quarter_turn_about_z = Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]);
/// let rotation = SO3::<f64>::exp(quarter_turn_about_z);
///
/// let point = Vector::new([1.0, 0.0, 0.0]);
/// let rotated = rotation.act(point);          // the x axis swings onto the y axis
/// assert!(rotated[0].abs() < 1e-12);
/// assert!((rotated[1] - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct SO3<T: Numeric = f64> {
    q: Quaternion<T>,
}

impl<T: Numeric> SO3<T> {
    /// The zero rotation.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        SO3 {
            q: Quaternion::identity(),
        }
    }

    /// From a quaternion, normalized to unit norm. Yields NaN components for a zero quaternion, as
    /// float division does; use [`SO3::try_from_quaternion`] for a checked version.
    #[inline]
    #[must_use]
    pub fn from_quaternion(q: Quaternion<T>) -> Self {
        SO3 { q: q.normalized() }
    }

    /// From a quaternion, or `None` if its norm is non-finite or underflows.
    #[inline]
    #[must_use]
    pub fn try_from_quaternion(q: Quaternion<T>) -> Option<Self> {
        q.try_normalized().map(|q| SO3 { q })
    }

    /// The underlying unit quaternion.
    #[inline]
    #[must_use]
    pub fn quaternion(self) -> Quaternion<T> {
        self.q
    }

    /// Composition (also available as `*`).
    #[inline]
    #[must_use]
    pub fn compose(self, rhs: Self) -> Self {
        SO3 { q: self.q * rhs.q }
    }

    /// The inverse rotation.
    #[inline]
    #[must_use]
    pub fn inverse(self) -> Self {
        SO3 {
            q: self.q.conjugate(),
        }
    }

    /// Rotates a 3D point.
    #[inline]
    pub fn act(self, p: Vector3D<T>) -> Vector3D<T> {
        self.q.transform_point(p)
    }

    /// The exponential map from a rotation vector `φ = θ·n̂`. Near θ = 0 the underlying quaternion
    /// uses a Taylor series, so the derivative stays finite at φ = 0.
    #[inline]
    #[must_use]
    pub fn exp(phi: Vector3D<T>) -> Self {
        SO3 {
            q: Quaternion::from_scaled_axis(phi),
        }
    }

    /// The logarithm, returning `φ` with `‖φ‖ ≤ π` (shortest path). Well-defined across θ = π.
    #[inline]
    pub fn log(self) -> Vector3D<T> {
        self.q.to_scaled_axis()
    }

    /// The Lie-algebra element `[φ]×` (skew-symmetric).
    #[inline]
    pub fn hat(phi: Vector3D<T>) -> Matrix3D<T> {
        skew3(phi)
    }

    /// The inverse of [`SO3::hat`].
    #[inline]
    pub fn vee(m: Matrix3D<T>) -> Vector3D<T> {
        let [[_, _, m02], [m10, _, _], [_, m21, _]] = m.into_array();
        Vector::new([m21, m02, m10])
    }

    /// The adjoint, equal to the rotation matrix (`Ad_R = R`).
    #[inline]
    pub fn adjoint(self) -> Matrix3D<T> {
        self.q.to_rotation_matrix()
    }

    /// The 3×3 rotation matrix.
    #[inline]
    pub fn to_matrix(self) -> Matrix3D<T> {
        self.q.to_rotation_matrix()
    }

    /// Builds a rotation from a finite 3×3 matrix sufficiently close to a proper unit rotation;
    /// `None` otherwise.
    #[inline]
    #[must_use]
    pub fn try_from_matrix(m: Matrix3D<T>) -> Option<Self> {
        let gram = m.transpose() * m;
        for row in 0..3 {
            for column in 0..3 {
                let target = if row == column { T::ONE } else { T::ZERO };
                let error = (gram[(row, column)] - target).abs();
                if !error.is_finite() || error > T::EPSILON_X30 {
                    return None;
                }
            }
        }

        let determinant = m.determinant();
        if !determinant.is_finite() || determinant <= T::ZERO {
            return None;
        }

        Quaternion::try_from_rotation_matrix(m).map(|q| SO3 { q })
    }

    /// The orientation of a body from two directions it can see, given where those directions
    /// point in the world.
    ///
    /// Standing still, a drone's push sensor tells it which way is down and its compass tells it
    /// roughly which way is north — two directions measured in the body's own frame, whose world
    /// directions are already known. Two directions that are not parallel pin the orientation down
    /// completely.
    ///
    /// The primary pair is trusted exactly: the returned orientation carries `primary_observed`
    /// onto `primary_reference` with no error left over. The secondary pair only settles the
    /// remaining spin about that first direction, so the noisier of the two readings belongs
    /// there. Lengths do not matter; only the directions do.
    ///
    /// Returns `None` when a direction has zero or non-finite length, or when the two observed
    /// directions point the same way as each other (or the two reference directions do) — parallel
    /// directions leave the spin unsettled.
    ///
    /// ```
    /// use multicalc::spatial::SO3;
    /// use multicalc::linear_algebra::Vector;
    /// // Level and facing north: down reads as down, north reads as north.
    /// let down_in_world = Vector::new([0.0_f64, 0.0, -1.0]);
    /// let north_in_world = Vector::new([1.0, 0.0, 0.0]);
    /// // The body is turned a quarter turn about z, so it sees north off to its right.
    /// let down_in_body = Vector::new([0.0, 0.0, -1.0]);
    /// let north_in_body = Vector::new([0.0, -1.0, 0.0]);
    /// let orientation = SO3::from_two_direction_pairs(
    ///     down_in_body, north_in_body, down_in_world, north_in_world,
    /// )
    /// .unwrap();
    /// let recovered_north = orientation.act(north_in_body);
    /// assert!((recovered_north[0] - north_in_world[0]).abs() < 1e-12);
    /// assert!((recovered_north[1] - north_in_world[1]).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn from_two_direction_pairs(
        primary_observed: Vector3D<T>,
        secondary_observed: Vector3D<T>,
        primary_reference: Vector3D<T>,
        secondary_reference: Vector3D<T>,
    ) -> Option<Self> {
        if !primary_observed.is_finite()
            || !secondary_observed.is_finite()
            || !primary_reference.is_finite()
            || !secondary_reference.is_finite()
        {
            return None;
        }

        let observed_first = primary_observed.try_normalized()?;
        let observed_third = observed_first.cross(secondary_observed).try_normalized()?;
        let observed_second = observed_third.cross(observed_first);

        let reference_first = primary_reference.try_normalized()?;
        let reference_third = reference_first
            .cross(secondary_reference)
            .try_normalized()?;
        let reference_second = reference_third.cross(reference_first);

        // Each set of axes goes in as columns, so the product carries body directions to world
        // ones.
        let observed_axes = Matrix3D::from_fn(|row, column| match column {
            0 => observed_first[row],
            1 => observed_second[row],
            _ => observed_third[row],
        });
        let reference_axes = Matrix3D::from_fn(|row, column| match column {
            0 => reference_first[row],
            1 => reference_second[row],
            _ => reference_third[row],
        });
        SO3::try_from_matrix(reference_axes * observed_axes.transpose())
    }

    /// Geodesic interpolation (slerp); `t = 0` gives `self`, `t = 1` gives `other`.
    #[inline]
    #[must_use]
    pub fn interpolate(self, other: Self, t: T) -> Self {
        SO3 {
            q: self.q.slerp(other.q, t),
        }
    }

    /// This rotation renormalized, removing drift accumulated over long composition chains.
    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        SO3 {
            q: self.q.normalized(),
        }
    }

    /// The SO(3) left Jacobian `J_l(φ)`, relating a tangent perturbation to the resulting rotation.
    #[inline]
    pub fn left_jacobian(phi: Vector3D<T>) -> Matrix3D<T> {
        left_jacobian_so3(phi)
    }

    /// The SO(3) right Jacobian `J_r(φ) = J_l(−φ)`.
    #[inline]
    pub fn right_jacobian(phi: Vector3D<T>) -> Matrix3D<T> {
        left_jacobian_so3(-phi)
    }

    /// The inverse SO(3) left Jacobian `J_l⁻¹(φ)`.
    ///
    /// ```
    /// use multicalc::spatial::SO3;
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let phi = Vector::new([0.2_f64, -0.1, 0.4]);
    /// let prod = (SO3::left_jacobian(phi) * SO3::left_jacobian_inverse(phi));
    /// for i in 0..3 { assert!((prod[(i, i)] - 1.0).abs() < 1e-12); }
    /// ```
    #[inline]
    pub fn left_jacobian_inverse(phi: Vector3D<T>) -> Matrix3D<T> {
        inverse_left_jacobian_so3(phi)
    }

    /// The inverse SO(3) right Jacobian `J_r⁻¹(φ) = J_l⁻¹(−φ)`.
    #[inline]
    pub fn right_jacobian_inverse(phi: Vector3D<T>) -> Matrix3D<T> {
        inverse_left_jacobian_so3(-phi)
    }
}

impl<T: Numeric> Mul for SO3<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.compose(rhs)
    }
}

impl<T: Numeric> Default for SO3<T> {
    /// Returns the identity as default.
    ///
    /// ```
    /// use multicalc::SO3;
    ///
    /// let default_so3 = SO3::default();
    /// let so3 = SO3::<f64>::from_quaternion(multicalc::Quaternion::from_array([1.0, 2.0, 3.0, 4.0]));
    ///
    /// assert_eq!(so3 * default_so3, so3);
    /// ```
    fn default() -> Self {
        Self::identity()
    }
}
