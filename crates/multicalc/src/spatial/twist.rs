//! Typed spatial velocity.

use core::ops::{Add, Neg, Sub};

use crate::linear_algebra::{Vector, Vector3D, Vector6D};
use crate::scalar::Numeric;
use crate::spatial::Wrench;

/// A spatial velocity (twist), stored linear-first in the crate-wide `[v; ω]` ordering.
///
/// The type owns its layout: the only value constructor takes the linear and angular parts by name,
/// so an `[ω; v]` mix-up is unrepresentable. Converters to and from a flat `[v; ω]` `Vector6D` are
/// the explicit seam to the group API (`SE3::exp` and friends). This is a plain element of a vector
/// space. `Add`/`Sub`/`Neg`/[`scale`](Twist::scale) act component-wise. The spatial algebra is
/// [`cross`](Twist::cross) (motion ×), [`cross_wrench`](Twist::cross_wrench) (force ×\*),
/// [`dot_wrench`](Twist::dot_wrench) (power), and
/// [`SE3::act_twist`](crate::spatial::SE3::act_twist) for a change of frame.
///
/// ```
/// use multicalc::linear_algebra::Vector6D;
/// use multicalc::spatial::Twist;
/// let a = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let b = Twist::from_array([1.0_f64; 6]);
/// assert_eq!((a + b).as_array(), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
/// assert_eq!((a - b).as_array(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert_eq!((-a).as_array(), [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
/// assert_eq!(a.scale(2.0).as_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
/// let _: Vector6D = a.into();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Twist<T: Numeric = f64> {
    linear: Vector3D<T>,
    angular: Vector3D<T>,
}

impl<T: Numeric> Twist<T> {
    /// A twist from its linear and angular parts.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Twist;
    /// let linear = Vector::new([1.0_f64, 2.0, 3.0]);
    /// let angular = Vector::new([4.0, 5.0, 6.0]);
    /// let twist = Twist::new(linear, angular);
    /// assert_eq!(twist.linear(), Vector::new([1.0, 2.0, 3.0]));
    /// assert_eq!(twist.angular(), Vector::new([4.0, 5.0, 6.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn new(linear: Vector3D<T>, angular: Vector3D<T>) -> Self {
        Twist { linear, angular }
    }

    /// The zero twist.
    ///
    /// ```
    /// use multicalc::spatial::Twist;
    /// assert_eq!(Twist::<f64>::zeros().as_array(), [0.0; 6]);
    /// ```
    #[inline]
    #[must_use]
    pub fn zeros() -> Self {
        Twist {
            linear: Vector::zeros(),
            angular: Vector::zeros(),
        }
    }

    /// A twist from a `[vx, vy, vz, ωx, ωy, ωz]` array.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Twist;
    /// let twist = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(twist.angular(), Vector::new([4.0, 5.0, 6.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_array(a: [T; 6]) -> Self {
        let [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z] = a;
        Twist {
            linear: Vector::new([lin_x, lin_y, lin_z]),
            angular: Vector::new([ang_x, ang_y, ang_z]),
        }
    }

    /// The twist as a `[vx, vy, vz, ωx, ωy, ωz]` array.
    ///
    /// ```
    /// use multicalc::spatial::Twist;
    /// let twist = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(twist.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_array(self) -> [T; 6] {
        let [lin_x, lin_y, lin_z] = *self.linear.as_array();
        let [ang_x, ang_y, ang_z] = *self.angular.as_array();
        [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z]
    }

    /// A twist from a flat `[v; ω]` `Vector6D` (the group-API ordering).
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Twist;
    /// let twist = Twist::from_vector(Vector::new([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// assert_eq!(twist.linear(), Vector::new([1.0, 2.0, 3.0]));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_vector(vector: Vector6D<T>) -> Self {
        Self::from_array(vector.into_array())
    }

    /// The twist as a flat `[v; ω]` `Vector6D` for the group API.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Twist;
    /// let twist = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(twist.to_vector(), Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// ```
    #[inline]
    pub fn to_vector(self) -> Vector6D<T> {
        Vector::new(self.as_array())
    }

    /// The linear (translational) part `v`.
    #[inline]
    pub fn linear(self) -> Vector3D<T> {
        self.linear
    }

    /// The angular (rotational) part `ω`.
    #[inline]
    pub fn angular(self) -> Vector3D<T> {
        self.angular
    }

    /// Multiplies both parts by `scalar`.
    ///
    /// ```
    /// use multicalc::spatial::Twist;
    /// let twist = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(twist.scale(2.0).as_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn scale(self, scalar: T) -> Self {
        Twist {
            linear: self.linear.scale(scalar),
            angular: self.angular.scale(scalar),
        }
    }

    /// The motion cross product `self × other`: `[ω×v' + v×ω' ; ω×ω']`. Antisymmetric.
    ///
    /// ```
    /// use multicalc::spatial::Twist;
    /// let turn = Twist::from_array([0.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0]);
    /// let slide = Twist::from_array([1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
    /// // ω_z carries v_x onto y.
    /// assert_eq!(turn.cross(slide).as_array(), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    /// assert_eq!(turn.cross(turn).as_array(), [0.0; 6]);
    /// ```
    #[inline]
    #[must_use]
    pub fn cross(self, other: Twist<T>) -> Twist<T> {
        Twist {
            linear: self.angular.cross(other.linear) + self.linear.cross(other.angular),
            angular: self.angular.cross(other.angular),
        }
    }

    /// The force cross product `self ×* wrench`: `[ω×f ; ω×τ + v×f]`.
    ///
    /// As a 6×6 it is the negative transpose of [`cross`](Twist::cross).
    ///
    /// ```
    /// use multicalc::spatial::{Twist, Wrench};
    /// let turn = Twist::from_array([0.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0]);
    /// let push = Wrench::from_array([1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0]);
    /// // ω_z carries f_x onto y; no moment, since v and τ are zero.
    /// assert_eq!(turn.cross_wrench(push).as_array(), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn cross_wrench(self, wrench: Wrench<T>) -> Wrench<T> {
        Wrench::new(
            self.angular.cross(wrench.force()),
            self.angular.cross(wrench.torque()) + self.linear.cross(wrench.force()),
        )
    }

    /// The power product `v·f + ω·τ` against a [`Wrench`].
    ///
    /// ```
    /// use multicalc::spatial::{Twist, Wrench};
    /// let motion = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let wrench = Wrench::from_array([1.0_f64; 6]);
    /// assert_eq!(motion.dot_wrench(wrench), 21.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn dot_wrench(self, wrench: Wrench<T>) -> T {
        self.linear.dot(wrench.force()) + self.angular.dot(wrench.torque())
    }
}

impl<T: Numeric> Add for Twist<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Twist {
            linear: self.linear + rhs.linear,
            angular: self.angular + rhs.angular,
        }
    }
}

impl<T: Numeric> Sub for Twist<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Twist {
            linear: self.linear - rhs.linear,
            angular: self.angular - rhs.angular,
        }
    }
}

impl<T: Numeric> Neg for Twist<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Twist {
            linear: -self.linear,
            angular: -self.angular,
        }
    }
}

impl<T: Numeric> From<Vector6D<T>> for Twist<T> {
    /// Reinterprets a flat `[v; ω]` `Vector6D` as a twist.
    #[inline]
    fn from(vector: Vector6D<T>) -> Self {
        Self::from_vector(vector)
    }
}

impl<T: Numeric> From<Twist<T>> for Vector6D<T> {
    /// Flattens a twist into `[v; ω]`.
    #[inline]
    fn from(twist: Twist<T>) -> Self {
        twist.to_vector()
    }
}

impl<T: Numeric> Default for Twist<T> {
    /// Returns the zero twist as default.
    ///
    /// ```
    /// use multicalc::Twist;
    ///
    /// let default_twist = Twist::default();
    /// let twist = Twist::<f64>::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    ///
    /// assert_eq!(twist + default_twist, twist);
    /// ```
    fn default() -> Self {
        Self::zeros()
    }
}
