//! Typed spatial velocity.

use core::ops::{Add, Neg, Sub};

use crate::linear_algebra::{Vector, Vector3D, Vector6D};
use crate::scalar::Numeric;

/// A spatial velocity (twist), stored linear-first in the crate-wide `[v; ω]` ordering.
///
/// The type owns its layout: the only value constructor takes the linear and angular parts by name,
/// so an `[ω; v]` mix-up is unrepresentable. Converters to and from a flat `[v; ω]` `Vector6D` are
/// the explicit seam to the group API (`SE3::exp` and friends). This is a plain element of a vector
/// space — `Add`/`Sub`/`Neg`/[`scale`](Twist::scale) act component-wise; the spatial *algebra*
/// (adjoint action, Lie bracket) is not defined here.
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
    /// let t = Twist::new(linear, angular);
    /// assert_eq!(t.linear(), Vector::new([1.0, 2.0, 3.0]));
    /// assert_eq!(t.angular(), Vector::new([4.0, 5.0, 6.0]));
    /// ```
    #[inline]
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
    /// let t = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(t.angular(), Vector::new([4.0, 5.0, 6.0]));
    /// ```
    #[inline]
    pub fn from_array(a: [T; 6]) -> Self {
        let [vx, vy, vz, wx, wy, wz] = a;
        Twist {
            linear: Vector::new([vx, vy, vz]),
            angular: Vector::new([wx, wy, wz]),
        }
    }

    /// The twist as a `[vx, vy, vz, ωx, ωy, ωz]` array.
    ///
    /// ```
    /// use multicalc::spatial::Twist;
    /// let t = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(t.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    #[inline]
    pub fn as_array(self) -> [T; 6] {
        let [vx, vy, vz] = *self.linear.as_array();
        let [wx, wy, wz] = *self.angular.as_array();
        [vx, vy, vz, wx, wy, wz]
    }

    /// A twist from a flat `[v; ω]` `Vector6D` (the group-API ordering).
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Twist;
    /// let t = Twist::from_vector(Vector::new([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// assert_eq!(t.linear(), Vector::new([1.0, 2.0, 3.0]));
    /// ```
    #[inline]
    pub fn from_vector(v: Vector6D<T>) -> Self {
        Self::from_array(v.into_array())
    }

    /// The twist as a flat `[v; ω]` `Vector6D` for the group API.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Twist;
    /// let t = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(t.to_vector(), Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
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
    /// let t = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(t.scale(2.0).as_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    /// ```
    #[inline]
    pub fn scale(self, scalar: T) -> Self {
        Twist {
            linear: self.linear.scale(scalar),
            angular: self.angular.scale(scalar),
        }
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
    fn from(v: Vector6D<T>) -> Self {
        Self::from_vector(v)
    }
}

impl<T: Numeric> From<Twist<T>> for Vector6D<T> {
    /// Flattens a twist into `[v; ω]`.
    #[inline]
    fn from(t: Twist<T>) -> Self {
        t.to_vector()
    }
}
