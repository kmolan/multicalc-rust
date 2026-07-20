//! Typed spatial velocity.

use core::ops::{Add, Mul, Neg, Sub};
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// A spatial velocity (twist), stored linear-first in the crate-wide `[v; ω]` ordering.
///
/// The type owns its layout: the only value constructor takes the linear and angular parts by name,
/// so an `[ω; v]` mix-up is unrepresentable. Converters to and from a flat `[v; ω]` `Vector<6>` are
/// the explicit seam to the group API (`SE3::exp` and friends). This is a plain element of a vector
/// space — `Add`/`Sub`/`Neg`/[`scale`](Twist::scale) act component-wise; the spatial *algebra*
/// (adjoint action, Lie bracket) is not defined here.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// use multicalc::spatial::Twist;
/// let a = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let b = Twist::from_array([1.0_f64; 6]);
/// assert_eq!((a + b).as_array(), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
/// assert_eq!((a - b).as_array(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert_eq!((-a).as_array(), [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
/// assert_eq!(a.scale(2.0).as_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
/// let _: Vector<6, f64> = a.into();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Twist<T: Numeric> {
    linear: Vector<3, T>,
    angular: Vector<3, T>,
}

impl<T: Numeric> Twist<T> {
    /// A twist from its linear and angular parts.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Twist;
    /// let t = Twist::new(Vector::new([1.0_f64, 2.0, 3.0]), Vector::new([4.0, 5.0, 6.0]));
    /// assert_eq!(t.linear(), Vector::new([1.0, 2.0, 3.0]));
    /// assert_eq!(t.angular(), Vector::new([4.0, 5.0, 6.0]));
    /// ```
    #[inline]
    pub fn new(linear: Vector<3, T>, angular: Vector<3, T>) -> Self {
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
        Twist {
            linear: Vector::new([a[0], a[1], a[2]]),
            angular: Vector::new([a[3], a[4], a[5]]),
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
        let v = self.linear;
        let w = self.angular;
        [v[0], v[1], v[2], w[0], w[1], w[2]]
    }

    /// A twist from a flat `[v; ω]` `Vector<6>` (the group-API ordering).
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Twist;
    /// let t = Twist::from_vector(Vector::new([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// assert_eq!(t.linear(), Vector::new([1.0, 2.0, 3.0]));
    /// ```
    #[inline]
    pub fn from_vector(v: Vector<6, T>) -> Self {
        Self::from_array(v.into_array())
    }

    /// The twist as a flat `[v; ω]` `Vector<6>` for the group API.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Twist;
    /// let t = Twist::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(t.to_vector(), Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// ```
    #[inline]
    pub fn to_vector(self) -> Vector<6, T> {
        Vector::new(self.as_array())
    }

    /// The linear (translational) part `v`.
    #[inline]
    pub fn linear(self) -> Vector<3, T> {
        self.linear
    }

    /// The angular (rotational) part `ω`.
    #[inline]
    pub fn angular(self) -> Vector<3, T> {
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

impl<T: Numeric> From<Vector<6, T>> for Twist<T> {
    /// Reinterprets a flat `[v; ω]` `Vector<6>` as a twist.
    #[inline]
    fn from(v: Vector<6, T>) -> Self {
        Self::from_vector(v)
    }
}

impl<T: Numeric> From<Twist<T>> for Vector<6, T> {
    /// Flattens a twist into `[v; ω]`.
    #[inline]
    fn from(t: Twist<T>) -> Self {
        t.to_vector()
    }
}



impl<T: Numeric> From<[T; 6]> for Twist<T> {
    /// Builds a twist from a `[vx, vy, vz, ωx, ωy, ωz]` array.
    #[inline]
    fn from(a: [T; 6]) -> Self {
        Twist::from_array(a)
    }
}

impl<T: Numeric> Mul<T> for Twist<T> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: T) -> Self {
        self.scale(scalar)
    }
}