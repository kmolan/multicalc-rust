//! Typed spatial force.

use core::ops::{Add, Mul, Neg, Sub};

use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// A spatial force (wrench), stored force-first in the `[force; torque]` ordering — reciprocal to a
/// [`Twist`](crate::spatial::Twist), so the two line up component-for-component.
///
/// Like `Twist`, the type owns its layout: the only value constructor takes the force and torque
/// parts by name, and the flat converters emit `[f; τ]`. `Add`/`Sub`/`Neg`/[`scale`](Wrench::scale)
/// act component-wise; the spatial *algebra* (coordinate transforms, the `twist · wrench` power
/// product) is not defined here.
///
/// ```
/// use multicalc::spatial::Wrench;
/// let a = Wrench::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let b = Wrench::from_array([1.0_f64; 6]);
/// assert_eq!((a + b).as_array(), [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
/// assert_eq!((a - b).as_array(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert_eq!((-a).as_array(), [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]);
/// assert_eq!(a.scale(2.0).as_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wrench<T: Numeric> {
    force: Vector<3, T>,
    torque: Vector<3, T>,
}

impl<T: Numeric> Wrench<T> {
    /// A wrench from its force and torque parts.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Wrench;
    /// let w = Wrench::new(Vector::new([1.0_f64, 2.0, 3.0]), Vector::new([4.0, 5.0, 6.0]));
    /// assert_eq!(w.force(), Vector::new([1.0, 2.0, 3.0]));
    /// assert_eq!(w.torque(), Vector::new([4.0, 5.0, 6.0]));
    /// ```
    #[inline]
    pub fn new(force: Vector<3, T>, torque: Vector<3, T>) -> Self {
        Wrench { force, torque }
    }

    /// The zero wrench.
    ///
    /// ```
    /// use multicalc::spatial::Wrench;
    /// assert_eq!(Wrench::<f64>::zeros().as_array(), [0.0; 6]);
    /// ```
    #[inline]
    pub fn zeros() -> Self {
        Wrench {
            force: Vector::zeros(),
            torque: Vector::zeros(),
        }
    }

    /// A wrench from a `[fx, fy, fz, τx, τy, τz]` array.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Wrench;
    /// let w = Wrench::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(w.torque(), Vector::new([4.0, 5.0, 6.0]));
    /// ```
    #[inline]
    pub fn from_array(a: [T; 6]) -> Self {
        Wrench {
            force: Vector::new([a[0], a[1], a[2]]),
            torque: Vector::new([a[3], a[4], a[5]]),
        }
    }

    /// The wrench as a `[fx, fy, fz, τx, τy, τz]` array.
    ///
    /// ```
    /// use multicalc::spatial::Wrench;
    /// let w = Wrench::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(w.as_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    #[inline]
    pub fn as_array(self) -> [T; 6] {
        let f = self.force;
        let t = self.torque;
        [f[0], f[1], f[2], t[0], t[1], t[2]]
    }

    /// A wrench from a flat `[f; τ]` `Vector<6>`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Wrench;
    /// let w = Wrench::from_vector(Vector::new([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// assert_eq!(w.force(), Vector::new([1.0, 2.0, 3.0]));
    /// ```
    #[inline]
    pub fn from_vector(v: Vector<6, T>) -> Self {
        Self::from_array(v.into_array())
    }

    /// The wrench as a flat `[f; τ]` `Vector<6>`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::Wrench;
    /// let w = Wrench::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(w.to_vector(), Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    /// ```
    #[inline]
    pub fn to_vector(self) -> Vector<6, T> {
        Vector::new(self.as_array())
    }

    /// The force part `f`.
    #[inline]
    pub fn force(self) -> Vector<3, T> {
        self.force
    }

    /// The torque (moment) part `τ`.
    #[inline]
    pub fn torque(self) -> Vector<3, T> {
        self.torque
    }

    /// Multiplies both parts by `scalar`.
    ///
    /// ```
    /// use multicalc::spatial::Wrench;
    /// let w = Wrench::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(w.scale(2.0).as_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    /// ```
    #[inline]
    pub fn scale(self, scalar: T) -> Self {
        Wrench {
            force: self.force.scale(scalar),
            torque: self.torque.scale(scalar),
        }
    }
}

impl<T: Numeric> Add for Wrench<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Wrench {
            force: self.force + rhs.force,
            torque: self.torque + rhs.torque,
        }
    }
}

impl<T: Numeric> Sub for Wrench<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Wrench {
            force: self.force - rhs.force,
            torque: self.torque - rhs.torque,
        }
    }
}

impl<T: Numeric> Neg for Wrench<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Wrench {
            force: -self.force,
            torque: -self.torque,
        }
    }
}

impl<T: Numeric> From<Vector<6, T>> for Wrench<T> {
    /// Reinterprets a flat `[f; τ]` `Vector<6>` as a wrench.
    #[inline]
    fn from(v: Vector<6, T>) -> Self {
        Self::from_vector(v)
    }
}

impl<T: Numeric> From<Wrench<T>> for Vector<6, T> {
    /// Flattens a wrench into `[f; τ]`.
    #[inline]
    fn from(w: Wrench<T>) -> Self {
        w.to_vector()
    }
}



impl<T: Numeric> From<[T; 6]> for Wrench<T> {
    /// Builds a wrench from a `[fx, fy, fz, τx, τy, τz]` array.
    #[inline]
    fn from(a: [T; 6]) -> Self {
        Wrench::from_array(a)
    }
}

impl<T: Numeric> Mul<T> for Wrench<T> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: T) -> Self {
        self.scale(scalar)
    }
}