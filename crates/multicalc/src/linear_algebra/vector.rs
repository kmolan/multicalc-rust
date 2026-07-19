//! Fixed-size, stack-allocated column vector.

use core::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

use crate::scalar::Numeric;

/// A column vector of `N` components, stored inline on the stack.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// let a = Vector::new([1.0, 2.0, 3.0]);
/// let b = Vector::from([4.0, 5.0, 6.0]);
///
/// assert_eq!(a[0], 1.0);
/// assert_eq!(a + b, Vector::new([5.0, 7.0, 9.0]));
/// assert_eq!(b - a, Vector::new([3.0, 3.0, 3.0]));
/// assert_eq!(-a, Vector::new([-1.0, -2.0, -3.0]));
/// assert_eq!(a * 2.0, Vector::new([2.0, 4.0, 6.0]));
/// assert_eq!(a.dot(b), 32.0);
/// ```
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]

#[must_use]
pub struct Vector<const N: usize, T = f64> {
    data: [T; N],
}

impl<const N: usize, T> Vector<N, T> {
    /// Wraps `N` components into a vector.
    #[inline]
    pub const fn new(data: [T; N]) -> Self {
        Vector { data }
    }

    /// Builds a vector by calling `f` with each index in `0..N`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let v = Vector::<4>::from_fn(|i| i as f64);
    /// assert_eq!(v.into_array(), [0.0, 1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn from_fn(f: impl FnMut(usize) -> T) -> Self {
        Vector {
            data: core::array::from_fn(f),
        }
    }

    /// Borrows the components as an array.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert_eq!(Vector::new([1.0, 2.0]).as_array(), &[1.0, 2.0]);
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_array(&self) -> &[T; N] {
        &self.data
    }

    /// Borrows the components as a slice.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert_eq!(Vector::new([1.0, 2.0]).as_slice(), &[1.0, 2.0]);
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Borrows the components as a mutable slice.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut v = Vector::new([1.0, 2.0]);
    /// v.as_mut_slice()[0] = 9.0;
    /// assert_eq!(v[0], 9.0);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the vector, returning its components.
    #[inline]
    #[must_use]
    pub fn into_array(self) -> [T; N] {
        self.data
    }
}

impl<const N: usize, T: Copy> Vector<N, T> {
    /// Builds a vector from a slice, or `None` if `slice.len()` is not `N`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert!(Vector::<3>::try_from_slice(&[1.0, 2.0, 3.0]).is_some());
    /// assert!(Vector::<3>::try_from_slice(&[1.0, 2.0]).is_none());
    /// ```
    #[inline]
    #[must_use]
    pub fn try_from_slice(slice: &[T]) -> Option<Self> {
        (slice.len() == N).then(|| Self::from_fn(|i| slice[i]))
    }
}

impl<const N: usize, T: Numeric> Vector<N, T> {
    /// The zero vector.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let v: Vector<3> = Vector::zeros();
    /// assert_eq!(v.into_array(), [0.0, 0.0, 0.0]);
    /// ```
    #[inline]
    pub fn zeros() -> Self {
        Vector::from_fn(|_| T::ZERO)
    }

    /// Multiplies every component by `scalar`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert_eq!(Vector::new([1.0, 2.0]).scale(3.0), Vector::new([3.0, 6.0]));
    /// ```
    #[inline]
    pub fn scale(self, scalar: T) -> Self {
        Vector::from_fn(|i| self.data[i] * scalar)
    }

    /// The dot product `Σ self[i] * rhs[i]`, summed left to right.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert_eq!(Vector::new([1.0, 2.0, 3.0]).dot(Vector::new([4.0, 5.0, 6.0])), 32.0);
    /// assert_eq!(Vector::new([1.0, 0.0]).dot(Vector::new([0.0, 1.0])), 0.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> T {
        let mut sum = T::ZERO;
        for (&a, &b) in self.data.iter().zip(&rhs.data) {
            sum += a * b;
        }
        sum
    }

    /// The squared Euclidean norm `self · self` (no square root).
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert_eq!(Vector::new([3.0, 4.0]).norm_squared(), 25.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }

    /// The Euclidean norm `√(self · self)`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert_eq!(Vector::new([3.0, 4.0]).norm(), 5.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// Returns `true` when every component is neither infinite nor NaN.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert!(Vector::new([1.0, -2.0]).is_finite());
    /// assert!(!Vector::new([1.0, f64::NAN]).is_finite());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }
}

impl<const N: usize, T> From<[T; N]> for Vector<N, T> {
    #[inline]
    fn from(data: [T; N]) -> Self {
        Vector { data }
    }
}

impl<const N: usize, T> Index<usize> for Vector<N, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<const N: usize, T> IndexMut<usize> for Vector<N, T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

impl<const N: usize, T: Numeric> Add for Vector<N, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Vector::from_fn(|i| self.data[i] + rhs.data[i])
    }
}

impl<const N: usize, T: Numeric> AddAssign for Vector<N, T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for (a, &b) in self.data.iter_mut().zip(&rhs.data) {
            *a += b;
        }
    }
}

impl<const N: usize, T: Numeric> Sub for Vector<N, T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Vector::from_fn(|i| self.data[i] - rhs.data[i])
    }
}

impl<const N: usize, T: Numeric> SubAssign for Vector<N, T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for (a, &b) in self.data.iter_mut().zip(&rhs.data) {
            *a -= b;
        }
    }
}

impl<const N: usize, T: Numeric> Neg for Vector<N, T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Vector::from_fn(|i| -self.data[i])
    }
}

impl<const N: usize, T: Numeric> Mul<T> for Vector<N, T> {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: T) -> Self {
        self.scale(scalar)
    }
}

impl<T: Numeric> Vector<3, T> {
    /// The cross product `self × rhs`, available only for 3-D vectors.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let x = Vector::new([1.0, 0.0, 0.0]);
    /// let y = Vector::new([0.0, 1.0, 0.0]);
    /// assert_eq!(x.cross(y), Vector::new([0.0, 0.0, 1.0]));
    /// ```
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        let [ax, ay, az] = self.data;
        let [bx, by, bz] = rhs.data;
        Vector::new([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])
    }

    /// The scalar triple product `self · (b × c)`: the signed volume spanned by the three vectors.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let x = Vector::new([1.0, 0.0, 0.0]);
    /// let y = Vector::new([0.0, 1.0, 0.0]);
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// assert_eq!(x.scalar_triple(y, z), 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn scalar_triple(self, b: Self, c: Self) -> T {
        self.dot(b.cross(c))
    }
}

impl<T: Numeric> Vector<2, T> {
    /// The 2-D cross product `self[0] * rhs[1] - self[1] * rhs[0]` — the scalar z-component of the
    /// 3-D cross, available only for 2-D vectors.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let x = Vector::new([1.0, 0.0]);
    /// let y = Vector::new([0.0, 1.0]);
    /// assert_eq!(x.cross(y), 1.0);
    /// assert_eq!(y.cross(x), -1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn cross(self, rhs: Self) -> T {
        self[0] * rhs[1] - self[1] * rhs[0]
    }
}
