//! Fixed-size, stack-allocated column vector.

use core::ops::{Index, IndexMut};

use crate::scalar::Numeric;

/// A column vector of `N` components, stored inline on the stack.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// let v = Vector::from([1.0, 2.0, 3.0]);
/// assert_eq!(v[0], 1.0);
/// assert_eq!(v.into_array(), [1.0, 2.0, 3.0]);
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
