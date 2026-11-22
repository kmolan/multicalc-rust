//! Fixed-size, stack-allocated column vector.

use core::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign};

use crate::scalar::Numeric;

/// A column vector of `N` components, stored inline on the stack.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// let vector_a = Vector::new([1.0, 2.0, 3.0]);
/// let vector_b = Vector::from([4.0, 5.0, 6.0]);
///
/// assert_eq!(vector_a[0], 1.0);
/// assert_eq!(vector_a.get(0), Some(&1.0));
/// assert_eq!(vector_a + vector_b, Vector::new([5.0, 7.0, 9.0]));
/// assert_eq!(vector_b - vector_a, Vector::new([3.0, 3.0, 3.0]));
/// assert_eq!(-vector_a, Vector::new([-1.0, -2.0, -3.0]));
/// assert_eq!(vector_a * 2.0, Vector::new([2.0, 4.0, 6.0]));
/// assert_eq!(vector_a.dot(vector_b), 32.0);
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
    /// let vector = Vector::<4>::from_fn(|i| i as f64);
    /// assert_eq!(vector.into_array(), [0.0, 1.0, 2.0, 3.0]);
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
    /// let mut vector = Vector::new([1.0, 2.0]);
    /// vector.as_mut_slice()[0] = 9.0;
    /// assert_eq!(vector[0], 9.0);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    // Crate-internal panic path (also used by Index). Public: prefer `[]`; use `get` when fallible.
    #[inline]
    #[track_caller]
    #[must_use]
    pub(crate) fn get_unchecked(&self, i: usize) -> &T {
        #[allow(clippy::indexing_slicing)]
        &self.data[i]
    }

    #[inline]
    #[track_caller]
    pub(crate) fn get_unchecked_mut(&mut self, i: usize) -> &mut T {
        #[allow(clippy::indexing_slicing)]
        &mut self.data[i]
    }

    /// Returns a reference to component `i`, or `None` if `i >= N`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let vector = Vector::new([1.0, 2.0]);
    /// assert_eq!(vector.get(0), Some(&1.0));
    /// assert_eq!(vector.get(2), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&T> {
        self.data.get(i)
    }

    /// Returns a mutable reference to component `i`, or `None` if `i >= N`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([1.0, 2.0]);
    /// if let Some(x) = vector.get_mut(1) {
    ///     *x = 9.0;
    /// }
    /// assert_eq!(vector.get(1), Some(&9.0));
    /// ```
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        self.data.get_mut(i)
    }

    /// Consumes the vector, returning its components.
    #[inline]
    #[must_use]
    pub fn into_array(self) -> [T; N] {
        self.data
    }

    /// Construct a new vector by applying a function to each component.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let vector = Vector::new([1.0, 2.0, 3.0]);
    /// let mapped_vector = vector.map(|x| 2.0 * x);
    /// assert_eq!(mapped_vector, Vector::new([2.0, 4.0, 6.0]));
    /// ```
    #[inline]
    pub fn map<F, U>(self, f: F) -> Vector<N, U>
    where
        F: Fn(T) -> U,
    {
        Vector::new(self.data.map(f))
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
    /// let vector: Vector<3> = Vector::zeros();
    /// assert_eq!(vector.into_array(), [0.0, 0.0, 0.0]);
    /// ```
    #[inline]
    pub fn zeros() -> Self {
        Vector::from_fn(|_| T::ZERO)
    }

    /// Multiplies every component by `scalar`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let vector = Vector::new([1.0, 2.0]);
    /// let factor = 3.0;
    /// assert_eq!(vector.scale(factor), Vector::new([3.0, 6.0]));
    /// ```
    #[inline]
    pub fn scale(self, scalar: T) -> Self {
        Vector::from_fn(|i| self.data[i] * scalar)
    }

    /// The dot product `Σ self[i] * rhs[i]`, summed left to right.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let vector_a = Vector::new([1.0, 2.0, 3.0]);
    /// let vector_b = Vector::new([4.0, 5.0, 6.0]);
    /// assert_eq!(vector_a.dot(vector_b), 32.0);
    ///
    /// // perpendicular vectors have a zero dot product
    /// let vector_along_x = Vector::new([1.0, 0.0]);
    /// let vector_along_y = Vector::new([0.0, 1.0]);
    /// assert_eq!(vector_along_x.dot(vector_along_y), 0.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> T {
        let mut sum = T::ZERO;
        for (&component_a, &component_b) in self.data.iter().zip(&rhs.data) {
            sum += component_a * component_b;
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

    /// Returns a normalized copy of the vector (i.e. one where the norm is equal to 1).
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert_eq!(Vector::new([30.0, 40.0]).normalized(), Vector::new([0.6, 0.8]));
    /// ```
    #[inline]
    pub fn normalized(self) -> Self {
        self / self.norm()
    }

    /// Attempts to return a normalized copy of the vector (i.e. one where the norm is equal to 1),
    /// returning `None` in the case of a zero vector or a vector with a `NAN` entry.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// assert_eq!(Vector::new([30.0, 40.0]).try_normalized(), Some(Vector::new([0.6, 0.8])));
    /// assert_eq!(Vector::new([0.0, 0.0]).try_normalized(), None);
    /// assert_eq!(Vector::new([f64::NAN, 0.0]).try_normalized(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn try_normalized(self) -> Option<Self> {
        let norm = self.norm();
        if norm.is_nan() || norm == T::ZERO {
            return None;
        }
        Some(self / norm)
    }

    /// Normalize the vector in-place (i.e. after this operation the norm is equal to 1).
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([30.0, 40.0]);
    /// vector.normalize();
    /// assert_eq!(vector, Vector::new([0.6, 0.8]));
    /// ```
    #[inline]
    pub fn normalize(&mut self) {
        let norm = self.norm();
        self.do_normalize(norm);
    }

    /// Attempt to normalize the vector in-place (i.e. after this operation the norm is equal to 1).
    /// This method returns `None` and leaves the vector unchanged if the norm is zero or NAN.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([30.0, 40.0]);
    /// assert!(vector.try_normalize().is_some());
    /// assert_eq!(vector, Vector::new([0.6, 0.8]));
    ///
    /// let mut vector = Vector::new([0.0, 0.0]);
    /// assert!(vector.try_normalize().is_none());
    /// assert_eq!(vector, Vector::new([0.0, 0.0]));
    ///
    /// let mut vector = Vector::new([f64::NAN, 1.0]);
    /// assert!(vector.try_normalize().is_none());
    /// assert!(vector[0].is_nan());
    /// assert_eq!(vector[1], 1.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn try_normalize(&mut self) -> Option<()> {
        let norm = self.norm();
        if norm.is_nan() || norm == T::ZERO {
            return None;
        }
        self.do_normalize(norm);
        Some(())
    }

    fn do_normalize(&mut self, norm: T) {
        for x in self.data.iter_mut() {
            *x = x.safe_div(norm);
        }
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

    /// Panics if `index >= N`. Use [`Self::get`] when the index may be invalid.
    #[inline]
    #[track_caller]
    fn index(&self, index: usize) -> &T {
        self.get_unchecked(index)
    }
}

impl<const N: usize, T> IndexMut<usize> for Vector<N, T> {
    /// Panics if `index >= N`. Use [`Self::get_mut`] when the index may be invalid.
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.get_unchecked_mut(index)
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

impl<const N: usize, T: Numeric> Div<T> for Vector<N, T> {
    type Output = Self;

    #[inline]
    fn div(self, scalar: T) -> Self {
        Self::from_fn(|i| self.data[i].safe_div(scalar))
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
        let [self_x, self_y, self_z] = self.data;
        let [rhs_x, rhs_y, rhs_z] = rhs.data;
        Vector::new([
            self_y * rhs_z - self_z * rhs_y,
            self_z * rhs_x - self_x * rhs_z,
            self_x * rhs_y - self_y * rhs_x,
        ])
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
    pub fn scalar_triple(self, vector_b: Self, vector_c: Self) -> T {
        self.dot(vector_b.cross(vector_c))
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
        let [self_x, self_y] = self.data;
        let [rhs_x, rhs_y] = rhs.data;
        self_x * rhs_y - self_y * rhs_x
    }
}
