//! The borrowed vector views, and the [`Vector`] methods that hand them out.

use core::ops::{Index, IndexMut};

use super::required_len;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

// Panics for the same reasons `matrix_out_of_bounds` does: `Index` returns `&T` and has no way
// to report an error, `Vector` already behaves this way, and a strided view cannot leave the
// bound to the slice because an out-of-range subscript can still land inside the buffer.
#[cold]
#[track_caller]
#[allow(clippy::panic)]
fn vector_out_of_bounds(index: usize, len: usize) -> ! {
    panic!("vector view index {index} out of range for a view of {len} components")
}

/// A borrowed, strided, read-only window of `N` components.
///
/// A stride other than `1` is what lets
/// [`MatrixView::column`](crate::linear_algebra::MatrixView::column) avoid a copy: the components of a
/// column sit `row_stride` apart in the flat buffer.
#[derive(Debug)]
#[must_use]
pub struct VectorView<'a, const N: usize, T = f64> {
    data: &'a [T],
    offset: usize,
    stride: usize,
}

/// A borrowed, strided, writable window of `N` components.
#[derive(Debug)]
#[must_use]
pub struct VectorViewMut<'a, const N: usize, T = f64> {
    data: &'a mut [T],
    offset: usize,
    stride: usize,
}

// Written out rather than derived: a derive would demand `T: Copy`, but what is copied is the
// handle -- slice reference, offset, stride -- not the components.
impl<const N: usize, T> Clone for VectorView<'_, N, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<const N: usize, T> Copy for VectorView<'_, N, T> {}

impl<'a, const N: usize, T> VectorView<'a, N, T> {
    /// Builds a view over `data`, or `None` if the span would reach past the end of the slice.
    #[inline]
    pub(super) fn from_parts(data: &'a [T], offset: usize, stride: usize) -> Option<Self> {
        let needed = required_len(N, 1, offset, stride, 1)?;
        (needed <= data.len()).then_some(VectorView {
            data,
            offset,
            stride,
        })
    }

    /// Views the first `N` elements of a slice, or `None` if it is shorter than that.
    ///
    /// ```
    /// use multicalc::linear_algebra::VectorView;
    /// let buffer = [1.0, 2.0, 3.0];
    /// assert_eq!(VectorView::<2>::from_slice(&buffer).unwrap()[1], 2.0);
    /// assert!(VectorView::<4>::from_slice(&buffer).is_none());
    /// ```
    #[inline]
    pub fn from_slice(slice: &'a [T]) -> Option<Self> {
        Self::from_parts(slice, 0, 1)
    }

    /// The component count, `N`.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        N
    }

    /// Whether the view has no components.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// How far apart consecutive components sit in the underlying buffer. A column of a row-major
    /// matrix has the matrix's row stride here.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    fn index_of(&self, index: usize) -> Option<usize> {
        (index < N).then_some(())?;
        self.offset.checked_add(index.checked_mul(self.stride)?)
    }

    /// Returns a reference to component `index`, or `None` if `index >= N`.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(self.index_of(index)?)
    }

    /// The components `start..start + M`, or `None` if that range runs past the end.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let v = Vector::new([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(v.view().segment::<2>(1).unwrap().to_vector().into_array(), [2.0, 3.0]);
    /// ```
    #[inline]
    pub fn segment<const M: usize>(self, start: usize) -> Option<VectorView<'a, M, T>> {
        (start.checked_add(M)? <= N).then_some(())?;
        let offset = self.offset.checked_add(start.checked_mul(self.stride)?)?;
        VectorView::from_parts(self.data, offset, self.stride)
    }
}

impl<const N: usize, T: Copy> VectorView<'_, N, T> {
    /// Copies the window into an owned vector.
    #[inline]
    pub fn to_vector(self) -> Vector<N, T> {
        Vector::from_fn(|index| self[index])
    }
}

impl<const N: usize, T: Numeric> VectorView<'_, N, T> {
    /// The dot product with `rhs`, summed left to right, without materializing either side.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// // Column 0 dotted with row 0, neither of them copied.
    /// assert_eq!(m.view().column(0).unwrap().dot(m.view().row(0).unwrap()), 7.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: VectorView<'_, N, T>) -> T {
        let mut sum = T::ZERO;
        for i in 0..N {
            if let (Some(a), Some(b)) = (self.get(i), rhs.get(i)) {
                sum += *a * *b;
            }
        }
        sum
    }
}

impl<const N: usize, T> Index<usize> for VectorView<'_, N, T> {
    type Output = T;

    /// Panics if `index >= N`. Use [`Self::get`] when it may be out of range.
    #[inline]
    #[track_caller]
    fn index(&self, index: usize) -> &T {
        match self.get(index) {
            Some(value) => value,
            None => vector_out_of_bounds(index, N),
        }
    }
}

impl<const N: usize, T: PartialEq> PartialEq for VectorView<'_, N, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (0..N).all(|i| self.get(i) == other.get(i))
    }
}

impl<'a, const N: usize, T> VectorViewMut<'a, N, T> {
    /// Builds a view over `data`, or `None` if the span would reach past the end of the slice.
    #[inline]
    pub(super) fn from_parts(data: &'a mut [T], offset: usize, stride: usize) -> Option<Self> {
        let needed = required_len(N, 1, offset, stride, 1)?;
        (needed <= data.len()).then_some(VectorViewMut {
            data,
            offset,
            stride,
        })
    }

    /// Views the first `N` elements of a slice writably, or `None` if it is shorter than that.
    #[inline]
    pub fn from_slice(slice: &'a mut [T]) -> Option<Self> {
        Self::from_parts(slice, 0, 1)
    }

    /// The component count, `N`.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        N
    }

    /// Whether the view has no components.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// How far apart consecutive components sit in the underlying buffer.
    #[inline]
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    fn index_of(&self, index: usize) -> Option<usize> {
        (index < N).then_some(())?;
        self.offset.checked_add(index.checked_mul(self.stride)?)
    }

    /// Returns a reference to component `index`, or `None` if `index >= N`.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(self.index_of(index)?)
    }

    /// Returns a mutable reference to component `index`, or `None` if `index >= N`.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        let at = self.index_of(index)?;
        self.data.get_mut(at)
    }

    /// Borrows this window read-only for as long as `self` is untouched.
    #[inline]
    pub fn as_view(&self) -> VectorView<'_, N, T> {
        VectorView {
            data: self.data,
            offset: self.offset,
            stride: self.stride,
        }
    }

    /// Borrows this window writably for a shorter lifetime.
    #[inline]
    pub fn reborrow(&mut self) -> VectorViewMut<'_, N, T> {
        VectorViewMut {
            data: self.data,
            offset: self.offset,
            stride: self.stride,
        }
    }

    /// Splits into the first `HEAD` components and the remaining `TAIL`, as two views that can be
    /// written through at the same time.
    ///
    /// Returns `None` unless `HEAD + TAIL == N` and the stride is `1` — a strided view interleaves
    /// its components with something else, so no cut of the slice separates them.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut v = Vector::<4>::zeros();
    /// let (mut head, mut tail) = v.view_mut().split_at::<1, 3>().unwrap();
    /// head[0] = 1.0;
    /// tail[2] = 4.0;
    /// assert_eq!(v.into_array(), [1.0, 0.0, 0.0, 4.0]);
    /// ```
    #[inline]
    pub fn split_at<const HEAD: usize, const TAIL: usize>(
        self,
    ) -> Option<(VectorViewMut<'a, HEAD, T>, VectorViewMut<'a, TAIL, T>)> {
        (HEAD.checked_add(TAIL)? == N && self.stride == 1).then_some(())?;
        let split = self.offset.checked_add(HEAD)?;
        (split <= self.data.len()).then_some(())?;
        let (head, tail) = self.data.split_at_mut(split);
        Some((
            VectorViewMut::from_parts(head, self.offset, 1)?,
            VectorViewMut::from_parts(tail, 0, 1)?,
        ))
    }
}

impl<const N: usize, T: Copy> VectorViewMut<'_, N, T> {
    /// Copies the window into an owned vector.
    #[inline]
    pub fn to_vector(&self) -> Vector<N, T> {
        self.as_view().to_vector()
    }

    /// Overwrites every component with `value`.
    #[inline]
    pub fn fill(&mut self, value: T) {
        for i in 0..N {
            if let Some(slot) = self.get_mut(i) {
                *slot = value;
            }
        }
    }

    /// Copies `src` in component by component; the two may have different strides.
    #[inline]
    pub fn copy_from(&mut self, src: VectorView<'_, N, T>) {
        for i in 0..N {
            if let (Some(value), Some(slot)) = (src.get(i).copied(), self.get_mut(i)) {
                *slot = value;
            }
        }
    }
}

impl<const N: usize, T> Index<usize> for VectorViewMut<'_, N, T> {
    type Output = T;

    /// Panics if `index >= N`. Use [`Self::get`] when it may be out of range.
    #[inline]
    #[track_caller]
    fn index(&self, index: usize) -> &T {
        match self.get(index) {
            Some(value) => value,
            None => vector_out_of_bounds(index, N),
        }
    }
}

impl<const N: usize, T> IndexMut<usize> for VectorViewMut<'_, N, T> {
    /// Panics if `index >= N`. Use [`Self::get_mut`] when it may be out of range.
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut T {
        match self.get_mut(index) {
            Some(value) => value,
            None => vector_out_of_bounds(index, N),
        }
    }
}

impl<const N: usize, T> Vector<N, T> {
    /// A read-only view of the whole vector.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let v = Vector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(v.view().segment::<2>(1).unwrap().to_vector().into_array(), [2.0, 3.0]);
    /// ```
    #[inline]
    pub fn view(&self) -> VectorView<'_, N, T> {
        VectorView {
            data: self.as_slice(),
            offset: 0,
            stride: 1,
        }
    }

    /// A writable view of the whole vector.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut v = Vector::<3>::zeros();
    /// v.view_mut()[2] = 7.0;
    /// assert_eq!(v.into_array(), [0.0, 0.0, 7.0]);
    /// ```
    #[inline]
    pub fn view_mut(&mut self) -> VectorViewMut<'_, N, T> {
        VectorViewMut {
            data: self.as_mut_slice(),
            offset: 0,
            stride: 1,
        }
    }
}
