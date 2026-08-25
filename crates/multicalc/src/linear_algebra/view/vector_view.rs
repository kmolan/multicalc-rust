//! The borrowed vector views, and the [`Vector`] methods that hand them out.

use super::required_len;
use crate::error::LinalgError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// A borrowed, strided, read-only window of `N` components.
///
/// The stride is what lets [`MatrixView::try_column`](crate::linear_algebra::MatrixView::try_column)
/// avoid a copy: a column's components sit `row_stride` apart.
///
/// ```
/// use multicalc::linear_algebra::Matrix;
/// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let column = matrix.view().try_column(0).unwrap();
/// assert_eq!(column.stride(), 2);
/// assert_eq!(column.to_vector().into_array(), [1.0, 3.0]);
/// ```
#[derive(Debug)]
#[must_use]
pub struct VectorView<'data, const N: usize, T = f64> {
    data: &'data [T],
    offset: usize,
    stride: usize,
}

/// A borrowed, strided, writable window of `N` components.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// let mut vector = Vector::new([1.0, 2.0, 3.0]);
/// *vector.view_mut().try_get_mut(2).unwrap() = 9.0;
/// assert_eq!(vector.into_array(), [1.0, 2.0, 9.0]);
/// ```
#[derive(Debug)]
#[must_use]
pub struct VectorViewMut<'data, const N: usize, T = f64> {
    data: &'data mut [T],
    offset: usize,
    stride: usize,
}

// Written out rather than derived: a derive would demand `T: Copy`, but what is copied is the
// handle, not the components.
impl<'data, const N: usize, T> Clone for VectorView<'data, N, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'data, const N: usize, T> Copy for VectorView<'data, N, T> {}

impl<'data, const N: usize, T> VectorView<'data, N, T> {
    /// Every constructor funnels through here, so an existing view is always in range.
    #[inline]
    pub(super) fn from_parts(
        data: &'data [T],
        offset: usize,
        stride: usize,
    ) -> Result<Self, LinalgError> {
        let needed = required_len(N, 1, offset, stride, 1).ok_or(LinalgError::OutOfBounds)?;
        if needed > data.len() {
            return Err(LinalgError::OutOfBounds);
        }
        Ok(VectorView {
            data,
            offset,
            stride,
        })
    }

    /// Views the first `N` elements of a slice, or `OutOfBounds` if it is shorter.
    ///
    /// ```
    /// use multicalc::linear_algebra::VectorView;
    /// let buffer = [1.0, 2.0, 3.0];
    /// assert_eq!(VectorView::<2>::try_from_slice(&buffer).unwrap().try_get(1), Ok(&2.0));
    /// assert!(VectorView::<4>::try_from_slice(&buffer).is_err());
    /// ```
    #[inline]
    pub fn try_from_slice(slice: &'data [T]) -> Result<Self, LinalgError> {
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

    /// How far apart consecutive components sit in the buffer.
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

    /// Component `index`, or `OutOfBounds`. The length is checked here rather than left to the
    /// slice, because a strided view's out-of-range component can still land inside the parent
    /// buffer.
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::linear_algebra::Vector;
    /// let vector = Vector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(vector.view().try_get(2), Ok(&3.0));
    /// assert_eq!(vector.view().try_get(3), Err(LinalgError::OutOfBounds));
    /// ```
    #[inline]
    pub fn try_get(&self, index: usize) -> Result<&T, LinalgError> {
        self.index_of(index)
            .and_then(|flat| self.data.get(flat))
            .ok_or(LinalgError::OutOfBounds)
    }

    /// The components `start..start + LEN`, or `OutOfBounds`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let vector = Vector::new([1.0, 2.0, 3.0, 4.0]);
    /// let middle = vector.view().try_segment::<2>(1).unwrap();
    /// assert_eq!(middle.to_vector().into_array(), [2.0, 3.0]);
    /// ```
    #[inline]
    pub fn try_segment<const LEN: usize>(
        self,
        start: usize,
    ) -> Result<VectorView<'data, LEN, T>, LinalgError> {
        let offset = segment_offset(self.offset, self.stride, start, LEN, N)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorView::from_parts(self.data, offset, self.stride)
    }

    /// The first `HEAD` components and the remaining `TAIL`, or `OutOfBounds` unless they sum to
    /// `N`. Any stride does, since two shared halves may overlap.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let vector = Vector::new([1.0, 2.0, 3.0, 4.0]);
    /// let (head, tail) = vector.view().try_split_at::<1, 3>().unwrap();
    /// assert_eq!(head.to_vector().into_array(), [1.0]);
    /// assert_eq!(tail.to_vector().into_array(), [2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn try_split_at<const HEAD: usize, const TAIL: usize>(
        self,
    ) -> Result<(VectorView<'data, HEAD, T>, VectorView<'data, TAIL, T>), LinalgError> {
        if HEAD.checked_add(TAIL) != Some(N) {
            return Err(LinalgError::OutOfBounds);
        }
        Ok((
            self.try_segment::<HEAD>(0)?,
            self.try_segment::<TAIL>(HEAD)?,
        ))
    }
}

impl<'data, const N: usize, T: Copy> VectorView<'data, N, T> {
    /// Copies the window out.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let owned = matrix.view().try_column(1).unwrap().to_vector();
    /// assert_eq!(owned.into_array(), [2.0, 4.0]);
    /// ```
    // In-bounds by construction: `from_fn` only asks for components below `N`, and
    // `required_len` already proved those land inside `data`.
    #[inline]
    #[allow(clippy::indexing_slicing)]
    pub fn to_vector(self) -> Vector<N, T> {
        Vector::from_fn(|index| self.data[self.offset + index * self.stride])
    }
}

impl<'data, const N: usize, T: Numeric> VectorView<'data, N, T> {
    /// The dot product with `rhs`, summed left to right, materializing neither side.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let column = matrix.view().try_column(0).unwrap();
    /// let row = matrix.view().try_row(0).unwrap();
    /// assert_eq!(column.dot(row), 7.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: VectorView<'_, N, T>) -> T {
        let mut sum = T::ZERO;
        for index in 0..N {
            if let (Ok(left), Ok(right)) = (self.try_get(index), rhs.try_get(index)) {
                sum += *left * *right;
            }
        }
        sum
    }
}

impl<'data, const N: usize, T: PartialEq> PartialEq for VectorView<'data, N, T> {
    /// Component by component, so different strides over different buffers can be equal.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (0..N).all(|index| self.try_get(index) == other.try_get(index))
    }
}

impl<'data, const N: usize, T> VectorViewMut<'data, N, T> {
    /// Every constructor funnels through here, so an existing view is always in range.
    #[inline]
    pub(super) fn from_parts(
        data: &'data mut [T],
        offset: usize,
        stride: usize,
    ) -> Result<Self, LinalgError> {
        let needed = required_len(N, 1, offset, stride, 1).ok_or(LinalgError::OutOfBounds)?;
        if needed > data.len() {
            return Err(LinalgError::OutOfBounds);
        }
        Ok(VectorViewMut {
            data,
            offset,
            stride,
        })
    }

    /// Views the first `N` elements of a slice writably, or `OutOfBounds` if it is shorter.
    ///
    /// ```
    /// use multicalc::linear_algebra::VectorViewMut;
    /// let mut buffer = [0.0; 4];
    /// VectorViewMut::<3>::try_from_slice(&mut buffer).unwrap().fill(1.0);
    /// assert_eq!(buffer, [1.0, 1.0, 1.0, 0.0]);
    /// assert!(VectorViewMut::<5>::try_from_slice(&mut buffer).is_err());
    /// ```
    #[inline]
    pub fn try_from_slice(slice: &'data mut [T]) -> Result<Self, LinalgError> {
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

    /// How far apart consecutive components sit in the buffer.
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

    /// Component `index`, or `OutOfBounds`. See [`VectorView::try_get`].
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([1.0, 2.0, 3.0]);
    /// let view = vector.view_mut();
    /// assert_eq!(view.try_get(0), Ok(&1.0));
    /// assert_eq!(view.try_get(3), Err(LinalgError::OutOfBounds));
    /// ```
    #[inline]
    pub fn try_get(&self, index: usize) -> Result<&T, LinalgError> {
        self.index_of(index)
            .and_then(|flat| self.data.get(flat))
            .ok_or(LinalgError::OutOfBounds)
    }

    /// Component `index`, writably, or `OutOfBounds`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([1.0, 2.0, 3.0]);
    /// *vector.view_mut().try_get_mut(1).unwrap() = 9.0;
    /// assert_eq!(vector.into_array(), [1.0, 9.0, 3.0]);
    /// ```
    #[inline]
    pub fn try_get_mut(&mut self, index: usize) -> Result<&mut T, LinalgError> {
        let flat = self.index_of(index).ok_or(LinalgError::OutOfBounds)?;
        self.data.get_mut(flat).ok_or(LinalgError::OutOfBounds)
    }

    /// Borrows read-only for as long as `self` is untouched.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([3.0, 4.0]);
    /// let mut view = vector.view_mut();
    /// assert_eq!(view.as_view().dot(view.as_view()), 25.0);
    /// *view.try_get_mut(0).unwrap() = 0.0;
    /// assert_eq!(vector.into_array(), [0.0, 4.0]);
    /// ```
    #[inline]
    pub fn as_view(&self) -> VectorView<'_, N, T> {
        VectorView {
            data: self.data,
            offset: self.offset,
            stride: self.stride,
        }
    }

    /// Borrows writably for a shorter lifetime, so the original survives a consuming method.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([1.0, 2.0, 3.0, 4.0]);
    /// let mut view = vector.view_mut();
    /// let (mut head, _tail) = view.reborrow().try_split_at::<2, 2>().unwrap();
    /// head.fill(0.0);
    /// assert_eq!(view.to_vector().into_array(), [0.0, 0.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn reborrow(&mut self) -> VectorViewMut<'_, N, T> {
        VectorViewMut {
            data: self.data,
            offset: self.offset,
            stride: self.stride,
        }
    }

    /// The components `start..start + LEN`, writably. See [`VectorView::try_segment`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([1.0, 2.0, 3.0, 4.0]);
    /// vector.view_mut().try_segment::<2>(1).unwrap().fill(0.0);
    /// assert_eq!(vector.into_array(), [1.0, 0.0, 0.0, 4.0]);
    /// ```
    #[inline]
    pub fn try_segment<const LEN: usize>(
        self,
        start: usize,
    ) -> Result<VectorViewMut<'data, LEN, T>, LinalgError> {
        let offset = segment_offset(self.offset, self.stride, start, LEN, N)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorViewMut::from_parts(self.data, offset, self.stride)
    }

    /// Two halves that can be written through at once. Needs `HEAD + TAIL == N` and stride `1`,
    /// since a strided view interleaves its components with something else.
    /// [`VectorView::try_split_at`] has no such requirement.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::<4>::zeros();
    /// let (mut head, mut tail) = vector.view_mut().try_split_at::<1, 3>().unwrap();
    /// *head.try_get_mut(0).unwrap() = 1.0;
    /// *tail.try_get_mut(2).unwrap() = 4.0;
    /// assert_eq!(vector.into_array(), [1.0, 0.0, 0.0, 4.0]);
    /// ```
    #[inline]
    pub fn try_split_at<const HEAD: usize, const TAIL: usize>(
        self,
    ) -> Result<(VectorViewMut<'data, HEAD, T>, VectorViewMut<'data, TAIL, T>), LinalgError> {
        if HEAD.checked_add(TAIL) != Some(N) || self.stride != 1 {
            return Err(LinalgError::OutOfBounds);
        }
        let split = self
            .offset
            .checked_add(HEAD)
            .ok_or(LinalgError::OutOfBounds)?;
        if split > self.data.len() {
            return Err(LinalgError::OutOfBounds);
        }
        let (head, tail) = self.data.split_at_mut(split);
        Ok((
            VectorViewMut::from_parts(head, self.offset, 1)?,
            VectorViewMut::from_parts(tail, 0, 1)?,
        ))
    }
}

impl<'data, const N: usize, T: Copy> VectorViewMut<'data, N, T> {
    /// Copies the window out.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([1.0, 2.0, 3.0]);
    /// assert_eq!(vector.view_mut().to_vector().into_array(), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn to_vector(&self) -> Vector<N, T> {
        self.as_view().to_vector()
    }

    /// Overwrites every component with `value`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([1.0, 2.0, 3.0]);
    /// vector.view_mut().fill(0.0);
    /// assert_eq!(vector.into_array(), [0.0, 0.0, 0.0]);
    /// ```
    #[inline]
    pub fn fill(&mut self, value: T) {
        for index in 0..N {
            if let Ok(slot) = self.try_get_mut(index) {
                *slot = value;
            }
        }
    }

    /// Copies `source` in component by component; the strides may differ.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let mut vector = Vector::new([0.0, 0.0]);
    /// vector.view_mut().copy_from(matrix.view().try_column(1).unwrap());
    /// assert_eq!(vector.into_array(), [2.0, 4.0]);
    /// ```
    #[inline]
    pub fn copy_from(&mut self, source: VectorView<'_, N, T>) {
        for index in 0..N {
            if let (Ok(value), Ok(slot)) = (source.try_get(index).copied(), self.try_get_mut(index))
            {
                *slot = value;
            }
        }
    }
}

impl<'data, const N: usize, T: Numeric> VectorViewMut<'data, N, T> {
    /// The dot product with `rhs`, without giving up the write access. See [`VectorView::dot`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let mut vector = Vector::new([3.0, 4.0]);
    /// let unit = Vector::new([1.0, 0.0]);
    /// assert_eq!(vector.view_mut().dot(unit.view()), 3.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(&self, rhs: VectorView<'_, N, T>) -> T {
        self.as_view().dot(rhs)
    }
}

impl<'data, const N: usize, T: PartialEq> PartialEq for VectorViewMut<'data, N, T> {
    /// Component by component, matching [`VectorView`].
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_view() == other.as_view()
    }
}

// `None` if the range runs past the end of a view of `total` components, or overflows.
#[inline]
fn segment_offset(
    offset: usize,
    stride: usize,
    start: usize,
    len: usize,
    total: usize,
) -> Option<usize> {
    (start.checked_add(len)? <= total).then_some(())?;
    offset.checked_add(start.checked_mul(stride)?)
}

impl<const N: usize, T> Vector<N, T> {
    /// A read-only view of the whole vector.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    /// let vector = Vector::new([1.0, 2.0, 3.0]);
    /// let tail = vector.view().try_segment::<2>(1).unwrap();
    /// assert_eq!(tail.to_vector().into_array(), [2.0, 3.0]);
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
    /// let mut vector = Vector::<3>::zeros();
    /// *vector.view_mut().try_get_mut(2).unwrap() = 7.0;
    /// assert_eq!(vector.into_array(), [0.0, 0.0, 7.0]);
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
