//! The borrowed matrix views, and the [`Matrix`] methods that hand them out.

use super::{VectorView, VectorViewMut, required_len};
use crate::error::LinalgError;
use crate::linear_algebra::Matrix;

/// A borrowed, strided, read-only `ROWS`×`COLS` window onto someone else's storage.
///
/// ```
/// use multicalc::linear_algebra::Matrix;
/// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
/// let corner = matrix.view().try_submatrix::<2, 2>(1, 1).unwrap();
/// assert_eq!(corner.to_matrix().into_array(), [[5.0, 6.0], [8.0, 9.0]]);
/// ```
#[derive(Debug)]
#[must_use]
pub struct MatrixView<'data, const ROWS: usize, const COLS: usize, T = f64> {
    data: &'data [T],
    offset: usize,
    row_stride: usize,
    col_stride: usize,
}

/// A borrowed, strided, writable `ROWS`×`COLS` window onto someone else's storage.
///
/// Reshaping consumes the view, since the exclusive borrow moves with it;
/// [`as_view`](Self::as_view) and [`reborrow`](Self::reborrow) hand out shorter-lived ones.
///
/// ```
/// use multicalc::linear_algebra::Matrix;
/// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// *matrix.view_mut().transposed().try_get_mut(0, 1).unwrap() = 9.0;
/// assert_eq!(matrix.into_array(), [[1.0, 2.0], [9.0, 4.0]]);
/// ```
#[derive(Debug)]
#[must_use]
pub struct MatrixViewMut<'data, const ROWS: usize, const COLS: usize, T = f64> {
    data: &'data mut [T],
    offset: usize,
    row_stride: usize,
    col_stride: usize,
}

// Written out rather than derived: a derive would demand `T: Copy`, but what is copied is the
// handle, not the elements.
impl<'data, const ROWS: usize, const COLS: usize, T> Clone for MatrixView<'data, ROWS, COLS, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'data, const ROWS: usize, const COLS: usize, T> Copy for MatrixView<'data, ROWS, COLS, T> {}

impl<'data, const ROWS: usize, const COLS: usize, T> MatrixView<'data, ROWS, COLS, T> {
    /// Every constructor funnels through here, so an existing view is always in range.
    #[inline]
    fn from_parts(
        data: &'data [T],
        offset: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<Self, LinalgError> {
        let needed = required_len(ROWS, COLS, offset, row_stride, col_stride)
            .ok_or(LinalgError::OutOfBounds)?;
        if needed > data.len() {
            return Err(LinalgError::OutOfBounds);
        }
        Ok(MatrixView {
            data,
            offset,
            row_stride,
            col_stride,
        })
    }

    /// Views a row-major slice, ignoring any trailing elements, or `OutOfBounds` if it holds
    /// fewer than `ROWS * COLS`.
    ///
    /// ```
    /// use multicalc::linear_algebra::MatrixView;
    /// let buffer = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let view = MatrixView::<2, 2>::try_from_row_major_slice(&buffer).unwrap();
    /// assert_eq!(view.try_get(1, 0), Ok(&3.0));
    /// assert!(MatrixView::<3, 2>::try_from_row_major_slice(&buffer).is_err());
    /// ```
    #[inline]
    pub fn try_from_row_major_slice(slice: &'data [T]) -> Result<Self, LinalgError> {
        Self::from_parts(slice, 0, COLS, 1)
    }

    /// The row count, `ROWS`.
    #[inline]
    #[must_use]
    pub const fn rows(&self) -> usize {
        ROWS
    }

    /// The column count, `COLS`.
    #[inline]
    #[must_use]
    pub const fn cols(&self) -> usize {
        COLS
    }

    /// How far apart consecutive rows and columns sit in the buffer: `(COLS, 1)` for a fresh
    /// view of a [`Matrix`], `(1, COLS)` transposed.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// assert_eq!(matrix.view().strides(), (3, 1));
    /// assert_eq!(matrix.view().transposed().strides(), (1, 3));
    /// ```
    #[inline]
    #[must_use]
    pub const fn strides(&self) -> (usize, usize) {
        (self.row_stride, self.col_stride)
    }

    /// Whether the rows are unpadded runs, which is what
    /// [`MatrixViewMut::try_split_rows_at`] needs.
    #[inline]
    #[must_use]
    pub const fn is_row_major(&self) -> bool {
        // With one column or none, `col_stride` addresses nothing and any row stride wide
        // enough to keep the rows apart is separable.
        self.row_stride >= COLS && (COLS <= 1 || self.col_stride == 1)
    }

    #[inline]
    fn index_of(&self, row: usize, column: usize) -> Option<usize> {
        (row < ROWS && column < COLS).then_some(())?;
        self.offset
            .checked_add(row.checked_mul(self.row_stride)?)?
            .checked_add(column.checked_mul(self.col_stride)?)
    }

    #[inline]
    fn block_offset(
        &self,
        top: usize,
        left: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> Option<usize> {
        (top.checked_add(block_rows)? <= ROWS && left.checked_add(block_cols)? <= COLS)
            .then_some(())?;
        self.offset
            .checked_add(top.checked_mul(self.row_stride)?)?
            .checked_add(left.checked_mul(self.col_stride)?)
    }

    /// Entry `(row, column)`, or `OutOfBounds`. The shape is checked here rather than left to
    /// the slice, because a strided view's out-of-range subscript can still land on a real
    /// element of the parent buffer.
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.view().try_get(1, 0), Ok(&3.0));
    /// assert_eq!(matrix.view().try_get(2, 0), Err(LinalgError::OutOfBounds));
    /// ```
    #[inline]
    pub fn try_get(&self, row: usize, column: usize) -> Result<&T, LinalgError> {
        self.index_of(row, column)
            .and_then(|flat| self.data.get(flat))
            .ok_or(LinalgError::OutOfBounds)
    }

    /// The transpose: the two strides trade places.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0]]);
    /// let transposed = matrix.view().transposed();
    /// assert_eq!((transposed.rows(), transposed.cols()), (3, 1));
    /// assert_eq!(transposed.to_matrix().into_array(), [[1.0], [2.0], [3.0]]);
    /// ```
    #[inline]
    pub fn transposed(self) -> MatrixView<'data, COLS, ROWS, T> {
        MatrixView {
            data: self.data,
            offset: self.offset,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    /// The `BLOCK_ROWS`×`BLOCK_COLS` block at `(top, left)`, or `OutOfBounds` if it runs off an
    /// edge. Only the offset moves; the strides carry over. The size is a const parameter
    /// because it is part of the returned type, while the corner is not.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let block = matrix.view().try_submatrix::<2, 2>(0, 1).unwrap();
    /// assert_eq!(block.to_matrix().into_array(), [[2.0, 3.0], [5.0, 6.0]]);
    /// assert!(matrix.view().try_submatrix::<2, 2>(0, 2).is_err());
    /// ```
    #[inline]
    pub fn try_submatrix<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
        self,
        top: usize,
        left: usize,
    ) -> Result<MatrixView<'data, BLOCK_ROWS, BLOCK_COLS, T>, LinalgError> {
        let offset = self
            .block_offset(top, left, BLOCK_ROWS, BLOCK_COLS)
            .ok_or(LinalgError::OutOfBounds)?;
        MatrixView::from_parts(self.data, offset, self.row_stride, self.col_stride)
    }

    /// Row `row` at stride `col_stride`, or `OutOfBounds`. Unlike [`Matrix::try_row`], nothing
    /// is copied.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.view().try_row(1).unwrap().to_vector().into_array(), [3.0, 4.0]);
    /// ```
    #[inline]
    pub fn try_row(self, row: usize) -> Result<VectorView<'data, COLS, T>, LinalgError> {
        let offset = self
            .block_offset(row, 0, 1, COLS)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorView::from_parts(self.data, offset, self.col_stride)
    }

    /// Column `column` at stride `row_stride`, or `OutOfBounds`. Unlike [`Matrix::try_column`],
    /// nothing is copied — carrying the stride is what saves it.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.view().try_column(1).unwrap().to_vector().into_array(), [2.0, 4.0]);
    /// ```
    #[inline]
    pub fn try_column(self, column: usize) -> Result<VectorView<'data, ROWS, T>, LinalgError> {
        let offset = self
            .block_offset(0, column, ROWS, 1)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorView::from_parts(self.data, offset, self.row_stride)
    }

    /// The main diagonal, at stride `row_stride + col_stride`. `LEN` has to be `min(ROWS, COLS)`,
    /// which the type system cannot yet work out, so it is checked instead.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let diagonal = matrix.view().try_diagonal::<2>().unwrap();
    /// assert_eq!(diagonal.to_vector().into_array(), [1.0, 5.0]);
    /// ```
    #[inline]
    pub fn try_diagonal<const LEN: usize>(self) -> Result<VectorView<'data, LEN, T>, LinalgError> {
        let stride = diagonal_stride(LEN, ROWS, COLS, self.row_stride, self.col_stride)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorView::from_parts(self.data, self.offset, stride)
    }

    /// The first `TOP` rows and the remaining `BOTTOM`, or `OutOfBounds` unless they sum to
    /// `ROWS`. Any layout does, since two shared halves may overlap.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    /// let (top, bottom) = matrix.view().try_split_rows_at::<1, 2>().unwrap();
    /// assert_eq!(top.to_matrix().into_array(), [[1.0, 2.0]]);
    /// assert_eq!(bottom.to_matrix().into_array(), [[3.0, 4.0], [5.0, 6.0]]);
    /// ```
    #[inline]
    pub fn try_split_rows_at<const TOP: usize, const BOTTOM: usize>(
        self,
    ) -> Result<
        (
            MatrixView<'data, TOP, COLS, T>,
            MatrixView<'data, BOTTOM, COLS, T>,
        ),
        LinalgError,
    > {
        if TOP.checked_add(BOTTOM) != Some(ROWS) {
            return Err(LinalgError::OutOfBounds);
        }
        Ok((
            self.try_submatrix::<TOP, COLS>(0, 0)?,
            self.try_submatrix::<BOTTOM, COLS>(TOP, 0)?,
        ))
    }

    /// The first `LEFT` columns and the remaining `RIGHT`, or `OutOfBounds` unless they sum to
    /// `COLS`. Any layout does.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let (left, right) = matrix.view().try_split_cols_at::<1, 2>().unwrap();
    /// assert_eq!(left.to_matrix().into_array(), [[1.0], [4.0]]);
    /// assert_eq!(right.to_matrix().into_array(), [[2.0, 3.0], [5.0, 6.0]]);
    /// ```
    #[inline]
    pub fn try_split_cols_at<const LEFT: usize, const RIGHT: usize>(
        self,
    ) -> Result<
        (
            MatrixView<'data, ROWS, LEFT, T>,
            MatrixView<'data, ROWS, RIGHT, T>,
        ),
        LinalgError,
    > {
        if LEFT.checked_add(RIGHT) != Some(COLS) {
            return Err(LinalgError::OutOfBounds);
        }
        Ok((
            self.try_submatrix::<ROWS, LEFT>(0, 0)?,
            self.try_submatrix::<ROWS, RIGHT>(0, LEFT)?,
        ))
    }
}

impl<'data, const ROWS: usize, const COLS: usize, T: Copy> MatrixView<'data, ROWS, COLS, T> {
    /// Copies the window out. The only operation here that moves elements.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let owned = matrix.view().transposed().to_matrix();
    /// assert_eq!(owned.into_array(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    /// ```
    // In-bounds by construction: `from_fn` only asks for subscripts below `ROWS` and `COLS`, and
    // `required_len` already proved those land inside `data`.
    #[inline]
    #[allow(clippy::indexing_slicing)]
    pub fn to_matrix(self) -> Matrix<ROWS, COLS, T> {
        Matrix::from_fn(|row, column| {
            self.data[self.offset + row * self.row_stride + column * self.col_stride]
        })
    }
}

impl<'data, const ROWS: usize, const COLS: usize, T: PartialEq> PartialEq
    for MatrixView<'data, ROWS, COLS, T>
{
    /// Element by element, so different layouts over different buffers can be equal.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (0..ROWS).all(|row| {
            (0..COLS).all(|column| self.try_get(row, column) == other.try_get(row, column))
        })
    }
}

impl<'data, const ROWS: usize, const COLS: usize, T> MatrixViewMut<'data, ROWS, COLS, T> {
    /// Every constructor funnels through here, so an existing view is always in range.
    #[inline]
    pub(super) fn from_parts(
        data: &'data mut [T],
        offset: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Result<Self, LinalgError> {
        let needed = required_len(ROWS, COLS, offset, row_stride, col_stride)
            .ok_or(LinalgError::OutOfBounds)?;
        if needed > data.len() {
            return Err(LinalgError::OutOfBounds);
        }
        Ok(MatrixViewMut {
            data,
            offset,
            row_stride,
            col_stride,
        })
    }

    /// Views a row-major slice writably. See [`MatrixView::try_from_row_major_slice`].
    ///
    /// ```
    /// use multicalc::linear_algebra::MatrixViewMut;
    /// let mut buffer = [0.0; 6];
    /// let mut view = MatrixViewMut::<2, 3>::try_from_row_major_slice(&mut buffer).unwrap();
    /// *view.try_get_mut(1, 2).unwrap() = 7.0;
    /// assert_eq!(buffer, [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]);
    /// ```
    #[inline]
    pub fn try_from_row_major_slice(slice: &'data mut [T]) -> Result<Self, LinalgError> {
        Self::from_parts(slice, 0, COLS, 1)
    }

    /// The row count, `ROWS`.
    #[inline]
    #[must_use]
    pub const fn rows(&self) -> usize {
        ROWS
    }

    /// The column count, `COLS`.
    #[inline]
    #[must_use]
    pub const fn cols(&self) -> usize {
        COLS
    }

    /// See [`MatrixView::strides`].
    #[inline]
    #[must_use]
    pub const fn strides(&self) -> (usize, usize) {
        (self.row_stride, self.col_stride)
    }

    /// Whether the rows are unpadded runs, which the row split below needs.
    #[inline]
    #[must_use]
    pub const fn is_row_major(&self) -> bool {
        self.row_stride >= COLS && (COLS <= 1 || self.col_stride == 1)
    }

    #[inline]
    fn index_of(&self, row: usize, column: usize) -> Option<usize> {
        (row < ROWS && column < COLS).then_some(())?;
        self.offset
            .checked_add(row.checked_mul(self.row_stride)?)?
            .checked_add(column.checked_mul(self.col_stride)?)
    }

    #[inline]
    fn block_offset(
        &self,
        top: usize,
        left: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> Option<usize> {
        (top.checked_add(block_rows)? <= ROWS && left.checked_add(block_cols)? <= COLS)
            .then_some(())?;
        self.offset
            .checked_add(top.checked_mul(self.row_stride)?)?
            .checked_add(left.checked_mul(self.col_stride)?)
    }

    /// Entry `(row, column)`, or `OutOfBounds`. See [`MatrixView::try_get`].
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let view = matrix.view_mut();
    /// assert_eq!(view.try_get(1, 0), Ok(&3.0));
    /// assert_eq!(view.try_get(2, 0), Err(LinalgError::OutOfBounds));
    /// ```
    #[inline]
    pub fn try_get(&self, row: usize, column: usize) -> Result<&T, LinalgError> {
        self.index_of(row, column)
            .and_then(|flat| self.data.get(flat))
            .ok_or(LinalgError::OutOfBounds)
    }

    /// Entry `(row, column)`, writably, or `OutOfBounds`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// *matrix.view_mut().try_get_mut(0, 1).unwrap() = 9.0;
    /// assert_eq!(matrix.into_array(), [[1.0, 9.0], [3.0, 4.0]]);
    /// ```
    #[inline]
    pub fn try_get_mut(&mut self, row: usize, column: usize) -> Result<&mut T, LinalgError> {
        let flat = self.index_of(row, column).ok_or(LinalgError::OutOfBounds)?;
        self.data.get_mut(flat).ok_or(LinalgError::OutOfBounds)
    }

    /// Borrows read-only for as long as `self` is untouched.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let mut view = matrix.view_mut();
    /// assert_eq!(view.as_view().transposed().try_get(0, 1), Ok(&3.0));
    /// *view.try_get_mut(0, 0).unwrap() = 9.0;
    /// assert_eq!(matrix.into_array(), [[9.0, 2.0], [3.0, 4.0]]);
    /// ```
    #[inline]
    pub fn as_view(&self) -> MatrixView<'_, ROWS, COLS, T> {
        MatrixView {
            data: self.data,
            offset: self.offset,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
        }
    }

    /// Borrows writably for a shorter lifetime, so the original survives a consuming method.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<2, 2>::zeros();
    /// let mut view = matrix.view_mut();
    /// *view.reborrow().transposed().try_get_mut(0, 1).unwrap() = 5.0;
    /// *view.try_get_mut(0, 0).unwrap() = 1.0;
    /// assert_eq!(matrix.into_array(), [[1.0, 0.0], [5.0, 0.0]]);
    /// ```
    #[inline]
    pub fn reborrow(&mut self) -> MatrixViewMut<'_, ROWS, COLS, T> {
        MatrixViewMut {
            data: self.data,
            offset: self.offset,
            row_stride: self.row_stride,
            col_stride: self.col_stride,
        }
    }

    /// The transpose. Consumes the view; [`reborrow`](Self::reborrow) first to keep it.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<2, 3>::zeros();
    /// *matrix.view_mut().transposed().try_get_mut(2, 1).unwrap() = 9.0;
    /// assert_eq!(matrix[(1, 2)], 9.0);
    /// ```
    #[inline]
    pub fn transposed(self) -> MatrixViewMut<'data, COLS, ROWS, T> {
        MatrixViewMut {
            data: self.data,
            offset: self.offset,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    /// The writable block at `(top, left)`. See [`MatrixView::try_submatrix`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let identity = Matrix::<2, 2>::identity();
    /// let mut matrix = Matrix::<3, 3>::zeros();
    /// matrix.view_mut().try_submatrix::<2, 2>(1, 1).unwrap().copy_from(identity.view());
    /// assert_eq!(matrix.into_array(), [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// ```
    #[inline]
    pub fn try_submatrix<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
        self,
        top: usize,
        left: usize,
    ) -> Result<MatrixViewMut<'data, BLOCK_ROWS, BLOCK_COLS, T>, LinalgError> {
        let offset = self
            .block_offset(top, left, BLOCK_ROWS, BLOCK_COLS)
            .ok_or(LinalgError::OutOfBounds)?;
        MatrixViewMut::from_parts(self.data, offset, self.row_stride, self.col_stride)
    }

    /// Row `row`, writably. See [`MatrixView::try_row`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// matrix.view_mut().try_row(1).unwrap().fill(0.0);
    /// assert_eq!(matrix.into_array(), [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]);
    /// ```
    #[inline]
    pub fn try_row(self, row: usize) -> Result<VectorViewMut<'data, COLS, T>, LinalgError> {
        let offset = self
            .block_offset(row, 0, 1, COLS)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorViewMut::from_parts(self.data, offset, self.col_stride)
    }

    /// Column `column`, writably. See [`MatrixView::try_column`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// matrix.view_mut().try_column(2).unwrap().fill(0.0);
    /// assert_eq!(matrix.into_array(), [[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]]);
    /// ```
    #[inline]
    pub fn try_column(self, column: usize) -> Result<VectorViewMut<'data, ROWS, T>, LinalgError> {
        let offset = self
            .block_offset(0, column, ROWS, 1)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorViewMut::from_parts(self.data, offset, self.row_stride)
    }

    /// The main diagonal, writably. See [`MatrixView::try_diagonal`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<3, 3>::zeros();
    /// matrix.view_mut().try_diagonal::<3>().unwrap().fill(1.0);
    /// assert_eq!(matrix, Matrix::<3, 3>::identity());
    /// ```
    #[inline]
    pub fn try_diagonal<const LEN: usize>(
        self,
    ) -> Result<VectorViewMut<'data, LEN, T>, LinalgError> {
        let stride = diagonal_stride(LEN, ROWS, COLS, self.row_stride, self.col_stride)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorViewMut::from_parts(self.data, self.offset, stride)
    }

    /// Two halves that can be written through at once. Needs `TOP + BOTTOM == ROWS` and a
    /// [row-major](Self::is_row_major) view, since a transposed one interleaves its rows and no
    /// cut of the slice separates them. [`MatrixView::try_split_rows_at`] has no such
    /// requirement.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<3, 2>::zeros();
    /// let (mut top, mut bottom) = matrix.view_mut().try_split_rows_at::<1, 2>().unwrap();
    /// *top.try_get_mut(0, 0).unwrap() = 1.0;
    /// *bottom.try_get_mut(1, 1).unwrap() = 2.0;
    /// assert_eq!(matrix.into_array(), [[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]);
    /// ```
    #[inline]
    pub fn try_split_rows_at<const TOP: usize, const BOTTOM: usize>(
        self,
    ) -> Result<
        (
            MatrixViewMut<'data, TOP, COLS, T>,
            MatrixViewMut<'data, BOTTOM, COLS, T>,
        ),
        LinalgError,
    > {
        if TOP.checked_add(BOTTOM) != Some(ROWS) || !self.is_row_major() {
            return Err(LinalgError::OutOfBounds);
        }
        let split = TOP
            .checked_mul(self.row_stride)
            .and_then(|rows| self.offset.checked_add(rows))
            .ok_or(LinalgError::OutOfBounds)?;
        if split > self.data.len() {
            return Err(LinalgError::OutOfBounds);
        }
        let (head, tail) = self.data.split_at_mut(split);
        Ok((
            MatrixViewMut::from_parts(head, self.offset, self.row_stride, self.col_stride)?,
            MatrixViewMut::from_parts(tail, 0, self.row_stride, self.col_stride)?,
        ))
    }

    /// The column split, which is the row split seen through a transpose: it needs
    /// `LEFT + RIGHT == COLS` and a *column*-major view, since row-major storage interleaves its
    /// columns. `transposed()` on an ordinary view produces one.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<3, 2>::zeros();
    /// let (mut left, mut right) =
    ///     matrix.view_mut().transposed().try_split_cols_at::<1, 2>().unwrap();
    /// *left.try_get_mut(0, 0).unwrap() = 1.0;
    /// *right.try_get_mut(1, 1).unwrap() = 2.0;
    /// assert_eq!(matrix.into_array(), [[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]);
    /// ```
    #[inline]
    pub fn try_split_cols_at<const LEFT: usize, const RIGHT: usize>(
        self,
    ) -> Result<
        (
            MatrixViewMut<'data, ROWS, LEFT, T>,
            MatrixViewMut<'data, ROWS, RIGHT, T>,
        ),
        LinalgError,
    > {
        let (left, right) = self.transposed().try_split_rows_at::<LEFT, RIGHT>()?;
        Ok((left.transposed(), right.transposed()))
    }
}

impl<'data, const ROWS: usize, const COLS: usize, T: Copy> MatrixViewMut<'data, ROWS, COLS, T> {
    /// Copies the window out.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.view_mut().to_matrix().into_array(), [[1.0, 2.0], [3.0, 4.0]]);
    /// ```
    #[inline]
    pub fn to_matrix(&self) -> Matrix<ROWS, COLS, T> {
        self.as_view().to_matrix()
    }

    /// Overwrites every entry with `value`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<2, 2>::zeros();
    /// matrix.view_mut().try_submatrix::<1, 2>(1, 0).unwrap().fill(7.0);
    /// assert_eq!(matrix.into_array(), [[0.0, 0.0], [7.0, 7.0]]);
    /// ```
    #[inline]
    pub fn fill(&mut self, value: T) {
        for row in 0..ROWS {
            for column in 0..COLS {
                if let Ok(slot) = self.try_get_mut(row, column) {
                    *slot = value;
                }
            }
        }
    }

    /// Copies `source` in element by element; the layouts may differ, which is how a transpose
    /// lands in a caller's buffer without an intermediate.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, MatrixViewMut};
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let mut scratch = [0.0; 6];
    /// let mut destination = MatrixViewMut::<3, 2>::try_from_row_major_slice(&mut scratch).unwrap();
    /// destination.copy_from(matrix.view().transposed());
    /// assert_eq!(scratch, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    #[inline]
    pub fn copy_from(&mut self, source: MatrixView<'_, ROWS, COLS, T>) {
        for row in 0..ROWS {
            for column in 0..COLS {
                if let (Ok(value), Ok(slot)) = (
                    source.try_get(row, column).copied(),
                    self.try_get_mut(row, column),
                ) {
                    *slot = value;
                }
            }
        }
    }
}

impl<'data, const ROWS: usize, const COLS: usize, T: PartialEq> PartialEq
    for MatrixViewMut<'data, ROWS, COLS, T>
{
    /// Element by element, matching [`MatrixView`].
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_view() == other.as_view()
    }
}

// `None` if `len` is not the shorter side, or the strides overflow when added.
#[inline]
fn diagonal_stride(
    len: usize,
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
) -> Option<usize> {
    let shorter_side = if rows < cols { rows } else { cols };
    (len == shorter_side).then_some(())?;
    row_stride.checked_add(col_stride)
}

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> {
    /// A read-only view of the whole matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.view().try_get(0, 1), Ok(&2.0));
    /// ```
    #[inline]
    pub fn view(&self) -> MatrixView<'_, ROWS, COLS, T> {
        MatrixView {
            data: self.as_slice_rows().as_flattened(),
            offset: 0,
            row_stride: COLS,
            col_stride: 1,
        }
    }

    /// A writable view of the whole matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<2, 2>::zeros();
    /// *matrix.view_mut().transposed().try_get_mut(0, 1).unwrap() = 5.0;
    /// assert_eq!(matrix.into_array(), [[0.0, 0.0], [5.0, 0.0]]);
    /// ```
    #[inline]
    pub fn view_mut(&mut self) -> MatrixViewMut<'_, ROWS, COLS, T> {
        MatrixViewMut {
            data: self.as_mut_slice_rows().as_flattened_mut(),
            offset: 0,
            row_stride: COLS,
            col_stride: 1,
        }
    }
}

impl<const ROWS: usize, const COLS: usize, T: Copy> Matrix<ROWS, COLS, T> {
    /// Copies out the block at `(top, left)`, or `OutOfBounds`. This is
    /// [`MatrixView::try_submatrix`] plus one copy; take the view directly to skip it.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    /// let block = matrix.try_submatrix::<2, 2>(1, 1).unwrap();
    /// assert_eq!(block.into_array(), [[5.0, 6.0], [8.0, 9.0]]);
    /// assert!(matrix.try_submatrix::<2, 2>(2, 0).is_err());
    /// ```
    #[inline]
    pub fn try_submatrix<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
        &self,
        top: usize,
        left: usize,
    ) -> Result<Matrix<BLOCK_ROWS, BLOCK_COLS, T>, LinalgError> {
        Ok(self
            .view()
            .try_submatrix::<BLOCK_ROWS, BLOCK_COLS>(top, left)?
            .to_matrix())
    }

    /// Writes `block` in at `(top, left)`, or returns `OutOfBounds` and leaves the matrix
    /// untouched.
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<3, 3>::zeros();
    /// matrix.try_set_submatrix(1, 1, Matrix::<2, 2>::identity().view()).unwrap();
    /// assert_eq!(matrix.into_array()[1], [0.0, 1.0, 0.0]);
    /// let overhanging = Matrix::<2, 2>::identity();
    /// assert_eq!(
    ///     matrix.try_set_submatrix(2, 2, overhanging.view()),
    ///     Err(LinalgError::OutOfBounds)
    /// );
    /// ```
    #[inline]
    pub fn try_set_submatrix<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
        &mut self,
        top: usize,
        left: usize,
        block: MatrixView<'_, BLOCK_ROWS, BLOCK_COLS, T>,
    ) -> Result<(), LinalgError> {
        self.view_mut()
            .try_submatrix::<BLOCK_ROWS, BLOCK_COLS>(top, left)?
            .copy_from(block);
        Ok(())
    }
}
