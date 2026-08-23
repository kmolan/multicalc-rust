//! The borrowed matrix views, and the [`Matrix`] methods that hand them out.

use super::{VectorView, VectorViewMut, required_len};
use crate::error::LinalgError;
use crate::linear_algebra::Matrix;

/// A borrowed, strided, read-only `ROWS`×`COLS` window onto someone else's storage.
///
/// Reshaping is free: [`transposed`](Self::transposed) and [`submatrix`](Self::submatrix) only
/// rewrite the offset and strides, so neither touches an element.
///
/// Every fallible operation reports [`LinalgError::OutOfBounds`] and nothing else, so a caller
/// that only needs to know whether a window fits can test with `is_ok`. There is no `Index`
/// impl: a subscript that misses has to be answerable, and `Index` returns `&T` with nowhere to
/// put an error, so [`get`](Self::get) is the only way in.
///
/// ```
/// use multicalc::linear_algebra::Matrix;
///
/// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
/// let corner = matrix.view().submatrix::<2, 2>(1, 1).unwrap();
///
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
/// The reshaping methods consume the view, because the exclusive borrow has to move with them.
/// [`as_view`](Self::as_view) and [`reborrow`](Self::reborrow) hand out shorter-lived views when
/// the original needs to stay usable.
///
/// The surface mirrors [`MatrixView`] method for method, with [`get_mut`](Self::get_mut),
/// [`fill`](Self::fill), and [`copy_from`](Self::copy_from) added for the writable side.
///
/// ```
/// use multicalc::linear_algebra::Matrix;
///
/// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
/// let mut transposed = matrix.view_mut().transposed();
/// *transposed.get_mut(0, 1).unwrap() = 9.0; // writes through to matrix[(1, 0)]
///
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

// `Clone` and `Copy` are written out rather than derived, because a derive would demand
// `T: Copy`. What is copied is the handle -- the slice reference, offset, and two strides -- so
// it is `Copy` for the same reason `&[T]` is, whatever `T` turns out to be. The elements are
// never touched; `to_matrix` is the only thing that copies those.
impl<'data, const ROWS: usize, const COLS: usize, T> Clone for MatrixView<'data, ROWS, COLS, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<'data, const ROWS: usize, const COLS: usize, T> Copy for MatrixView<'data, ROWS, COLS, T> {}

impl<'data, const ROWS: usize, const COLS: usize, T> MatrixView<'data, ROWS, COLS, T> {
    /// Builds a view over `data`, or [`LinalgError::OutOfBounds`] if the shape would reach past
    /// the end of the slice. Every constructor funnels through here, so an existing view is
    /// always in range and `get` can never be defeated by a bad stride.
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

    /// Views a row-major slice as a matrix, or [`LinalgError::OutOfBounds`] if it holds fewer
    /// than `ROWS * COLS` elements. Trailing elements are ignored, which is what makes a scratch
    /// buffer reusable.
    ///
    /// ```
    /// use multicalc::linear_algebra::MatrixView;
    /// let buffer = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let view = MatrixView::<2, 2>::from_row_major_slice(&buffer).unwrap();
    /// assert_eq!(view.get(1, 0), Ok(&3.0));
    /// assert!(MatrixView::<3, 2>::from_row_major_slice(&buffer).is_err());
    /// ```
    #[inline]
    pub fn from_row_major_slice(slice: &'data [T]) -> Result<Self, LinalgError> {
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

    /// The `(row_stride, column_stride)` pair: how far apart consecutive rows and columns sit in
    /// the underlying buffer. A freshly taken view of a [`Matrix`] reads `(COLS, 1)`; transposing
    /// it reads `(1, COLS)`.
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

    /// Whether the elements are laid out contiguously in row-major order, which is the layout
    /// [`MatrixViewMut::split_rows_at`] needs.
    #[inline]
    #[must_use]
    pub const fn is_row_major(&self) -> bool {
        // With one column or none, `col_stride` addresses nothing and any row stride wide
        // enough to keep the rows apart is separable; otherwise the rows must be unpadded runs.
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

    /// Returns a reference to entry `(row, column)`, or [`LinalgError::OutOfBounds`] if the
    /// subscript misses.
    ///
    /// This is the only accessor. A strided view cannot leave the bound to the slice the way an
    /// owned [`Matrix`] leaves it to its arrays: in a 2×2 block taken from a 3×3 matrix (strides
    /// 3 and 1) the invalid subscript `(2, 0)` works out to flat index 6, a perfectly valid
    /// element of the 9-element parent buffer. Deferring to the slice bound would hand back the
    /// wrong entry instead of failing, so the shape is compared here.
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    ///
    /// assert_eq!(matrix.view().get(1, 0), Ok(&3.0));
    /// assert_eq!(matrix.view().get(2, 0), Err(LinalgError::OutOfBounds));
    /// ```
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Result<&T, LinalgError> {
        self.index_of(row, column)
            .and_then(|flat| self.data.get(flat))
            .ok_or(LinalgError::OutOfBounds)
    }

    /// The transpose, in constant time: the two strides trade places and nothing is copied.
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

    /// The `BLOCK_ROWS`×`BLOCK_COLS` block whose top-left corner sits at `(top, left)`, or
    /// [`LinalgError::OutOfBounds`] if that block would run past an edge.
    ///
    /// `top` and `left` say *where* the block starts, counted in the current view's rows and
    /// columns. The two const parameters say *how big* it is. The size is a const parameter
    /// rather than an argument because it is part of the returned type — a 2×2 block and a 3×3
    /// block are different types, so the shape has to be known at compile time, while the corner
    /// is free to be a runtime value.
    ///
    /// Only the offset moves, which is what makes this constant time. Stepping down one row means
    /// moving `row_stride` elements along the buffer and stepping right one column means moving
    /// `col_stride`, so the corner sits `top * row_stride + left * col_stride` elements past
    /// wherever the view already started. The strides carry over unchanged, because the block
    /// looks at the same buffer: its rows are still as far apart as the parent's were.
    ///
    /// In a 3×3 row-major matrix (`row_stride` 3, `col_stride` 1) the block at `(1, 1)` starts
    /// `1 * 3 + 1 * 1 = 4` elements in, which is entry `(1, 1)`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let block = matrix.view().submatrix::<2, 2>(0, 1).unwrap();
    /// assert_eq!(block.to_matrix().into_array(), [[2.0, 3.0], [5.0, 6.0]]);
    /// assert!(matrix.view().submatrix::<2, 2>(0, 2).is_err());
    /// ```
    #[inline]
    pub fn submatrix<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
        self,
        top: usize,
        left: usize,
    ) -> Result<MatrixView<'data, BLOCK_ROWS, BLOCK_COLS, T>, LinalgError> {
        let offset = self
            .block_offset(top, left, BLOCK_ROWS, BLOCK_COLS)
            .ok_or(LinalgError::OutOfBounds)?;
        MatrixView::from_parts(self.data, offset, self.row_stride, self.col_stride)
    }

    /// Row `row` as a view, or [`LinalgError::OutOfBounds`] if `row >= ROWS`. Unlike
    /// [`Matrix::try_row`], which copies the entries into an owned vector, this only works out
    /// where the row starts.
    ///
    /// The result has `COLS` components, because a row spans every column. Its stride is the
    /// view's `col_stride` — the gap between neighbouring entries *within* a row.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.view().row(1).unwrap().to_vector().into_array(), [3.0, 4.0]);
    /// ```
    #[inline]
    pub fn row(self, row: usize) -> Result<VectorView<'data, COLS, T>, LinalgError> {
        let offset = self
            .block_offset(row, 0, 1, COLS)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorView::from_parts(self.data, offset, self.col_stride)
    }

    /// Column `column` as a view, or [`LinalgError::OutOfBounds`] if `column >= COLS`. Unlike
    /// [`Matrix::try_column`], which copies, this only works out where the column starts.
    ///
    /// The result has `ROWS` components, because a column spans every row. Its stride is the
    /// view's `row_stride`: consecutive entries of a column are one whole row apart in the
    /// buffer, and carrying that stride around instead of gathering the entries is what saves
    /// the copy.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.view().column(1).unwrap().to_vector().into_array(), [2.0, 4.0]);
    /// ```
    #[inline]
    pub fn column(self, column: usize) -> Result<VectorView<'data, ROWS, T>, LinalgError> {
        let offset = self
            .block_offset(0, column, ROWS, 1)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorView::from_parts(self.data, offset, self.row_stride)
    }

    /// The main diagonal — entries `(0, 0)`, `(1, 1)`, `(2, 2)` and so on — as a view, without
    /// gathering them into a vector.
    ///
    /// A diagonal stops at whichever edge it reaches first, so it holds `min(ROWS, COLS)`
    /// entries. Rust cannot yet work that out inside a type, so the length arrives as `LEN` and
    /// is checked instead: pass the shorter side, or get [`LinalgError::OutOfBounds`] back. For a
    /// 2×4 view that is `2`.
    ///
    /// The stride is `row_stride + col_stride`, because one step along the diagonal moves down a
    /// row *and* right a column. On a 3×3 row-major matrix that is `3 + 1 = 4`, landing on buffer
    /// positions 0, 4 and 8 — the diagonal.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let diagonal = matrix.view().diagonal::<2>().unwrap();
    /// assert_eq!(diagonal.to_vector().into_array(), [1.0, 5.0]);
    /// ```
    #[inline]
    pub fn diagonal<const LEN: usize>(self) -> Result<VectorView<'data, LEN, T>, LinalgError> {
        let stride = diagonal_stride(LEN, ROWS, COLS, self.row_stride, self.col_stride)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorView::from_parts(self.data, self.offset, stride)
    }

    /// Splits into the first `TOP` rows and the remaining `BOTTOM`, or
    /// [`LinalgError::OutOfBounds`] unless `TOP + BOTTOM == ROWS`.
    ///
    /// Both halves keep looking at the whole buffer and differ only in offset and shape, so
    /// unlike [`MatrixViewMut::split_rows_at`] this works on any layout — a read-only split has
    /// no disjointness to prove.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    /// let (top, bottom) = matrix.view().split_rows_at::<1, 2>().unwrap();
    ///
    /// assert_eq!(top.to_matrix().into_array(), [[1.0, 2.0]]);
    /// assert_eq!(bottom.to_matrix().into_array(), [[3.0, 4.0], [5.0, 6.0]]);
    /// ```
    #[inline]
    pub fn split_rows_at<const TOP: usize, const BOTTOM: usize>(
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
            self.submatrix::<TOP, COLS>(0, 0)?,
            self.submatrix::<BOTTOM, COLS>(TOP, 0)?,
        ))
    }

    /// Splits into the first `LEFT` columns and the remaining `RIGHT`, or
    /// [`LinalgError::OutOfBounds`] unless `LEFT + RIGHT == COLS`. The column counterpart of
    /// [`split_rows_at`](Self::split_rows_at), and likewise free of any layout requirement.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let (left, right) = matrix.view().split_cols_at::<1, 2>().unwrap();
    ///
    /// assert_eq!(left.to_matrix().into_array(), [[1.0], [4.0]]);
    /// assert_eq!(right.to_matrix().into_array(), [[2.0, 3.0], [5.0, 6.0]]);
    /// ```
    #[inline]
    pub fn split_cols_at<const LEFT: usize, const RIGHT: usize>(
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
            self.submatrix::<ROWS, LEFT>(0, 0)?,
            self.submatrix::<ROWS, RIGHT>(0, LEFT)?,
        ))
    }
}

impl<'data, const ROWS: usize, const COLS: usize, T: Copy> MatrixView<'data, ROWS, COLS, T> {
    /// Copies the window into an owned matrix. This is the one operation here that moves elements.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let owned = matrix.view().transposed().to_matrix();
    ///
    /// assert_eq!(owned.into_array(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    /// ```
    // `Matrix::from_fn` only ever asks for subscripts below `ROWS` and `COLS`, and every
    // constructor has already proved that such a subscript lands inside `data` without
    // overflowing -- that is exactly what `required_len` checks. So the slice index below cannot
    // miss and needs no error channel, the same reasoning (and the same allow) as
    // `Matrix::get_unchecked`.
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
    /// Compares element by element, so two views of different layouts over different buffers are
    /// equal when they present the same entries.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (0..ROWS).all(|row| (0..COLS).all(|column| self.get(row, column) == other.get(row, column)))
    }
}

impl<'data, const ROWS: usize, const COLS: usize, T> MatrixViewMut<'data, ROWS, COLS, T> {
    /// Builds a view over `data`, or [`LinalgError::OutOfBounds`] if the shape would reach past
    /// the end of the slice.
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

    /// Views a row-major slice as a writable matrix, or [`LinalgError::OutOfBounds`] if it holds
    /// fewer than `ROWS * COLS` elements.
    ///
    /// ```
    /// use multicalc::linear_algebra::MatrixViewMut;
    /// let mut buffer = [0.0; 6];
    /// let mut view = MatrixViewMut::<2, 3>::from_row_major_slice(&mut buffer).unwrap();
    /// *view.get_mut(1, 2).unwrap() = 7.0;
    /// assert_eq!(buffer, [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]);
    /// ```
    #[inline]
    pub fn from_row_major_slice(slice: &'data mut [T]) -> Result<Self, LinalgError> {
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

    /// The `(row_stride, column_stride)` pair. See [`MatrixView::strides`].
    #[inline]
    #[must_use]
    pub const fn strides(&self) -> (usize, usize) {
        (self.row_stride, self.col_stride)
    }

    /// Whether the elements are laid out contiguously in row-major order. The row splits below
    /// require this, since they cut the underlying slice in two.
    #[inline]
    #[must_use]
    pub const fn is_row_major(&self) -> bool {
        // With one column or none, `col_stride` addresses nothing and any row stride wide
        // enough to keep the rows apart is separable; otherwise the rows must be unpadded runs.
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

    /// Returns a reference to entry `(row, column)`, or [`LinalgError::OutOfBounds`] if the
    /// subscript misses. See [`MatrixView::get`] for why the bound cannot be left to the slice.
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let view = matrix.view_mut();
    ///
    /// assert_eq!(view.get(1, 0), Ok(&3.0));
    /// assert_eq!(view.get(2, 0), Err(LinalgError::OutOfBounds));
    /// ```
    #[inline]
    pub fn get(&self, row: usize, column: usize) -> Result<&T, LinalgError> {
        self.index_of(row, column)
            .and_then(|flat| self.data.get(flat))
            .ok_or(LinalgError::OutOfBounds)
    }

    /// Returns a mutable reference to entry `(row, column)`, or [`LinalgError::OutOfBounds`] if
    /// the subscript misses.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let mut view = matrix.view_mut();
    /// *view.get_mut(0, 1).unwrap() = 9.0;
    ///
    /// assert_eq!(matrix.into_array(), [[1.0, 9.0], [3.0, 4.0]]);
    /// ```
    #[inline]
    pub fn get_mut(&mut self, row: usize, column: usize) -> Result<&mut T, LinalgError> {
        let flat = self.index_of(row, column).ok_or(LinalgError::OutOfBounds)?;
        self.data.get_mut(flat).ok_or(LinalgError::OutOfBounds)
    }

    /// Borrows this window read-only for as long as `self` is untouched.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let mut view = matrix.view_mut();
    ///
    /// // `transposed` would consume `view`; `as_view` leaves it usable.
    /// assert_eq!(view.as_view().transposed().get(0, 1), Ok(&3.0));
    /// *view.get_mut(0, 0).unwrap() = 9.0;
    ///
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

    /// Borrows this window writably for a shorter lifetime, so the original stays usable after
    /// a method that would otherwise consume it.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<2, 2>::zeros();
    /// let mut view = matrix.view_mut();
    /// *view.reborrow().transposed().get_mut(0, 1).unwrap() = 5.0;
    /// *view.get_mut(0, 0).unwrap() = 1.0;
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

    /// The transpose, in constant time. Consumes the view because the exclusive borrow moves into
    /// the result; use [`reborrow`](Self::reborrow) first to keep the original.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<2, 3>::zeros();
    /// *matrix.view_mut().transposed().get_mut(2, 1).unwrap() = 9.0;
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

    /// The `BLOCK_ROWS`×`BLOCK_COLS` writable block whose top-left corner sits at `(top, left)`,
    /// or [`LinalgError::OutOfBounds`] if that block would run past an edge. The read-only
    /// [`MatrixView::submatrix`] explains the coordinates and the stride arithmetic; this is the
    /// same operation on a writable view.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let identity = Matrix::<2, 2>::identity();
    /// let mut matrix = Matrix::<3, 3>::zeros();
    ///
    /// matrix.view_mut().submatrix::<2, 2>(1, 1).unwrap().copy_from(identity.view());
    ///
    /// assert_eq!(matrix.into_array(), [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// ```
    #[inline]
    pub fn submatrix<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
        self,
        top: usize,
        left: usize,
    ) -> Result<MatrixViewMut<'data, BLOCK_ROWS, BLOCK_COLS, T>, LinalgError> {
        let offset = self
            .block_offset(top, left, BLOCK_ROWS, BLOCK_COLS)
            .ok_or(LinalgError::OutOfBounds)?;
        MatrixViewMut::from_parts(self.data, offset, self.row_stride, self.col_stride)
    }

    /// Row `row` as a writable view of `COLS` components, or [`LinalgError::OutOfBounds`] if
    /// `row >= ROWS`. See [`MatrixView::row`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let mut matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// matrix.view_mut().row(1).unwrap().fill(0.0);
    ///
    /// assert_eq!(matrix.into_array(), [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]);
    /// ```
    #[inline]
    pub fn row(self, row: usize) -> Result<VectorViewMut<'data, COLS, T>, LinalgError> {
        let offset = self
            .block_offset(row, 0, 1, COLS)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorViewMut::from_parts(self.data, offset, self.col_stride)
    }

    /// Column `column` as a writable view of `ROWS` components, or [`LinalgError::OutOfBounds`]
    /// if `column >= COLS`. See [`MatrixView::column`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let mut matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// matrix.view_mut().column(2).unwrap().fill(0.0);
    ///
    /// assert_eq!(matrix.into_array(), [[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]]);
    /// ```
    #[inline]
    pub fn column(self, column: usize) -> Result<VectorViewMut<'data, ROWS, T>, LinalgError> {
        let offset = self
            .block_offset(0, column, ROWS, 1)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorViewMut::from_parts(self.data, offset, self.row_stride)
    }

    /// The main diagonal as a writable view. The read-only [`MatrixView::diagonal`] explains the
    /// length rule and the stride; this is the same operation on a writable view.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let mut matrix = Matrix::<3, 3>::zeros();
    /// matrix.view_mut().diagonal::<3>().unwrap().fill(1.0);
    ///
    /// assert_eq!(matrix, Matrix::<3, 3>::identity());
    /// ```
    #[inline]
    pub fn diagonal<const LEN: usize>(self) -> Result<VectorViewMut<'data, LEN, T>, LinalgError> {
        let stride = diagonal_stride(LEN, ROWS, COLS, self.row_stride, self.col_stride)
            .ok_or(LinalgError::OutOfBounds)?;
        VectorViewMut::from_parts(self.data, self.offset, stride)
    }

    /// Splits into the first `TOP` rows and the remaining `BOTTOM`, as two views that can be
    /// written through at the same time.
    ///
    /// Returns [`LinalgError::OutOfBounds`] unless `TOP + BOTTOM == ROWS` and the view
    /// [is row-major](Self::is_row_major) — a transposed view interleaves its rows in the buffer,
    /// so no cut of the slice separates them. The read-only
    /// [`MatrixView::split_rows_at`] has no such requirement, because two shared views are
    /// allowed to overlap.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<3, 2>::zeros();
    /// let (mut top, mut bottom) = matrix.view_mut().split_rows_at::<1, 2>().unwrap();
    /// *top.get_mut(0, 0).unwrap() = 1.0;
    /// *bottom.get_mut(1, 1).unwrap() = 2.0;
    /// assert_eq!(matrix.into_array(), [[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]);
    /// ```
    #[inline]
    pub fn split_rows_at<const TOP: usize, const BOTTOM: usize>(
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

    /// Splits into the first `LEFT` columns and the remaining `RIGHT`, as two views that can be
    /// written through at the same time.
    ///
    /// This is [`split_rows_at`](Self::split_rows_at) seen through a transpose, so it carries the
    /// mirrored requirement: `LEFT + RIGHT == COLS` and a *column*-major view, since in
    /// row-major storage the columns interleave and no cut of the slice separates them. A
    /// row-major view transposed is column-major, which is where such a view usually comes from.
    /// The read-only [`MatrixView::split_cols_at`] has no layout requirement.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut matrix = Matrix::<3, 2>::zeros();
    ///
    /// // `transposed()` turns the row-major view into the column-major one this needs.
    /// let (mut left, mut right) =
    ///     matrix.view_mut().transposed().split_cols_at::<1, 2>().unwrap();
    /// *left.get_mut(0, 0).unwrap() = 1.0;
    /// *right.get_mut(1, 1).unwrap() = 2.0;
    ///
    /// assert_eq!(matrix.into_array(), [[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]);
    /// ```
    #[inline]
    pub fn split_cols_at<const LEFT: usize, const RIGHT: usize>(
        self,
    ) -> Result<
        (
            MatrixViewMut<'data, ROWS, LEFT, T>,
            MatrixViewMut<'data, ROWS, RIGHT, T>,
        ),
        LinalgError,
    > {
        let (left, right) = self.transposed().split_rows_at::<LEFT, RIGHT>()?;
        Ok((left.transposed(), right.transposed()))
    }
}

impl<'data, const ROWS: usize, const COLS: usize, T: Copy> MatrixViewMut<'data, ROWS, COLS, T> {
    /// Copies the window into an owned matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let mut matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// let view = matrix.view_mut();
    ///
    /// assert_eq!(view.to_matrix().into_array(), [[1.0, 2.0], [3.0, 4.0]]);
    /// ```
    #[inline]
    pub fn to_matrix(&self) -> Matrix<ROWS, COLS, T> {
        self.as_view().to_matrix()
    }

    /// Overwrites every entry with `value`.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let mut matrix = Matrix::<2, 2>::zeros();
    /// matrix.view_mut().submatrix::<1, 2>(1, 0).unwrap().fill(7.0);
    ///
    /// assert_eq!(matrix.into_array(), [[0.0, 0.0], [7.0, 7.0]]);
    /// ```
    #[inline]
    pub fn fill(&mut self, value: T) {
        for row in 0..ROWS {
            for column in 0..COLS {
                if let Ok(slot) = self.get_mut(row, column) {
                    *slot = value;
                }
            }
        }
    }

    /// Copies `source` in element by element. The two may have different layouts — writing a
    /// transposed view through here is how a transpose lands in a caller's scratch buffer without
    /// an intermediate stack matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, MatrixViewMut};
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let mut scratch = [0.0; 6];
    /// let mut destination = MatrixViewMut::<3, 2>::from_row_major_slice(&mut scratch).unwrap();
    /// destination.copy_from(matrix.view().transposed());
    /// assert_eq!(scratch, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    #[inline]
    pub fn copy_from(&mut self, source: MatrixView<'_, ROWS, COLS, T>) {
        for row in 0..ROWS {
            for column in 0..COLS {
                if let (Ok(value), Ok(slot)) =
                    (source.get(row, column).copied(), self.get_mut(row, column))
                {
                    *slot = value;
                }
            }
        }
    }
}

impl<'data, const ROWS: usize, const COLS: usize, T: PartialEq> PartialEq
    for MatrixViewMut<'data, ROWS, COLS, T>
{
    /// Compares element by element, matching [`MatrixView`]'s impl.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_view() == other.as_view()
    }
}

// The stride along the main diagonal, or `None` if `len` is not the shorter side or the strides
// overflow when added. Shared by the two views, which agree on the rule and differ only in the
// mutability of the slice they carry.
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
    /// A read-only view of the whole matrix. The entry point for the zero-copy reshaping in
    /// [`MatrixView`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(matrix.view().get(0, 1), Ok(&2.0));
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
    /// *matrix.view_mut().transposed().get_mut(0, 1).unwrap() = 5.0;
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
    /// Copies out the `BLOCK_ROWS`×`BLOCK_COLS` block whose top-left corner is `(top, left)`, or
    /// [`LinalgError::OutOfBounds`] if that block would run past an edge.
    ///
    /// This is [`MatrixView::submatrix`] followed by one copy. Take the view directly when the
    /// block only needs reading.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    /// assert_eq!(
    ///     matrix.submatrix::<2, 2>(1, 1).unwrap().into_array(),
    ///     [[5.0, 6.0], [8.0, 9.0]]
    /// );
    /// assert!(matrix.submatrix::<2, 2>(2, 0).is_err());
    /// ```
    #[inline]
    pub fn submatrix<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
        &self,
        top: usize,
        left: usize,
    ) -> Result<Matrix<BLOCK_ROWS, BLOCK_COLS, T>, LinalgError> {
        Ok(self
            .view()
            .submatrix::<BLOCK_ROWS, BLOCK_COLS>(top, left)?
            .to_matrix())
    }

    /// Writes `block` in with its top-left corner at `(top, left)`, or returns
    /// [`LinalgError::OutOfBounds`] — leaving the matrix untouched — if it would run past an edge.
    ///
    /// ```
    /// use multicalc::error::LinalgError;
    /// use multicalc::linear_algebra::Matrix;
    ///
    /// let mut matrix = Matrix::<3, 3>::zeros();
    /// matrix.set_submatrix(1, 1, Matrix::<2, 2>::identity().view()).unwrap();
    /// assert_eq!(matrix.into_array()[1], [0.0, 1.0, 0.0]);
    ///
    /// // A block that would hang off the edge is reported rather than partly written.
    /// let mut small = Matrix::<2, 2>::zeros();
    /// let overhanging = Matrix::<2, 2>::identity();
    /// assert_eq!(
    ///     small.set_submatrix(1, 1, overhanging.view()),
    ///     Err(LinalgError::OutOfBounds)
    /// );
    /// assert_eq!(small, Matrix::<2, 2>::zeros());
    /// ```
    #[inline]
    pub fn set_submatrix<const BLOCK_ROWS: usize, const BLOCK_COLS: usize>(
        &mut self,
        top: usize,
        left: usize,
        block: MatrixView<'_, BLOCK_ROWS, BLOCK_COLS, T>,
    ) -> Result<(), LinalgError> {
        self.view_mut()
            .submatrix::<BLOCK_ROWS, BLOCK_COLS>(top, left)?
            .copy_from(block);
        Ok(())
    }
}
