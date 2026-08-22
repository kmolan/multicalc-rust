//! The borrowed matrix views, and the [`Matrix`] methods that hand them out.

use core::ops::{Index, IndexMut};

use super::{VectorView, VectorViewMut, required_len};
use crate::linear_algebra::Matrix;

// Out-of-range subscript handling, shared by the `Index` impls below.
//
// The crate denies `clippy::panic` because computational paths must return `LinalgError` rather
// than abort. `Index` is not one of those paths: it returns `&T`, so there is no error to give
// back, and every `Index` impl in `core` panics for exactly this reason. `Matrix` and `Vector`
// already do the same -- `m[(r, c)]` panics today, it just reaches the panic through slice
// indexing, which the lint does not see. So the behaviour here is the crate's existing contract
// made explicit and given a message that names the shape. `get` / `get_mut` are the fallible
// path and are what the docs point callers at.
//
// The check cannot be delegated to the slice the way `Matrix` delegates it to its arrays. A view
// is strided, so an out-of-range *subscript* often computes an in-range *flat index*: in a 2x2
// block taken from a 3x3 matrix (strides 3 and 1), the invalid subscript (2, 0) works out to
// flat index 6, which is a perfectly valid element of the 9-element parent buffer. Letting the
// slice bound decide would silently return the wrong entry instead of failing. Hence the
// explicit comparison against ROWS/COLS, and hence the explicit panic when it fails.
#[cold]
#[track_caller]
#[allow(clippy::panic)]
fn matrix_out_of_bounds(row: usize, col: usize, rows: usize, cols: usize) -> ! {
    panic!("matrix view index ({row}, {col}) out of range for a {rows}x{cols} view")
}

/// A borrowed, strided, read-only `ROWS`×`COLS` window onto someone else's storage.
///
/// Reshaping is free: [`transposed`](Self::transposed) and [`submatrix`](Self::submatrix) only
/// rewrite the offset and strides, so neither touches an element.
///
/// ```
/// use multicalc::linear_algebra::Matrix;
///
/// let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
/// let corner = m.view().submatrix::<2, 2>(1, 1).unwrap();
///
/// assert_eq!(corner.to_matrix().into_array(), [[5.0, 6.0], [8.0, 9.0]]);
/// ```
#[derive(Debug)]
#[must_use]
pub struct MatrixView<'a, const ROWS: usize, const COLS: usize, T = f64> {
    data: &'a [T],
    offset: usize,
    row_stride: usize,
    col_stride: usize,
}

/// A borrowed, strided, writable `ROWS`×`COLS` window onto someone else's storage.
///
/// The reshaping methods consume the view, because the exclusive borrow has to move with them.
/// [`as_view`](Self::as_view) and [`reborrow`](Self::reborrow) hand out shorter-lived views when
/// the original needs to stay usable.
#[derive(Debug)]
#[must_use]
pub struct MatrixViewMut<'a, const ROWS: usize, const COLS: usize, T = f64> {
    data: &'a mut [T],
    offset: usize,
    row_stride: usize,
    col_stride: usize,
}

// `Clone` and `Copy` are written out rather than derived, because a derive would demand
// `T: Copy`. What is copied is the handle -- the slice reference, offset, and two strides -- so
// it is `Copy` for the same reason `&[T]` is, whatever `T` turns out to be. The elements are
// never touched; `to_matrix` is the only thing that copies those.
impl<const ROWS: usize, const COLS: usize, T> Clone for MatrixView<'_, ROWS, COLS, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}
impl<const ROWS: usize, const COLS: usize, T> Copy for MatrixView<'_, ROWS, COLS, T> {}

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixView<'a, ROWS, COLS, T> {
    /// Builds a view over `data`, or `None` if the shape would reach past the end of the slice.
    /// Every constructor funnels through here, so an existing view is always in range and `get`
    /// can never be defeated by a bad stride.
    #[inline]
    fn from_parts(
        data: &'a [T],
        offset: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Option<Self> {
        let needed = required_len(ROWS, COLS, offset, row_stride, col_stride)?;
        (needed <= data.len()).then_some(MatrixView {
            data,
            offset,
            row_stride,
            col_stride,
        })
    }

    /// Views a row-major slice as a matrix, or `None` if it holds fewer than `ROWS * COLS`
    /// elements. Trailing elements are ignored, which is what makes a scratch buffer reusable.
    ///
    /// ```
    /// use multicalc::linear_algebra::MatrixView;
    /// let buffer = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let v = MatrixView::<2, 2>::from_row_major_slice(&buffer).unwrap();
    /// assert_eq!(v[(1, 0)], 3.0);
    /// assert!(MatrixView::<3, 2>::from_row_major_slice(&buffer).is_none());
    /// ```
    #[inline]
    pub fn from_row_major_slice(slice: &'a [T]) -> Option<Self> {
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
    /// let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// assert_eq!(m.view().strides(), (3, 1));
    /// assert_eq!(m.view().transposed().strides(), (1, 3));
    /// ```
    #[inline]
    #[must_use]
    pub const fn strides(&self) -> (usize, usize) {
        (self.row_stride, self.col_stride)
    }

    /// Whether the elements are laid out contiguously in row-major order, which is the layout the
    /// row-splitting methods on [`MatrixViewMut`] need.
    #[inline]
    #[must_use]
    pub const fn is_row_major(&self) -> bool {
        // With one column or none, `col_stride` addresses nothing and any row stride wide
        // enough to keep the rows apart is separable; otherwise the rows must be unpadded runs.
        self.row_stride >= COLS && (COLS <= 1 || self.col_stride == 1)
    }

    #[inline]
    fn index_of(&self, row: usize, col: usize) -> Option<usize> {
        (row < ROWS && col < COLS).then_some(())?;
        self.offset
            .checked_add(row.checked_mul(self.row_stride)?)?
            .checked_add(col.checked_mul(self.col_stride)?)
    }

    /// Returns a reference to entry `(row, col)`, or `None` if out of range.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.view().get(1, 0), Some(&3.0));
    /// assert_eq!(m.view().get(2, 0), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(self.index_of(row, col)?)
    }

    /// The transpose, in constant time: the two strides trade places and nothing is copied.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::new([[1.0, 2.0, 3.0]]);
    /// let t = m.view().transposed();
    /// assert_eq!((t.rows(), t.cols()), (3, 1));
    /// assert_eq!(t.to_matrix().into_array(), [[1.0], [2.0], [3.0]]);
    /// ```
    #[inline]
    pub fn transposed(self) -> MatrixView<'a, COLS, ROWS, T> {
        MatrixView {
            data: self.data,
            offset: self.offset,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    /// The `R`×`C` block whose top-left corner sits at `(top, left)`, or `None` if that block
    /// would run past an edge.
    ///
    /// `top` and `left` say *where* the block starts, counted in the current view's rows and
    /// columns. `R` and `C` say *how big* it is. The size is a const parameter rather than an
    /// argument because it is part of the returned type — a 2×2 block and a 3×3 block are
    /// different types, so the shape has to be known at compile time, while the corner is free to
    /// be a runtime value.
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
    /// let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let block = m.view().submatrix::<2, 2>(0, 1).unwrap();
    /// assert_eq!(block.to_matrix().into_array(), [[2.0, 3.0], [5.0, 6.0]]);
    /// assert!(m.view().submatrix::<2, 2>(0, 2).is_none());
    /// ```
    #[inline]
    pub fn submatrix<const R: usize, const C: usize>(
        self,
        top: usize,
        left: usize,
    ) -> Option<MatrixView<'a, R, C, T>> {
        (top.checked_add(R)? <= ROWS && left.checked_add(C)? <= COLS).then_some(())?;
        let offset = self
            .offset
            .checked_add(top.checked_mul(self.row_stride)?)?
            .checked_add(left.checked_mul(self.col_stride)?)?;
        MatrixView::from_parts(self.data, offset, self.row_stride, self.col_stride)
    }

    /// Row `row` as a view, or `None` if `row >= ROWS`. Unlike [`Matrix::try_row`], which copies
    /// the entries into an owned vector, this only works out where the row starts.
    ///
    /// The result has `COLS` components, because a row spans every column. Its stride is the
    /// view's `col_stride` — the gap between neighbouring entries *within* a row.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.view().row(1).unwrap().to_vector().into_array(), [3.0, 4.0]);
    /// ```
    #[inline]
    pub fn row(self, row: usize) -> Option<VectorView<'a, COLS, T>> {
        (row < ROWS).then_some(())?;
        let offset = self.offset.checked_add(row.checked_mul(self.row_stride)?)?;
        VectorView::from_parts(self.data, offset, self.col_stride)
    }

    /// Column `column` as a view, or `None` if `column >= COLS`. Unlike [`Matrix::try_column`],
    /// which copies, this only works out where the column starts.
    ///
    /// The result has `ROWS` components, because a column spans every row. Its stride is the
    /// view's `row_stride`: consecutive entries of a column are one whole row apart in the
    /// buffer, and carrying that stride around instead of gathering the entries is what saves
    /// the copy.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.view().column(1).unwrap().to_vector().into_array(), [2.0, 4.0]);
    /// ```
    #[inline]
    pub fn column(self, column: usize) -> Option<VectorView<'a, ROWS, T>> {
        (column < COLS).then_some(())?;
        let offset = self
            .offset
            .checked_add(column.checked_mul(self.col_stride)?)?;
        VectorView::from_parts(self.data, offset, self.row_stride)
    }

    /// The main diagonal — entries `(0, 0)`, `(1, 1)`, `(2, 2)` and so on — as a view, without
    /// gathering them into a vector.
    ///
    /// A diagonal stops at whichever edge it reaches first, so it holds `min(ROWS, COLS)`
    /// entries. Rust cannot yet work that out inside a type, so the length arrives as `N` and is
    /// checked instead: pass the shorter side, or get `None` back. For a 2×4 view that is `2`.
    ///
    /// The stride is `row_stride + col_stride`, because one step along the diagonal moves down a
    /// row *and* right a column. On a 3×3 row-major matrix that is `3 + 1 = 4`, landing on buffer
    /// positions 0, 4 and 8 — the diagonal.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// assert_eq!(m.view().diagonal::<2>().unwrap().to_vector().into_array(), [1.0, 5.0]);
    /// ```
    #[inline]
    pub fn diagonal<const N: usize>(self) -> Option<VectorView<'a, N, T>> {
        let shorter_side = if ROWS < COLS { ROWS } else { COLS };
        (N == shorter_side).then_some(())?;
        VectorView::from_parts(
            self.data,
            self.offset,
            self.row_stride.checked_add(self.col_stride)?,
        )
    }
}

impl<const ROWS: usize, const COLS: usize, T: Copy> MatrixView<'_, ROWS, COLS, T> {
    /// Copies the window into an owned matrix. This is the one operation here that moves elements.
    #[inline]
    pub fn to_matrix(self) -> Matrix<ROWS, COLS, T> {
        Matrix::from_fn(|row, column| self[(row, column)])
    }
}

impl<const ROWS: usize, const COLS: usize, T> Index<(usize, usize)>
    for MatrixView<'_, ROWS, COLS, T>
{
    type Output = T;

    /// Panics if the subscript is out of range. Use [`Self::get`] when it may be.
    #[inline]
    #[track_caller]
    fn index(&self, (row, column): (usize, usize)) -> &T {
        match self.get(row, column) {
            Some(value) => value,
            None => matrix_out_of_bounds(row, column, ROWS, COLS),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, T: PartialEq> PartialEq
    for MatrixView<'_, ROWS, COLS, T>
{
    /// Compares element by element, so two views of different layouts over different buffers are
    /// equal when they present the same entries.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        (0..ROWS).all(|r| (0..COLS).all(|c| self.get(r, c) == other.get(r, c)))
    }
}

impl<'a, const ROWS: usize, const COLS: usize, T> MatrixViewMut<'a, ROWS, COLS, T> {
    /// Builds a view over `data`, or `None` if the shape would reach past the end of the slice.
    #[inline]
    pub(super) fn from_parts(
        data: &'a mut [T],
        offset: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Option<Self> {
        let needed = required_len(ROWS, COLS, offset, row_stride, col_stride)?;
        (needed <= data.len()).then_some(MatrixViewMut {
            data,
            offset,
            row_stride,
            col_stride,
        })
    }

    /// Views a row-major slice as a writable matrix, or `None` if it holds fewer than
    /// `ROWS * COLS` elements.
    ///
    /// ```
    /// use multicalc::linear_algebra::MatrixViewMut;
    /// let mut buffer = [0.0; 6];
    /// let mut v = MatrixViewMut::<2, 3>::from_row_major_slice(&mut buffer).unwrap();
    /// v[(1, 2)] = 7.0;
    /// assert_eq!(buffer, [0.0, 0.0, 0.0, 0.0, 0.0, 7.0]);
    /// ```
    #[inline]
    pub fn from_row_major_slice(slice: &'a mut [T]) -> Option<Self> {
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
    fn index_of(&self, row: usize, col: usize) -> Option<usize> {
        (row < ROWS && col < COLS).then_some(())?;
        self.offset
            .checked_add(row.checked_mul(self.row_stride)?)?
            .checked_add(col.checked_mul(self.col_stride)?)
    }

    /// Returns a reference to entry `(row, col)`, or `None` if out of range.
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(self.index_of(row, col)?)
    }

    /// Returns a mutable reference to entry `(row, col)`, or `None` if out of range.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        let index = self.index_of(row, col)?;
        self.data.get_mut(index)
    }

    /// Borrows this window read-only for as long as `self` is untouched.
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
    /// let mut m = Matrix::<2, 2>::zeros();
    /// let mut v = m.view_mut();
    /// v.reborrow().transposed()[(0, 1)] = 5.0;
    /// v[(0, 0)] = 1.0;
    /// assert_eq!(m.into_array(), [[1.0, 0.0], [5.0, 0.0]]);
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
    /// let mut m = Matrix::<2, 3>::zeros();
    /// m.view_mut().transposed()[(2, 1)] = 9.0;
    /// assert_eq!(m[(1, 2)], 9.0);
    /// ```
    #[inline]
    pub fn transposed(self) -> MatrixViewMut<'a, COLS, ROWS, T> {
        MatrixViewMut {
            data: self.data,
            offset: self.offset,
            row_stride: self.col_stride,
            col_stride: self.row_stride,
        }
    }

    /// The `R`×`C` writable block whose top-left corner sits at `(top, left)`, or `None` if that
    /// block would run past an edge. The read-only [`MatrixView::submatrix`] explains the
    /// coordinates and the stride arithmetic; this is the same operation on a writable view.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let identity = Matrix::<2, 2>::identity();
    /// let mut m = Matrix::<3, 3>::zeros();
    ///
    /// m.view_mut().submatrix::<2, 2>(1, 1).unwrap().copy_from(identity.view());
    ///
    /// assert_eq!(m.into_array(), [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// ```
    #[inline]
    pub fn submatrix<const R: usize, const C: usize>(
        self,
        top: usize,
        left: usize,
    ) -> Option<MatrixViewMut<'a, R, C, T>> {
        (top.checked_add(R)? <= ROWS && left.checked_add(C)? <= COLS).then_some(())?;
        let offset = self
            .offset
            .checked_add(top.checked_mul(self.row_stride)?)?
            .checked_add(left.checked_mul(self.col_stride)?)?;
        MatrixViewMut::from_parts(self.data, offset, self.row_stride, self.col_stride)
    }

    /// Row `row` as a writable view of `COLS` components, or `None` if `row >= ROWS`. See
    /// [`MatrixView::row`].
    #[inline]
    pub fn row(self, row: usize) -> Option<VectorViewMut<'a, COLS, T>> {
        (row < ROWS).then_some(())?;
        let offset = self.offset.checked_add(row.checked_mul(self.row_stride)?)?;
        VectorViewMut::from_parts(self.data, offset, self.col_stride)
    }

    /// Column `column` as a writable view of `ROWS` components, or `None` if `column >= COLS`.
    /// See [`MatrixView::column`].
    #[inline]
    pub fn column(self, column: usize) -> Option<VectorViewMut<'a, ROWS, T>> {
        (column < COLS).then_some(())?;
        let offset = self
            .offset
            .checked_add(column.checked_mul(self.col_stride)?)?;
        VectorViewMut::from_parts(self.data, offset, self.row_stride)
    }

    /// Splits into the first `TOP` rows and the remaining `BOTTOM`, as two views that can be
    /// written through at the same time.
    ///
    /// Returns `None` unless `TOP + BOTTOM == ROWS` and the view
    /// [is row-major](Self::is_row_major) — a transposed view interleaves its rows in the buffer,
    /// so no cut of the slice separates them.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut m = Matrix::<3, 2>::zeros();
    /// let (mut top, mut bottom) = m.view_mut().split_rows_at::<1, 2>().unwrap();
    /// top[(0, 0)] = 1.0;
    /// bottom[(1, 1)] = 2.0;
    /// assert_eq!(m.into_array(), [[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]]);
    /// ```
    #[inline]
    pub fn split_rows_at<const TOP: usize, const BOTTOM: usize>(
        self,
    ) -> Option<(
        MatrixViewMut<'a, TOP, COLS, T>,
        MatrixViewMut<'a, BOTTOM, COLS, T>,
    )> {
        (TOP.checked_add(BOTTOM)? == ROWS && self.is_row_major()).then_some(())?;
        let split = self.offset.checked_add(TOP.checked_mul(self.row_stride)?)?;
        (split <= self.data.len()).then_some(())?;
        let (head, tail) = self.data.split_at_mut(split);
        Some((
            MatrixViewMut::from_parts(head, self.offset, self.row_stride, self.col_stride)?,
            MatrixViewMut::from_parts(tail, 0, self.row_stride, self.col_stride)?,
        ))
    }
}

impl<const ROWS: usize, const COLS: usize, T: Copy> MatrixViewMut<'_, ROWS, COLS, T> {
    /// Copies the window into an owned matrix.
    #[inline]
    pub fn to_matrix(&self) -> Matrix<ROWS, COLS, T> {
        self.as_view().to_matrix()
    }

    /// Overwrites every entry with `value`.
    #[inline]
    pub fn fill(&mut self, value: T) {
        for r in 0..ROWS {
            for c in 0..COLS {
                if let Some(slot) = self.get_mut(r, c) {
                    *slot = value;
                }
            }
        }
    }

    /// Copies `src` in element by element. The two may have different layouts — writing a
    /// transposed view through here is how a transpose lands in a caller's workspace without an
    /// intermediate stack matrix.
    ///
    /// ```
    /// use multicalc::linear_algebra::{Matrix, MatrixViewMut};
    /// let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    /// let mut scratch = [0.0; 6];
    /// let mut dst = MatrixViewMut::<3, 2>::from_row_major_slice(&mut scratch).unwrap();
    /// dst.copy_from(m.view().transposed());
    /// assert_eq!(scratch, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    /// ```
    #[inline]
    pub fn copy_from(&mut self, src: MatrixView<'_, ROWS, COLS, T>) {
        for r in 0..ROWS {
            for c in 0..COLS {
                if let (Some(value), Some(slot)) = (src.get(r, c).copied(), self.get_mut(r, c)) {
                    *slot = value;
                }
            }
        }
    }
}

impl<const ROWS: usize, const COLS: usize, T> Index<(usize, usize)>
    for MatrixViewMut<'_, ROWS, COLS, T>
{
    type Output = T;

    /// Panics if the subscript is out of range. Use [`Self::get`] when it may be.
    #[inline]
    #[track_caller]
    fn index(&self, (row, col): (usize, usize)) -> &T {
        match self.get(row, col) {
            Some(value) => value,
            None => matrix_out_of_bounds(row, col, ROWS, COLS),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, T> IndexMut<(usize, usize)>
    for MatrixViewMut<'_, ROWS, COLS, T>
{
    /// Panics if the subscript is out of range. Use [`Self::get_mut`] when it may be.
    #[inline]
    #[track_caller]
    fn index_mut(&mut self, (row, column): (usize, usize)) -> &mut T {
        match self.get_mut(row, column) {
            Some(value) => value,
            None => matrix_out_of_bounds(row, column, ROWS, COLS),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, T> Matrix<ROWS, COLS, T> {
    /// A read-only view of the whole matrix. The entry point for the zero-copy reshaping in
    /// [`MatrixView`].
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    /// assert_eq!(m.view()[(0, 1)], 2.0);
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
    /// let mut m = Matrix::<2, 2>::zeros();
    /// m.view_mut().transposed()[(0, 1)] = 5.0;
    /// assert_eq!(m.into_array(), [[0.0, 0.0], [5.0, 0.0]]);
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
    /// Copies out the `R`×`C` block whose top-left corner is `(top, left)`, or `None` if that
    /// block would run past an edge.
    ///
    /// This is [`MatrixView::submatrix`] followed by one copy. Take the view directly when the
    /// block only needs reading.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    /// assert_eq!(
    ///     m.submatrix::<2, 2>(1, 1).unwrap().into_array(),
    ///     [[5.0, 6.0], [8.0, 9.0]]
    /// );
    /// assert!(m.submatrix::<2, 2>(2, 0).is_none());
    /// ```
    #[inline]
    pub fn submatrix<const R: usize, const C: usize>(
        &self,
        top: usize,
        left: usize,
    ) -> Option<Matrix<R, C, T>> {
        Some(self.view().submatrix::<R, C>(top, left)?.to_matrix())
    }

    /// Writes the `R`×`C` block `block` with its top-left corner at `(top, left)`, or returns
    /// `false` if it would run past an edge.
    ///
    /// ```
    /// use multicalc::linear_algebra::Matrix;
    /// let mut m = Matrix::<3, 3>::zeros();
    /// assert!(m.set_submatrix(1, 1, Matrix::<2, 2>::identity().view()));
    /// assert_eq!(m.into_array()[1], [0.0, 1.0, 0.0]);
    /// ```
    #[inline]
    pub fn set_submatrix<const R: usize, const C: usize>(
        &mut self,
        top: usize,
        left: usize,
        block: MatrixView<'_, R, C, T>,
    ) -> bool {
        match self.view_mut().submatrix::<R, C>(top, left) {
            Some(mut target) => {
                target.copy_from(block);
                true
            }
            None => false,
        }
    }
}
