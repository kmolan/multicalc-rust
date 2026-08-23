//! Zero-copy borrowed views over [`Matrix`] and [`Vector`] storage.
//!
//! A view is a flat slice plus an offset and a pair of strides, so the operations that would
//! otherwise copy into a fresh stack matrix become index arithmetic:
//!
//! - [`MatrixView::transposed`] swaps the two strides. No element moves.
//! - [`MatrixView::submatrix`] shifts the offset and narrows the shape.
//! - [`MatrixView::row`] / [`MatrixView::column`] / [`MatrixView::diagonal`] hand back a strided
//!   [`VectorView`].
//! - [`MatrixView::split_rows_at`] / [`MatrixView::split_cols_at`] and
//!   [`VectorView::split_at`] cut a view in two, and their `Mut` counterparts hand back two
//!   halves that can be written through at the same time.
//!
//! The read-only and writable views carry the same surface. Anything `MatrixView` can do,
//! [`MatrixViewMut`] can do — plus [`MatrixViewMut::get_mut`], [`MatrixViewMut::fill`], and
//! [`MatrixViewMut::copy_from`] — and the same holds for [`VectorView`] and [`VectorViewMut`].
//! The writable splits are the one place the two differ in what they accept, because two
//! `&mut` halves have to be provably disjoint and two shared halves do not; each method says so.
//!
//! Nothing here panics and nothing here is `Index`. Every fallible operation returns
//! `Result<_, LinalgError>` and the error is always
//! [`LinalgError::OutOfBounds`](crate::error::LinalgError::OutOfBounds), so a caller
//! that only wants to know whether a window fits can test with `is_ok`. `Index` is absent
//! because it returns `&T`, which leaves nowhere to put that error.
//!
//! All of it is safe code. The flat slice comes from `slice::as_flattened`, and the disjointness
//! of a writable split is `slice::split_at_mut`'s guarantee rather than a hand-checked invariant.
//!
//! The submodules split along the types: `matrix_view` holds the two matrix views and the
//! [`Matrix`] methods that hand them out, and `vector_view` does the same for [`Vector`].
//! `required_len` below is the one bounds rule both of them share.
//!
//! ```
//! use multicalc::linear_algebra::Matrix;
//!
//! let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let transposed = matrix.view().transposed();
//!
//! assert_eq!((transposed.rows(), transposed.cols()), (3, 2));
//! assert_eq!(transposed.get(2, 0), Ok(&3.0));
//! assert_eq!(transposed.to_matrix(), matrix.transpose());
//! ```

mod matrix_view;
mod vector_view;

pub use matrix_view::{MatrixView, MatrixViewMut};
pub use vector_view::{VectorView, VectorViewMut};

/// The smallest slice length that can hold a `rows`×`cols` view with this offset and these
/// strides, or `None` if the arithmetic overflows. An empty shape needs only `offset` elements,
/// since it never dereferences anything.
#[inline]
fn required_len(
    rows: usize,
    cols: usize,
    offset: usize,
    row_stride: usize,
    col_stride: usize,
) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(offset);
    }
    let last_row = (rows - 1).checked_mul(row_stride)?;
    let last_col = (cols - 1).checked_mul(col_stride)?;
    offset
        .checked_add(last_row)?
        .checked_add(last_col)?
        .checked_add(1)
}
