//! Zero-copy borrowed views over [`Matrix`] and [`Vector`] storage.
//!
//! A view is a flat slice, an offset, and a stride per axis, so transposing, blocking, and
//! slicing out a row, column, or diagonal are index arithmetic. Only `to_matrix` / `to_vector`
//! copies.
//!
//! The writable views mirror the read-only ones and add `try_get_mut`, `fill`, and `copy_from`.
//! The splits are the one place they differ: two `&mut` halves must be provably disjoint, so
//! those need a layout `slice::split_at_mut` can separate, while two shared halves may overlap.
//!
//! No `Index`: it returns `&T`, leaving nowhere to report a miss. Fallible calls return
//! `Result<_, LinalgError>`, always
//! [`OutOfBounds`](crate::error::LinalgError::OutOfBounds). All of it is safe code.
//!
//! ```
//! use multicalc::linear_algebra::Matrix;
//! let matrix = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let transposed = matrix.view().transposed();
//! assert_eq!((transposed.rows(), transposed.cols()), (3, 2));
//! assert_eq!(transposed.try_get(2, 0), Ok(&3.0));
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
