//! Zero-copy borrowed views over [`Matrix`] and [`Vector`] storage.
//!
//! A view is a flat slice plus an offset and a pair of strides, so the operations that would
//! otherwise copy into a fresh stack matrix become index arithmetic:
//!
//! - [`MatrixView::transposed`] swaps the two strides. No element moves.
//! - [`MatrixView::submatrix`] shifts the offset and narrows the shape.
//! - [`MatrixView::row`] / [`MatrixView::column`] hand back a strided [`VectorView`].
//!
//! All of it is safe code. The flat slice comes from `slice::as_flattened`, and the disjointness
//! of a split is `slice::split_at_mut`'s guarantee rather than a hand-checked invariant.
//!
//! The submodules split along the types: `matrix_view` holds the two matrix views and the
//! [`Matrix`] methods that hand them out, and `vector_view` does the same for [`Vector`].
//! `required_len` below is the one bounds rule both of them share.
//!
//! ```
//! use multicalc::linear_algebra::Matrix;
//!
//! let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let t = m.view().transposed();
//!
//! assert_eq!((t.rows(), t.cols()), (3, 2));
//! assert_eq!(t[(2, 0)], 3.0);
//! assert_eq!(t.to_matrix(), m.transpose());
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
