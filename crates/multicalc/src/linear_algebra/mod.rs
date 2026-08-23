//! Fixed-size, stack-allocated linear algebra.
//!
//! - [`Vector`] / [`Matrix`] — const-generic, array-backed (row-major), no allocation; shape
//!   mismatches are compile errors. `Index` (`v[i]`, `m[(r, c)]`) is the ergonomic path
//!   (panics on OOB); `get` / `get_mut` / `try_row` / `try_column` return `Option`.
//! - [`Vector2D`] / [`Vector3D`] / [`Vector6D`] and [`Matrix2D`] / [`Matrix3D`] / [`Matrix4D`] /
//!   [`Matrix6D`] — short names for the sizes that come up most, so a call site writes
//!   `Vector3D<T>` rather than `Vector<3, T>`. Each is the same type as what it stands for, so
//!   the two spellings mix freely.
//! - [`MatrixView`] / [`VectorView`] and their `Mut` counterparts — borrowed, strided windows
//!   onto existing storage. Transpose, submatrix, and row/column extraction are index arithmetic
//!   on a view, so they cost nothing until [`MatrixView::to_matrix`] copies one out.
//! - [`solve_discrete_riccati`] — the steady-state cost-to-go an optimal linear feedback law is
//!   built from.
//! - [`solve_discrete_lyapunov`] — solves `Aᵀ·P·A − P + Q = 0`, which is how a closed loop is
//!   shown to settle.

mod cholesky;
mod expm;
#[allow(clippy::min_ident_chars)]
mod lu;
mod lyapunov;
mod macros;
mod matrix;
#[allow(clippy::min_ident_chars)]
mod qr;
mod riccati;
mod svd;
mod symmetric_eigendecomposition;
mod vector;
mod view;

pub use cholesky::Cholesky;
pub use lyapunov::solve_discrete_lyapunov;
pub use matrix::Matrix;
pub use riccati::solve_discrete_riccati;
pub use svd::Svd;
pub use symmetric_eigendecomposition::SymmetricEigendecomposition;
pub use vector::Vector;
pub use view::{MatrixView, MatrixViewMut, VectorView, VectorViewMut};

// Clippy flags `lu`/`qr` in `use` paths, but `#[allow]`/`#[expect]` on those `use` items are
// reported as useless; a module-level allow is what actually covers the re-exports.
#[allow(clippy::min_ident_chars)]
mod short_name_reexports {
    pub use super::lu::LuDecomposition;
    pub use super::qr::{CholeskyFactor, DampedLeastSquares, PivotedQr};
    pub(crate) use super::qr::{enorm, max, min};
}
pub use short_name_reexports::{CholeskyFactor, DampedLeastSquares, LuDecomposition, PivotedQr};
/// Shared numeric helpers, reachable inside the crate now that `qr` is private.
pub(crate) use short_name_reexports::{enorm, max, min};

// Vector type aliases for ease of life
pub type Vector2D<T = f64> = Vector<2, T>;
pub type Vector3D<T = f64> = Vector<3, T>;
pub type Vector6D<T = f64> = Vector<6, T>;

// Matrix type aliases for ease of life
pub type Matrix2D<T = f64> = Matrix<2, 2, T>;
pub type Matrix3D<T = f64> = Matrix<3, 3, T>;
pub type Matrix4D<T = f64> = Matrix<4, 4, T>;
pub type Matrix6D<T = f64> = Matrix<6, 6, T>;

#[cfg(test)]
mod test;
