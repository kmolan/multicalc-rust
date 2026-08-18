//! Fixed-size, stack-allocated linear algebra.
//!
//! - [`Vector`] / [`Matrix`] — const-generic, array-backed (row-major), no allocation; shape
//!   mismatches are compile errors. `Index` (`v[i]`, `m[(r, c)]`) is the ergonomic path
//!   (panics on OOB); `get` / `get_mut` / `try_row` / `try_column` return `Option`.
//! - [`Vector2D`] / [`Vector3D`] / [`Vector6D`] and [`Matrix2D`] / [`Matrix3D`] / [`Matrix4D`] /
//!   [`Matrix6D`] — short names for the sizes that come up most, so a call site writes
//!   `Vector3D<T>` rather than `Vector<3, T>`. Each is the same type as what it stands for, so
//!   the two spellings mix freely.
//! - [`solve_discrete_riccati`] — the steady-state cost-to-go an optimal linear feedback law is
//!   built from.
//! - [`solve_discrete_lyapunov`] — solves `Aᵀ·P·A − P + Q = 0`, which is how a closed loop is
//!   shown to settle.

mod cholesky;
mod expm;
mod lu;
mod lyapunov;
mod macros;
mod matrix;
mod qr;
mod riccati;
mod svd;
mod symmetric_eigendecomposition;
mod vector;

pub use cholesky::Cholesky;
pub use lu::Lu;
pub use lyapunov::solve_discrete_lyapunov;
pub use matrix::Matrix;
pub use qr::{CholeskyFactor, DampedLeastSquares, PivotedQr};
pub use riccati::solve_discrete_riccati;
pub use svd::{Svd, SvdSettings};
pub use symmetric_eigendecomposition::SymmetricEigendecomposition;
pub use vector::Vector;

// Vector type aliases for ease of life
pub type Vector2D<T = f64> = Vector<2, T>;
pub type Vector3D<T = f64> = Vector<3, T>;
pub type Vector6D<T = f64> = Vector<6, T>;

// Matrix type aliases for ease of life
pub type Matrix2D<T = f64> = Matrix<2, 2, T>;
pub type Matrix3D<T = f64> = Matrix<3, 3, T>;
pub type Matrix4D<T = f64> = Matrix<4, 4, T>;
pub type Matrix6D<T = f64> = Matrix<6, 6, T>;

/// Shared numeric helpers, reachable inside the crate now that `qr` is private.
pub(crate) use qr::{enorm, max, min};

#[cfg(test)]
mod test;
