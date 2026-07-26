//! Fixed-size, stack-allocated linear algebra.
//!
//! - [`Vector`] / [`Matrix`] — const-generic, array-backed (row-major), no allocation; shape
//!   mismatches are compile errors. `Index` (`v[i]`, `m[(r, c)]`) is the ergonomic path
//!   (panics on OOB); `get` / `get_mut` / `try_row` / `try_column` return `Option`.

mod cholesky;
mod expm;
mod lu;
mod macros;
mod matrix;
mod qr;
mod svd;
mod vector;

pub use cholesky::Cholesky;
pub use lu::Lu;
pub use matrix::Matrix;
pub use qr::{CholeskyFactor, DampedLeastSquares, PivotedQr};
pub use svd::Svd;
pub use vector::Vector;

/// Shared numeric helpers, reachable inside the crate now that `qr` is private.
pub(crate) use qr::{enorm, max, min};

#[cfg(test)]
mod test;
