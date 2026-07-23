//! Fixed-size, stack-allocated linear algebra.
//!
//! - [`Vector`] / [`Matrix`] — const-generic, array-backed (row-major), no allocation; shape
//!   mismatches are compile errors and the math never panics. Element access is via `get` /
//!   `get_mut`; row/column copies via `try_row` / `try_column`.
//! - [`Cholesky`] / [`Lu`] / [`PivotedQr`] / [`Svd`] — factorizations and the solves built on them.

pub mod cholesky;
pub mod expm;
pub mod lu;
pub mod macros;
pub mod matrix;
pub mod qr;
pub mod svd;
pub mod vector;

pub use cholesky::Cholesky;
pub use lu::Lu;
pub use matrix::Matrix;
pub use qr::PivotedQr;
pub use svd::Svd;
pub use vector::Vector;

#[cfg(test)]
mod test;
