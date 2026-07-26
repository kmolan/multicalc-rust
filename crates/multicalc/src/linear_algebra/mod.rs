//! Fixed-size, stack-allocated linear algebra.
//!
//! - [`Vector`] / [`Matrix`] — const-generic, array-backed (row-major), no allocation; shape
//!   mismatches are compile errors. `Index` (`v[i]`, `m[(r, c)]`) is the ergonomic path
//!   (panics on OOB); `get` / `get_mut` / `try_row` / `try_column` return `Option`.

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
