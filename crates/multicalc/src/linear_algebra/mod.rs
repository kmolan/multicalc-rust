//! Fixed-size, stack-allocated linear algebra.
//!
//! - [`Vector`] / [`Matrix`] — const-generic, array-backed (row-major), no allocation; shape
//!   mismatches are compile errors and the math never panics (only out-of-range indexing does).
//! - [`Cholesky`] / [`Lu`] / [`PivotedQr`] / [`Svd`] — factorizations and the solves built on them.
//!
//! ```compile_fail
//! use multicalc::linear_algebra::Vector;
//! // adding a 2-vector to a 3-vector does not compile
//! let _ = Vector::new([1.0, 2.0]) + Vector::new([1.0, 2.0, 3.0]);
//! ```

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
