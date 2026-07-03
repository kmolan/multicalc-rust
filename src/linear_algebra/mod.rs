//! Fixed-size, stack-allocated linear algebra: [`Vector`] and [`Matrix`].
//!
//! The types are backed by fixed arrays, so they allocate nothing and live on the stack.
//! Dimensions are const generics, so shape mismatches are rejected at compile time. The math
//! operations never panic; only out-of-range indexing does. Matrices are stored row-major.
//!
//! ```compile_fail
//! use multicalc::linear_algebra::Vector;
//! // adding a 2-vector to a 3-vector does not compile
//! let _ = Vector::new([1.0, 2.0]) + Vector::new([1.0, 2.0, 3.0]);
//! ```

pub mod matrix;
pub mod qr;
pub mod vector;

pub use matrix::Matrix;
pub use qr::PivotedQr;
pub use vector::Vector;

#[cfg(test)]
mod test;
