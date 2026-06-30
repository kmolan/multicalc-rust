//! Fixed-size, stack-allocated linear algebra: [`Vector`] and [`Matrix`].
//!
//! The types are backed by fixed arrays, so they allocate nothing and live on the stack.
//! Dimensions are const generics, so shape mismatches are rejected at compile time. The math
//! operations never panic; only out-of-range indexing does. Matrices are stored row-major.

pub mod matrix;
pub mod vector;

pub use matrix::Matrix;
pub use vector::Vector;

#[cfg(test)]
mod test;
