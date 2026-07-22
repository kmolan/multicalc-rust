//! Differentiation: exact automatic differentiation and finite differences.
//!
//! - [`autodiff`] / [`finite_difference`] — the two [`derivator`] backends (autodiff is exact).
//! - [`jacobian`] / [`hessian`] — derivative matrices of vector- and scalar-valued functions.

pub mod autodiff;
pub mod derivator;
pub mod finite_difference;
pub mod hessian;
pub mod jacobian;
pub mod mode;
