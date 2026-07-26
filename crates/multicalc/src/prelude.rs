//! The traits and one-call functions worth importing together.
//!
//! Glob-import this, then name the types you need from the crate root:
//!
//! ```
//! use multicalc::prelude::*;
//! use multicalc::scalar_fn;
//!
//! // f(x) = x^2, so f'(3) = 6
//! let square = scalar_fn!(|x| x * x);
//! assert!((derivative(&square, 3.0_f64) - 6.0).abs() < 1e-12);
//! ```
//!
//! Only traits and functions live here; types stay at the crate root, so there is one place to
//! look for them. `SummationMethod` is the one name with two paths, since both
//! `numerical_integration` and `approximation` take it as a parameter.

pub use crate::error::CalcError;
pub use crate::numerical_derivative::{
    DerivatorMultiVariable, DerivatorSingleVariable, derivative, partial, second_derivative,
};
pub use crate::numerical_integration::{
    IntegratorMultiVariable, IntegratorSingleVariable, integral,
};
pub use crate::random::RandomSource;
pub use crate::scalar::{Numeric, Primal, ScalarFn, ScalarFnN, VectorFn};
