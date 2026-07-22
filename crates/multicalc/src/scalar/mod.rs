//! The scalar number system the calculus modules are generic over.
//!
//! - [`Numeric`] — the scalar trait, implemented for `f32` and `f64`.
//! - [`Dual`] / [`HyperDual`] / [`Jet`] — scalars that also implement it, giving exact first, second,
//!   and arbitrary-order derivatives by automatic differentiation.
//! - [`ScalarFn`] / [`ScalarFnN`] / [`VectorFn`] — functions evaluable at any [`Numeric`] scalar, so
//!   one formula drives both finite differences and autodiff.

pub mod dual;
pub mod function;
pub mod hyper_dual;
pub mod jet;
pub mod numeric;
pub mod primal;

pub use dual::Dual;
pub use function::{ScalarFn, ScalarFnN, VectorFn, c};
pub use hyper_dual::HyperDual;
pub use jet::Jet;
pub use numeric::Numeric;
pub use primal::Primal;
