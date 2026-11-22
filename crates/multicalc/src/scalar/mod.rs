//! The scalar number system the calculus modules are generic over.
//!
//! - [`Numeric`] — the scalar trait, implemented for `f32` and `f64`.
//! - [`Dual`] / [`HyperDual`] / [`Jet`] — scalars that also implement it, giving exact first, second,
//!   and arbitrary-order derivatives by automatic differentiation.
//! - [`ScalarFn`] / [`ScalarFnN`] / [`VectorFn`] — functions evaluable at any [`Numeric`] scalar, so
//!   one formula drives both finite differences and autodiff.

mod dual;
mod function;
mod hyper_dual;
mod jet;
mod numeric;
mod primal;

pub use dual::Dual;
pub use function::{Const, ScalarFn, ScalarFnN, VectorFn, constant};
pub use hyper_dual::HyperDual;
pub use jet::Jet;
pub use numeric::Numeric;
pub use primal::Primal;

/// One output of a vector function seen as a scalar function, reachable inside the crate now that
/// `function` is private.
pub(crate) use function::Component;
