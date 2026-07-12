//! The scalar number system the calculus modules are generic over.
//!
//! [`Numeric`] is the scalar trait, implemented for `f32` and `f64`. [`Dual`] and [`HyperDual`]
//! are scalar types that also implement it, giving exact first (and, for `HyperDual`, second)
//! derivatives by automatic differentiation.

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
