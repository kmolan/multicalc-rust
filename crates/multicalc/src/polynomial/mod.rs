//! Polynomials: as coefficients, as pieces, and in several variables.
//!
//! - [`Polynomial`] — a fixed number of coefficients, lowest power first, evaluated by repeated
//!   multiply-and-add. One call returns the value and as many derivatives as asked for.
//! - [`RealRoots`] — the real roots a polynomial has, in increasing order.
//!
//! Roots up to the fourth power come from exact closed form solutions.
//! Past that, [`Polynomial::count_real_roots`] says how many real roots a range holds and
//! [`Polynomial::real_roots_in`] finds them within a step budget.
//!
//! Everything is generic over [`Numeric`](crate::Numeric), works without the standard library, and
//! allocates nothing.
//!
//! A product needs more coefficients than either input holds, so operations that grow — multiply,
//! compose, divide — take the output size from the caller and report when it is too small.

// The file carries the same name as the module because it holds the module's namesake type.
#[allow(clippy::module_inception)]
mod polynomial;
mod roots;

pub use polynomial::Polynomial;
pub use roots::RealRoots;
