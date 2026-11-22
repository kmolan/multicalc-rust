//! Numerical integration.
//!
//! - [`integral`] — the short way to integrate one function over one interval.
//! - [`GaussianSingle`] / [`GaussianMulti`] — Gaussian quadrature (nodes from
//!   [`gaussian_tables`](crate::gaussian_tables)), picking a family with
//!   [`GaussianQuadratureMethod`].
//! - [`IterativeSingle`] / [`IterativeMulti`] — iterative refinement of a running estimate, picking
//!   a rule with [`IterativeMethod`].
//! - [`IntegratorSingleVariable`] / [`IntegratorMultiVariable`] — the shared integrator traits.

mod gaussian_integration;
mod integrator;
mod iterative_integration;
mod mode;

pub use crate::utils::summation::SummationMethod;

pub use gaussian_integration::{
    DEFAULT_QUADRATURE_ORDERS, GaussianConfig, GaussianMulti, GaussianSingle,
};
pub use integrator::{IntegratorMultiVariable, IntegratorSingleVariable};
pub use iterative_integration::{
    DEFAULT_TOTAL_ITERATIONS, IterativeConfig, IterativeMulti, IterativeSingle,
};
pub use mode::{GaussianQuadratureMethod, IterativeMethod};

use crate::error::IntegrateError;
use crate::scalar::Numeric;

/// The integral of a single-variable function over an interval.
///
/// This picks a method on your behalf: it walks the interval in [`DEFAULT_TOTAL_ITERATIONS`] steps
/// using Boole's rule, which is the strongest all-round choice for a smooth integrand. Reach for
/// [`IterativeSingle`] to change the rule or the step count, or [`GaussianSingle`] for quadrature.
/// Either limit may be infinite.
///
/// # Errors
/// [`IntegrateError::LimitsIllDefined`] if the limits are reversed, equal, `NaN`, or point the
/// wrong way at an infinity, or [`IntegrateError::NonFinite`] if the integrand blows up on the way.
///
/// # Examples
/// ```
/// use multicalc::numerical_integration::integral;
/// # fn main() -> Result<(), multicalc::error::IntegrateError> {
/// let line = |x: f64| 2.0 * x;
/// let limits = [0.0, 2.0];
///
/// let area = integral(&line, limits)?;                    // 2x over [0, 2] is 4
/// assert!((area - 4.0).abs() < 1e-9);
///
/// // a decaying integrand may run to infinity
/// let decay = |x: f64| (-x).exp();
/// let to_infinity = [0.0, f64::INFINITY];
///
/// let tail = integral(&decay, to_infinity)?;              // e^-x over [0, inf) is 1
/// assert!((tail - 1.0).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn integral<T: Numeric, F: Fn(T) -> T>(
    function: &F,
    limits: [T; 2],
) -> Result<T, IntegrateError> {
    IterativeSingle::<T>::default().single_integral(function, &limits)
}
