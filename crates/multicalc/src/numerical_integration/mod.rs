//! Numerical integration.
//!
//! - [`integral`] — the short way to integrate one function over one interval.
//! - [`gaussian_integration`] — Gaussian quadrature (nodes from
//!   [`gaussian_tables`](crate::gaussian_tables)).
//! - [`iterative_integration`] — iterative refinement of a running estimate.
//! - [`integrator`] — the shared integrator traits; [`mode`] picks the method.

pub mod gaussian_integration;
pub mod integrator;
pub mod iterative_integration;
pub mod mode;

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
/// // 2x over [0, 2] is 4
/// assert!((integral(&|x: f64| 2.0 * x, [0.0, 2.0])? - 4.0).abs() < 1e-9);
///
/// // a decaying integrand may run to infinity: e^-x over [0, inf) is 1
/// assert!((integral(&|x: f64| (-x).exp(), [0.0, f64::INFINITY])? - 1.0).abs() < 1e-6);
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn integral<T: Numeric, F: Fn(T) -> T>(
    function: &F,
    limits: [T; 2],
) -> Result<T, IntegrateError> {
    IterativeSingle::<T>::default().get_single(function, &limits)
}
