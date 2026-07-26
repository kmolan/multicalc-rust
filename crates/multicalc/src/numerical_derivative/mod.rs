//! Differentiation: exact automatic differentiation and finite differences.
//!
//! - [`derivative`] / [`second_derivative`] / [`partial`] — the short way to take one derivative.
//! - [`autodiff`] / [`finite_difference`] — the two [`derivator`] backends (autodiff is exact).
//! - [`jacobian`] / [`hessian`] — derivative matrices of vector- and scalar-valued functions.

pub mod autodiff;
pub mod derivator;
pub mod finite_difference;
pub mod hessian;
pub mod jacobian;
pub mod mode;

pub use autodiff::{AutoDiffMulti, AutoDiffSingle};
pub use derivator::{DerivatorMultiVariable, DerivatorSingleVariable};
pub use finite_difference::{
    FiniteDifferenceConfig, FiniteDifferenceMulti, FiniteDifferenceSingle,
};
pub use hessian::Hessian;
pub use jacobian::Jacobian;
pub use mode::{DEFAULT_STEP_SIZE, DEFAULT_STEP_SIZE_MULTIPLIER, FiniteDifferenceMode};

use crate::error::DiffError;
use crate::scalar::{Dual, HyperDual, Numeric, ScalarFn, ScalarFnN};

/// The derivative of a single-variable function at a point.
///
/// For a third or higher derivative, or to choose finite differences instead,
/// use [`AutoDiffSingle`] or [`FiniteDifferenceSingle`].
///
/// # Examples
/// ```
/// use multicalc::numerical_derivative::derivative;
/// use multicalc::scalar_fn;
///
/// // f(x) = x^3, so f'(2) = 12
/// let f = scalar_fn!(|x| x * x * x);
/// assert!((derivative(&f, 2.0_f64) - 12.0).abs() < 1e-12);
/// ```
#[must_use]
#[inline]
pub fn derivative<T: Numeric, F: ScalarFn>(function: &F, point: T) -> T {
    function.eval(Dual::variable(point)).deriv
}

/// The second derivative of a single-variable function at a point.
///
/// # Examples
/// ```
/// use multicalc::numerical_derivative::second_derivative;
/// use multicalc::scalar_fn;
///
/// // f(x) = x^3, so f''(2) = 12; f32 works the same way
/// let f = scalar_fn!(|x| x * x * x);
/// assert!((second_derivative(&f, 2.0_f32) - 12.0).abs() < 1e-4);
/// ```
#[must_use]
#[inline]
pub fn second_derivative<T: Numeric, F: ScalarFn>(function: &F, point: T) -> T {
    function.eval(HyperDual::variable(point)).eps1eps2
}

/// One partial derivative of a multi-variable function, picked by variable index.
///
/// For a mixed or higher-order partial, use [`AutoDiffMulti`].
///
/// # Errors
/// [`DiffError::IndexOutOfRange`] if `variable_index` is not a variable of `function`. The index is
/// an ordinary number rather than part of the type, so it is checked when the code runs.
///
/// # Examples
/// ```
/// use multicalc::numerical_derivative::partial;
/// use multicalc::scalar_fn;
/// # fn main() -> Result<(), multicalc::error::DiffError> {
/// // g(x, y) = x^2 * y, so dg/dx at (3, 4) is 2xy = 24
/// let g = scalar_fn!(|v: &[f64; 2]| v[0] * v[0] * v[1]);
/// assert!((partial(&g, 0, &[3.0_f64, 4.0])? - 24.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn partial<T: Numeric, F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize>(
    function: &F,
    variable_index: usize,
    point: &[T; NUM_VARS],
) -> Result<T, DiffError> {
    AutoDiffMulti::<T>::new().get_single_partial(function, variable_index, point)
}
