//! Differentiation: exact automatic differentiation and finite differences.
//!
//! - [`derivative`] / [`second_derivative`] / [`partial`] — the short way to take one derivative.
//! - [`AutoDiffSingle`] / [`FiniteDifferenceSingle`] and their multi-variable siblings — the two
//!   backends behind [`DerivatorSingleVariable`] and [`DerivatorMultiVariable`] (autodiff is exact).
//! - [`Jacobian`] / [`Hessian`] — derivative matrices of vector- and scalar-valued functions.

mod autodiff;
mod derivator;
mod finite_difference;
mod hessian;
mod jacobian;
mod mode;

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
/// let function = scalar_fn!(|x| x * x * x);   // f(x) = x^3
/// let point = 2.0_f64;
///
/// let slope = derivative(&function, point);   // f'(2) = 3x^2 = 12
/// assert!((slope - 12.0).abs() < 1e-12);
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
/// let function = scalar_fn!(|x| x * x * x);   // f(x) = x^3
/// let point = 2.0_f32;                        // f32 works the same way
///
/// let bend = second_derivative(&function, point);   // f''(2) = 6x = 12
/// assert!((bend - 12.0).abs() < 1e-4);
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
/// let function = scalar_fn!(|point: &[f64; 2]| point[0] * point[0] * point[1]);   // g(x, y) = x^2 * y
/// let variable_index = 0;                                         // differentiate by x
/// let point = [3.0_f64, 4.0];
///
/// let slope = partial(&function, variable_index, &point)?;        // dg/dx = 2xy = 24
/// assert!((slope - 24.0).abs() < 1e-12);
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn partial<T: Numeric, F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize>(
    function: &F,
    variable_index: usize,
    point: &[T; NUM_VARS],
) -> Result<T, DiffError> {
    AutoDiffMulti::<T>::new().first_partial_derivative(function, variable_index, point)
}
