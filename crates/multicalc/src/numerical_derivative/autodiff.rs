//! Forward-mode automatic-differentiation differentiators.
//!
//! Derivatives are exact (no step size, no truncation error). The differentiation order picks the
//! scalar: [`Dual`] for first derivatives, [`HyperDual`] for second, and [`Jet`] for higher
//! single-variable orders.

use core::marker::PhantomData;

use crate::error::DiffError;
use crate::numerical_derivative::derivator::{DerivatorMultiVariable, DerivatorSingleVariable};
use crate::scalar::{Dual, HyperDual, Jet, Numeric, ScalarFn, ScalarFnN};

/// Highest single-variable derivative order an [`AutoDiffSingle`] supports through its [`Jet`].
const MAX_ORDER: usize = 6;

/// Forward-mode autodiff differentiator for single-variable functions.
///
/// ```
/// use multicalc::numerical_derivative::autodiff::AutoDiffSingle;
/// use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
/// use multicalc::scalar_fn;
///
/// // f(x) = x^3 -> f' = 3x^2, f'' = 6x, f''' = 6
/// let f = scalar_fn!(|x| x * x * x);
/// let d = AutoDiffSingle::default();
///
/// // exact to rounding, no step size
/// assert!((d.get(1, &f, 2.0_f64).unwrap() - 12.0).abs() < 1e-12);
/// assert!((d.get(2, &f, 2.0_f64).unwrap() - 12.0).abs() < 1e-12);
/// assert!((d.get(3, &f, 2.0_f64).unwrap() - 6.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AutoDiffSingle<T = f64> {
    _marker: PhantomData<T>,
}

impl<T> Default for AutoDiffSingle<T> {
    fn default() -> Self {
        AutoDiffSingle {
            _marker: PhantomData,
        }
    }
}

impl<T: Numeric> DerivatorSingleVariable for AutoDiffSingle<T> {
    type Scalar = T;

    /// Orders 1 and 2 use [`Dual`]/[`HyperDual`]; orders 3..=`MAX_ORDER` use a [`Jet`]. Higher
    /// orders return [`DiffError::OrderUnsupported`].
    fn get<F: ScalarFn>(&self, order: usize, func: &F, point: T) -> Result<T, DiffError> {
        match order {
            0 => Err(DiffError::OrderZero),
            1 => Ok(func.eval(Dual::variable(point)).deriv),
            2 => Ok(func.eval(HyperDual::variable(point)).eps1eps2),
            o if o <= MAX_ORDER => Ok(func
                .eval(Jet::<T, { MAX_ORDER + 1 }>::variable(point))
                .derivative(o)),
            _ => Err(DiffError::OrderUnsupported),
        }
    }
}

/// Forward-mode autodiff differentiator for multi-variable functions.
///
/// ```
/// use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
/// use multicalc::numerical_derivative::derivator::DerivatorMultiVariable;
/// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
/// use multicalc::scalar_fn;
///
/// // f(x, y) = x^2 * y + sin(x)
/// let f = scalar_fn!(|v: &[f64; 2]| v[0] * v[0] * v[1] + v[0].sin());
/// let ad = AutoDiffMulti::default();
/// let fd = FiniteDifferenceMulti::default();
/// let point = [1.0, 2.0];
///
/// // df/dx = 2xy + cos(x): exact, and it agrees with finite differences
/// let ad_dx = ad.get_single_partial(&f, 0, &point).unwrap();
/// assert!((ad_dx - (2.0 * 1.0 * 2.0 + f64::cos(1.0))).abs() < 1e-12);
/// assert!((ad_dx - fd.get_single_partial(&f, 0, &point).unwrap()).abs() < 1e-5);
///
/// // mixed second partial d2f/dx dy = 2x
/// assert!((ad.get_double_partial(&f, &[0, 1], &point).unwrap() - 2.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AutoDiffMulti<T = f64> {
    _marker: PhantomData<T>,
}

impl<T> Default for AutoDiffMulti<T> {
    fn default() -> Self {
        AutoDiffMulti {
            _marker: PhantomData,
        }
    }
}

impl<T: Numeric> DerivatorMultiVariable for AutoDiffMulti<T> {
    type Scalar = T;

    /// First partials use [`Dual`], second (mixed) use [`HyperDual`], and third use a nested
    /// `Dual<HyperDual>` (three independent directions). Orders beyond 3 return
    /// [`DiffError::OrderUnsupported`] (use [`AutoDiffSingle`] for high single-variable
    /// orders, or a finite-difference differentiator).
    fn get<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[T; NUM_VARS],
    ) -> Result<T, DiffError> {
        if NUM_ORDER == 0 {
            return Err(DiffError::OrderZero);
        }
        for &idx in idx_to_differentiate {
            if idx >= NUM_VARS {
                return Err(DiffError::IndexOutOfRange);
            }
        }

        match NUM_ORDER {
            1 => {
                let i = idx_to_differentiate[0];
                let mut seed: [Dual<T>; NUM_VARS] =
                    core::array::from_fn(|k| Dual::constant(point[k]));
                seed[i] = Dual::variable(point[i]);
                Ok(func.eval(&seed).deriv)
            }
            2 => {
                let i = idx_to_differentiate[0];
                let j = idx_to_differentiate[1];
                let mut seed: [HyperDual<T>; NUM_VARS] =
                    core::array::from_fn(|k| HyperDual::constant(point[k]));
                // index i moves along direction 1, index j along direction 2; if i == j this seeds
                // both directions of the same variable, giving the pure second derivative.
                seed[i].eps1 = T::ONE;
                seed[j].eps2 = T::ONE;
                Ok(func.eval(&seed).eps1eps2)
            }
            3 => {
                let i = idx_to_differentiate[0];
                let j = idx_to_differentiate[1];
                let k = idx_to_differentiate[2];
                // three independent directions: the two HyperDual epsilons plus the outer Dual.
                // equal indices just seed the same variable on more than one direction.
                let seed: [Dual<HyperDual<T>>; NUM_VARS] = core::array::from_fn(|m| {
                    let a = if m == i { T::ONE } else { T::ZERO };
                    let b = if m == j { T::ONE } else { T::ZERO };
                    let c = if m == k { T::ONE } else { T::ZERO };
                    Dual::new(
                        HyperDual::new(point[m], a, b, T::ZERO),
                        HyperDual::new(c, T::ZERO, T::ZERO, T::ZERO),
                    )
                });
                Ok(func.eval(&seed).deriv.eps1eps2)
            }
            _ => Err(DiffError::OrderUnsupported),
        }
    }
}
