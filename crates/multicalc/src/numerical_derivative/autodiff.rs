//! Forward-mode automatic-differentiation differentiators.
//!
//! Derivatives are exact (no step size, no truncation error). The differentiation order picks the
//! scalar: [`Dual`] for first derivatives, [`HyperDual`] for second, and [`Jet`] for higher
//! single-variable orders.

use core::marker::PhantomData;

use crate::error::DiffError;
use crate::numerical_derivative::derivator::{DerivatorMultiVariable, DerivatorSingleVariable};
use crate::scalar::{Dual, HyperDual, Jet, Numeric, ScalarFn, ScalarFnN, VectorFn};

/// Highest single-variable derivative order an [`AutoDiffSingle`] supports through its [`Jet`].
const MAX_ORDER: usize = 6;

/// Forward-mode autodiff differentiator for single-variable functions.
///
/// ```
/// use multicalc::numerical_derivative::AutoDiffSingle;
/// use multicalc::numerical_derivative::DerivatorSingleVariable;
/// use multicalc::scalar_fn;
///
/// // f(x) = x^3 -> f' = 3x^2, f'' = 6x, f''' = 6
/// let function = scalar_fn!(|x| x * x * x);
/// let derivator = AutoDiffSingle::default();
/// let point = 2.0_f64;
///
/// // exact to rounding, no step size
/// assert!((derivator.differentiate(1, &function, point).unwrap() - 12.0).abs() < 1e-12);
/// assert!((derivator.differentiate(2, &function, point).unwrap() - 12.0).abs() < 1e-12);
/// assert!((derivator.differentiate(3, &function, point).unwrap() - 6.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoDiffSingle<T = f64> {
    _marker: PhantomData<T>,
}

impl<T> Default for AutoDiffSingle<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AutoDiffSingle<T> {
    /// Const constructor (same as [`Default::default`]).
    ///
    /// ```
    /// use multicalc::numerical_derivative::AutoDiffSingle;
    ///
    /// const D: AutoDiffSingle = AutoDiffSingle::new();
    /// ```
    #[inline]
    pub const fn new() -> Self {
        AutoDiffSingle {
            _marker: PhantomData,
        }
    }
}

impl<T: Numeric> DerivatorSingleVariable for AutoDiffSingle<T> {
    type Scalar = T;

    /// Orders 1 and 2 use [`Dual`]/[`HyperDual`]; orders 3..=`MAX_ORDER` use a [`Jet`]. Higher
    /// orders return [`DiffError::OrderUnsupported`].
    fn differentiate<F: ScalarFn>(&self, order: usize, func: &F, point: T) -> Result<T, DiffError> {
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
/// use multicalc::numerical_derivative::AutoDiffMulti;
/// use multicalc::numerical_derivative::DerivatorMultiVariable;
/// use multicalc::numerical_derivative::FiniteDifferenceMulti;
/// use multicalc::scalar_fn;
///
/// // f(x, y) = x^2 * y + sin(x)
/// let function = scalar_fn!(|v: &[f64; 2]| v[0] * v[0] * v[1] + v[0].sin());
/// let autodiff = AutoDiffMulti::default();
/// let finite_difference = FiniteDifferenceMulti::default();
/// let point = [1.0, 2.0];
/// let x_index = 0;
///
/// // df/dx = 2xy + cos(x): exact, and it agrees with finite differences
/// let slope_x = autodiff.first_partial_derivative(&function, x_index, &point).unwrap();
/// assert!((slope_x - (2.0 * 1.0 * 2.0 + f64::cos(1.0))).abs() < 1e-12);
/// let estimated = finite_difference
///     .first_partial_derivative(&function, x_index, &point)
///     .unwrap();
/// assert!((slope_x - estimated).abs() < 1e-5);
///
/// // mixed second partial d2f/dx dy = 2x
/// let variable_indices = [0, 1];
/// let mixed = autodiff
///     .second_partial_derivative(&function, &variable_indices, &point)
///     .unwrap();
/// assert!((mixed - 2.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoDiffMulti<T = f64> {
    _marker: PhantomData<T>,
}

impl<T> Default for AutoDiffMulti<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AutoDiffMulti<T> {
    /// Const constructor (same as [`Default::default`]).
    ///
    /// ```
    /// use multicalc::numerical_derivative::AutoDiffMulti;
    ///
    /// const D: AutoDiffMulti = AutoDiffMulti::new();
    /// ```
    #[inline]
    pub const fn new() -> Self {
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
    fn differentiate<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize, const NUM_ORDER: usize>(
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

    /// Computes input `col` on a single [`Dual`] direction and evaluates `func` once, reading the
    /// derivative of every output from that one pass.
    /// The whole Jacobian column in O(1) evaluations.
    fn jacobian_column<
        F: VectorFn<NUM_VARS, NUM_FUNCS>,
        const NUM_VARS: usize,
        const NUM_FUNCS: usize,
    >(
        &self,
        func: &F,
        col: usize,
        point: &[T; NUM_VARS],
    ) -> Result<[T; NUM_FUNCS], DiffError> {
        if col >= NUM_VARS {
            return Err(DiffError::IndexOutOfRange);
        }

        let mut seed: [Dual<T>; NUM_VARS] = core::array::from_fn(|k| Dual::constant(point[k]));
        seed[col] = Dual::variable(point[col]);

        let outputs = func.eval(&seed);
        Ok(core::array::from_fn(|m| outputs[m].deriv))
    }
}
