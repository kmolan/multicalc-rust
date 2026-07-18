//! Finite-difference differentiators.
//!
//! Accuracy falls with derivative order: repeated differencing amplifies rounding
//! error (roughly as the inverse of the step size raised to the order). For third
//! derivatives and higher, tune `step_size` and `step_size_multiplier` per problem.

use crate::error::DiffError;
use crate::numerical_derivative::derivator::{DerivatorMultiVariable, DerivatorSingleVariable};
use crate::numerical_derivative::mode::{self, FiniteDifferenceMode};
use crate::scalar::{Numeric, ScalarFn, ScalarFnN, VectorFn};

/// Low and high sample offsets (in units of the step size) and the divisor factor
/// for each finite-difference mode.
#[inline]
fn offsets<T: Numeric>(method: FiniteDifferenceMode) -> (T, T, T) {
    match method {
        FiniteDifferenceMode::Forward => (T::ZERO, T::ONE, T::ONE),
        FiniteDifferenceMode::Backward => (-T::ONE, T::ZERO, T::ONE),
        FiniteDifferenceMode::Central => (-T::ONE, T::ONE, T::TWO),
    }
}

/// Configuration shared by the single- and multi-variable finite-difference differentiators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteDifferenceConfig<T = f64> {
    /// The finite-difference step size. See [`mode::DEFAULT_STEP_SIZE`].
    pub step_size: T,
    /// Forward, Backward or Central difference.
    pub method: FiniteDifferenceMode,
    /// Factor the step is scaled by on each recursion level; only matters for third
    /// derivatives and higher. See [`mode::DEFAULT_STEP_SIZE_MULTIPLIER`].
    pub step_size_multiplier: T,
}

impl<T: Numeric> Default for FiniteDifferenceConfig<T> {
    /// Central difference with the default step size and multiplier; best for most cases.
    fn default() -> Self {
        FiniteDifferenceConfig {
            step_size: T::from_f64(mode::DEFAULT_STEP_SIZE),
            method: FiniteDifferenceMode::Central,
            step_size_multiplier: T::from_f64(mode::DEFAULT_STEP_SIZE_MULTIPLIER),
        }
    }
}

impl<T: Numeric> FiniteDifferenceConfig<T> {
    /// Builds a config with explicit parameters.
    pub fn from_parameters(step: T, method: FiniteDifferenceMode, multiplier: T) -> Self {
        FiniteDifferenceConfig {
            step_size: step,
            method,
            step_size_multiplier: multiplier,
        }
    }

    /// Returns [`DiffError::StepSizeZero`] if the step size is zero.
    fn check_step_size(&self) -> Result<(), DiffError> {
        if self.step_size == T::ZERO {
            return Err(DiffError::StepSizeZero);
        }
        Ok(())
    }
}

/// Finite-difference differentiator for single-variable functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteDifferenceSingle<T = f64> {
    pub config: FiniteDifferenceConfig<T>,
}

impl<T: Numeric> Default for FiniteDifferenceSingle<T> {
    fn default() -> Self {
        FiniteDifferenceSingle {
            config: FiniteDifferenceConfig::default(),
        }
    }
}

impl<T: Numeric> FiniteDifferenceSingle<T> {
    /// Builds a differentiator with explicit parameters.
    pub fn from_parameters(step: T, method: FiniteDifferenceMode, multiplier: T) -> Self {
        FiniteDifferenceSingle {
            config: FiniteDifferenceConfig::from_parameters(step, method, multiplier),
        }
    }

    #[inline]
    fn diff<F: ScalarFn>(&self, order: usize, func: &F, point: T, step: T) -> T {
        let (lo, hi, denom) = offsets::<T>(self.config.method);

        if order == 1 {
            let low = func.eval(point + lo * step);
            let high = func.eval(point + hi * step);
            return (high - low) / (denom * step);
        }

        let next = self.config.step_size_multiplier * step;
        let low = self.diff(order - 1, func, point + lo * step, next);
        let high = self.diff(order - 1, func, point + hi * step, next);
        (high - low) / (denom * step)
    }
}

impl<T: Numeric> DerivatorSingleVariable for FiniteDifferenceSingle<T> {
    type Scalar = T;

    fn get<F: ScalarFn>(&self, order: usize, func: &F, point: T) -> Result<T, DiffError> {
        if order == 0 {
            return Err(DiffError::OrderZero);
        }
        self.config.check_step_size()?;
        Ok(self.diff(order, func, point, self.config.step_size))
    }
}

/// Finite-difference differentiator for multi-variable functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteDifferenceMulti<T = f64> {
    pub config: FiniteDifferenceConfig<T>,
}

impl<T: Numeric> Default for FiniteDifferenceMulti<T> {
    fn default() -> Self {
        FiniteDifferenceMulti {
            config: FiniteDifferenceConfig::default(),
        }
    }
}

impl<T: Numeric> FiniteDifferenceMulti<T> {
    /// Builds a differentiator with explicit parameters.
    pub fn from_parameters(step: T, method: FiniteDifferenceMode, multiplier: T) -> Self {
        FiniteDifferenceMulti {
            config: FiniteDifferenceConfig::from_parameters(step, method, multiplier),
        }
    }

    #[inline]
    fn diff<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[T; NUM_VARS],
        step: T,
    ) -> T {
        let (lo, hi, denom) = offsets::<T>(self.config.method);
        let var = idx_to_differentiate[order - 1];

        let mut low_point = *point;
        low_point[var] += lo * step;
        let mut high_point = *point;
        high_point[var] += hi * step;

        if order == 1 {
            return (func.eval(&high_point) - func.eval(&low_point)) / (denom * step);
        }

        let next = self.config.step_size_multiplier * step;
        let low = self.diff(order - 1, func, idx_to_differentiate, &low_point, next);
        let high = self.diff(order - 1, func, idx_to_differentiate, &high_point, next);
        (high - low) / (denom * step)
    }
}

impl<T: Numeric> DerivatorMultiVariable for FiniteDifferenceMulti<T> {
    type Scalar = T;

    fn get<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[T; NUM_VARS],
    ) -> Result<T, DiffError> {
        if NUM_ORDER == 0 {
            return Err(DiffError::OrderZero);
        }
        self.config.check_step_size()?;
        for &idx in idx_to_differentiate {
            if idx >= NUM_VARS {
                return Err(DiffError::IndexOutOfRange);
            }
        }
        Ok(self.diff(
            NUM_ORDER,
            func,
            idx_to_differentiate,
            point,
            self.config.step_size,
        ))
    }

    /// Evaluates the whole vector function at the low and high
    /// sample points, reading every output's difference quotient from those two passes,
    /// getting the whole Jacobian column in two evaluations.
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
        self.config.check_step_size()?;
        if col >= NUM_VARS {
            return Err(DiffError::IndexOutOfRange);
        }

        let (lo, hi, denom) = offsets::<T>(self.config.method);
        let step = self.config.step_size;

        let mut low_point = *point;
        low_point[col] += lo * step;
        let mut high_point = *point;
        high_point[col] += hi * step;

        let low = func.eval(&low_point);
        let high = func.eval(&high_point);

        Ok(core::array::from_fn(|m| {
            (high[m] - low[m]) / (denom * step)
        }))
    }
}
