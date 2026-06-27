//! Finite-difference differentiators.
//!
//! Accuracy falls with derivative order: repeated differencing amplifies rounding
//! error (roughly as the inverse of the step size raised to the order). For third
//! derivatives and higher, tune `step_size` and `step_size_multiplier` per problem.

use crate::numerical_derivative::derivator::{DerivatorMultiVariable, DerivatorSingleVariable};
use crate::numerical_derivative::mode::{self, FiniteDifferenceMode};
use crate::utils::error_codes::CalcError;

/// Low and high sample offsets (in units of the step size) and the divisor factor
/// for each finite-difference mode.
#[inline]
fn offsets(method: FiniteDifferenceMode) -> (f64, f64, f64) {
    match method {
        FiniteDifferenceMode::Forward => (0.0, 1.0, 1.0),
        FiniteDifferenceMode::Backward => (-1.0, 0.0, 1.0),
        FiniteDifferenceMode::Central => (-1.0, 1.0, 2.0),
    }
}

/// Finite-difference differentiator for single-variable functions.
#[derive(Clone, Copy)]
pub struct FiniteDifferenceSingle {
    step_size: f64,
    method: FiniteDifferenceMode,
    step_size_multiplier: f64,
}

impl Default for FiniteDifferenceSingle {
    fn default() -> Self {
        FiniteDifferenceSingle {
            step_size: mode::DEFAULT_STEP_SIZE,
            method: FiniteDifferenceMode::Central,
            step_size_multiplier: mode::DEFAULT_STEP_SIZE_MULTIPLIER,
        }
    }
}

impl FiniteDifferenceSingle {
    /// Returns the step size.
    pub fn get_step_size(&self) -> f64 {
        self.step_size
    }

    /// Sets the step size.
    pub fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    /// Returns the differentiation method.
    pub fn get_method(&self) -> FiniteDifferenceMode {
        self.method
    }

    /// Sets the differentiation method.
    pub fn set_method(&mut self, method: FiniteDifferenceMode) {
        self.method = method;
    }

    /// Returns the step-size multiplier applied on each recursion level.
    pub fn get_step_size_multiplier(&self) -> f64 {
        self.step_size_multiplier
    }

    /// Sets the step-size multiplier. Only matters for third derivatives and higher.
    pub fn set_step_size_multiplier(&mut self, multiplier: f64) {
        self.step_size_multiplier = multiplier;
    }

    /// Builds a differentiator with explicit parameters.
    pub fn from_parameters(step: f64, method: FiniteDifferenceMode, multiplier: f64) -> Self {
        FiniteDifferenceSingle {
            step_size: step,
            method,
            step_size_multiplier: multiplier,
        }
    }

    #[inline]
    fn diff<F: Fn(f64) -> f64>(&self, order: usize, func: &F, point: f64, step: f64) -> f64 {
        let (lo, hi, denom) = offsets(self.method);

        if order == 1 {
            let low = func(point + lo * step);
            let high = func(point + hi * step);
            return (high - low) / (denom * step);
        }

        let next = self.step_size_multiplier * step;
        let low = self.diff(order - 1, func, point + lo * step, next);
        let high = self.diff(order - 1, func, point + hi * step, next);
        (high - low) / (denom * step)
    }
}

impl DerivatorSingleVariable for FiniteDifferenceSingle {
    fn get<F: Fn(f64) -> f64>(
        &self,
        order: usize,
        func: &F,
        point: f64,
    ) -> Result<f64, CalcError> {
        if order == 0 {
            return Err(CalcError::DerivativeOrderZero);
        }
        if self.step_size == 0.0 {
            return Err(CalcError::StepSizeZero);
        }
        Ok(self.diff(order, func, point, self.step_size))
    }
}

/// Finite-difference differentiator for multi-variable functions.
#[derive(Clone, Copy)]
pub struct FiniteDifferenceMulti {
    step_size: f64,
    method: FiniteDifferenceMode,
    step_size_multiplier: f64,
}

impl Default for FiniteDifferenceMulti {
    fn default() -> Self {
        FiniteDifferenceMulti {
            step_size: mode::DEFAULT_STEP_SIZE,
            method: FiniteDifferenceMode::Central,
            step_size_multiplier: mode::DEFAULT_STEP_SIZE_MULTIPLIER,
        }
    }
}

impl FiniteDifferenceMulti {
    /// Returns the step size.
    pub fn get_step_size(&self) -> f64 {
        self.step_size
    }

    /// Sets the step size.
    pub fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    /// Returns the differentiation method.
    pub fn get_method(&self) -> FiniteDifferenceMode {
        self.method
    }

    /// Sets the differentiation method.
    pub fn set_method(&mut self, method: FiniteDifferenceMode) {
        self.method = method;
    }

    /// Returns the step-size multiplier applied on each recursion level.
    pub fn get_step_size_multiplier(&self) -> f64 {
        self.step_size_multiplier
    }

    /// Sets the step-size multiplier. Only matters for third derivatives and higher.
    pub fn set_step_size_multiplier(&mut self, multiplier: f64) {
        self.step_size_multiplier = multiplier;
    }

    /// Builds a differentiator with explicit parameters.
    pub fn from_parameters(step: f64, method: FiniteDifferenceMode, multiplier: f64) -> Self {
        FiniteDifferenceMulti {
            step_size: step,
            method,
            step_size_multiplier: multiplier,
        }
    }

    #[inline]
    fn diff<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
        step: f64,
    ) -> f64 {
        let (lo, hi, denom) = offsets(self.method);
        let var = idx_to_differentiate[order - 1];

        let mut low_point = *point;
        low_point[var] += lo * step;
        let mut high_point = *point;
        high_point[var] += hi * step;

        if order == 1 {
            return (func(&high_point) - func(&low_point)) / (denom * step);
        }

        let next = self.step_size_multiplier * step;
        let low = self.diff(order - 1, func, idx_to_differentiate, &low_point, next);
        let high = self.diff(order - 1, func, idx_to_differentiate, &high_point, next);
        (high - low) / (denom * step)
    }
}

impl DerivatorMultiVariable for FiniteDifferenceMulti {
    fn get<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, CalcError> {
        if NUM_ORDER == 0 {
            return Err(CalcError::DerivativeOrderZero);
        }
        if self.step_size == 0.0 {
            return Err(CalcError::StepSizeZero);
        }
        for &idx in idx_to_differentiate {
            if idx >= NUM_VARS {
                return Err(CalcError::IndexOutOfRange);
            }
        }
        Ok(self.diff(NUM_ORDER, func, idx_to_differentiate, point, self.step_size))
    }
}
