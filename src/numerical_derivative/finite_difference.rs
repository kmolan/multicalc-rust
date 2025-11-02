use crate::numerical_derivative::derivator::*;
use crate::numerical_derivative::mode;
use crate::utils::error_codes::*;

/// @brief Implements the finite difference method for numerical differentiation
/// of single-variable functions.
///
/// This solver computes first and higher-order derivatives using finite
/// difference schemes: Forward, Backward, and Central.
#[derive(Clone, Copy)]
pub struct SingleVariableSolver {
    step_size: f64,
    method: mode::FiniteDifferenceMode,

    /// @brief The step size multiplier.
    ///
    /// The step size will be multiplied by this factor after each iteration.
    /// Only relevant for third or higher-order derivatives.
    step_size_multiplier: f64,
}

impl Default for SingleVariableSolver {
    /// @brief Default constructor, optimal for most equations.
    fn default() -> Self {
        SingleVariableSolver {
            step_size: mode::DEFAULT_STEP_SIZE,
            method: mode::FiniteDifferenceMode::Central,
            step_size_multiplier: mode::DEFAULT_STEP_SIZE_MULTIPLIER,
        }
    }
}

impl SingleVariableSolver {
    /// @brief Returns the current step size.
    pub fn get_step_size(&self) -> f64 {
        self.step_size
    }

    /// @brief Sets the step size.
    pub fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    /// @brief Returns the chosen method of differentiation.
    ///
    /// @note Possible choices are: Forward step, Backward step, and Central step.
    pub fn get_method(&self) -> mode::FiniteDifferenceMode {
        self.method
    }

    /// @brief Sets the method of differentiation.
    ///
    /// @note Possible choices are: Forward step, Backward step, and Central step.
    pub fn set_method(&mut self, method: mode::FiniteDifferenceMode) {
        self.method = method;
    }

    /// @brief Returns the chosen step size multiplier.
    pub fn get_step_size_multiplier(&self) -> f64 {
        self.step_size_multiplier
    }

    /// @brief Sets the step size multiplier.
    ///
    /// @note This parameter only matters if you are interested in triple derivatives or higher.
    pub fn set_step_size_multiplier(&mut self, multiplier: f64) {
        self.step_size_multiplier = multiplier;
    }

    /// @brief Custom constructor for configuring solver parameters.
    ///
    /// @param step The desired step size for each iteration.
    /// @param method The method of differentiation (Forward, Backward, or Central).
    /// @param multiplier The step size multiplier. Default is 10.0.
    ///
    /// @note The multiplier only matters for triple derivatives or higher.
    pub fn from_parameters(step: f64, method: mode::FiniteDifferenceMode, multiplier: f64) -> Self {
        SingleVariableSolver {
            step_size: step,
            method,
            step_size_multiplier: multiplier,
        }
    }

    /// @brief Computes the forward difference for single-variable functions.
    ///
    /// Computes f'(x) = (f(x + h) - f(x)) / h, where h is the chosen step size.
    ///
    /// @param order The number of derivative steps to compute.
    /// @param func The target function to differentiate.
    /// @param point The point at which to evaluate the derivative.
    /// @param step_size The step size used for the difference computation.
    ///
    /// @return The computed numerical derivative.
    fn get_forward_difference_single_variable(
        &self,
        order: usize,
        func: &dyn Fn(f64) -> f64,
        point: f64,
        step_size: f64,
    ) -> f64 {
        if order == 1 {
            let f0 = func(point);
            let f1 = func(point + step_size);
            return (f1 - f0) / step_size;
        }

        let f0 = self.get_forward_difference_single_variable(
            order - 1,
            func,
            point,
            self.step_size_multiplier * step_size,
        );
        let f1 = self.get_forward_difference_single_variable(
            order - 1,
            func,
            point + step_size,
            self.step_size_multiplier * step_size,
        );
        (f1 - f0) / step_size
    }

    /// @brief Computes the backward difference for single-variable functions.
    ///
    /// Computes f'(x) = (f(x) - f(x - h)) / h, where h is the chosen step size.
    ///
    /// @param order The number of derivative steps to compute.
    /// @param func The target function to differentiate.
    /// @param point The point at which to evaluate the derivative.
    /// @param step_size The step size used for the difference computation.
    ///
    /// @return The computed numerical derivative.
    fn get_backward_difference_single_variable(
        &self,
        order: usize,
        func: &dyn Fn(f64) -> f64,
        point: f64,
        step_size: f64,
    ) -> f64 {
        if order == 1 {
            let f0 = func(point - step_size);
            let f1 = func(point);
            return (f1 - f0) / step_size;
        }

        let f0 = self.get_backward_difference_single_variable(
            order - 1,
            func,
            point - step_size,
            self.step_size_multiplier * step_size,
        );
        let f1 = self.get_backward_difference_single_variable(
            order - 1,
            func,
            point,
            self.step_size_multiplier * step_size,
        );
        (f1 - f0) / step_size
    }

    /// @brief Computes the central difference for single-variable functions.
    ///
    /// Computes f'(x) = (f(x + h) - f(x - h)) / (2h), where h is the chosen step size.
    ///
    /// @param order The number of derivative steps to compute.
    /// @param func The target function to differentiate.
    /// @param point The point at which to evaluate the derivative.
    /// @param step_size The step size used for the difference computation.
    ///
    /// @return The computed numerical derivative.
    fn get_central_difference_single_variable(
        &self,
        order: usize,
        func: &dyn Fn(f64) -> f64,
        point: f64,
        step_size: f64,
    ) -> f64 {
        if order == 1 {
            let f0 = func(point - step_size);
            let f1 = func(point + step_size);
            return (f1 - f0) / (2.0 * step_size);
        }

        let f0 = self.get_central_difference_single_variable(
            order - 1,
            func,
            point - step_size,
            self.step_size_multiplier * step_size,
        );
        let f1 = self.get_central_difference_single_variable(
            order - 1,
            func,
            point + step_size,
            self.step_size_multiplier * step_size,
        );

        (f1 - f0) / (2.0 * step_size)
    }
}

impl DerivatorSingleVariable for SingleVariableSolver {
    /// @brief Returns the numerical differentiation value for a single-variable function.
    ///
    /// @param order Number of times the equation should be differentiated.
    /// @param func The single-variable function.
    /// @param point The point of interest around which we want to differentiate.
    ///
    /// @return Result<f64, &'static str> The computed derivative or an error code.
    ///
    /// @note Possible error codes:
    /// - `NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO`: If the step size is zero.
    /// - `DERIVATE_ORDER_CANNOT_BE_ZERO`: If the `order` argument is zero.
    ///
    /// @example
    /// ```
    /// use multicalc::numerical_derivative::derivator::*;
    /// use multicalc::numerical_derivative::finite_difference::*;
    ///
    /// let my_func = |x: f64| x * x * x;
    /// let point = 2.0;
    /// let derivator = SingleVariableSolver::default();
    /// let val = derivator.get(1, &my_func, point).unwrap();
    /// assert!(f64::abs(val - 12.0) < 1e-7);
    /// ```
    fn get(
        &self,
        order: usize,
        func: &dyn Fn(f64) -> f64,
        point: f64,
    ) -> Result<f64, &'static str> {
        if order == 0 {
            return Err(DERIVATE_ORDER_CANNOT_BE_ZERO);
        }
        if self.step_size == 0.0 {
            return Err(NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO);
        }

        match self.method {
            mode::FiniteDifferenceMode::Forward => {
                Ok(self.get_forward_difference_single_variable(order, func, point, self.step_size))
            }
            mode::FiniteDifferenceMode::Backward => Ok(
                self.get_backward_difference_single_variable(order, func, point, self.step_size)
            ),
            mode::FiniteDifferenceMode::Central => {
                Ok(self.get_central_difference_single_variable(order, func, point, self.step_size))
            }
        }
    }
}

/// @brief Implements the finite difference method for numerical differentiation
/// of multi-variable functions.
#[derive(Clone, Copy)]
pub struct MultiVariableSolver {
    step_size: f64,
    method: mode::FiniteDifferenceMode,

    /// @brief Step size multiplier.
    ///
    /// The step size will be multiplied by this factor after every iteration.
    /// Only matters for triple derivatives or higher.
    step_size_multiplier: f64,
}

impl Default for MultiVariableSolver {
    /// @brief Default constructor, optimal for most equations.
    fn default() -> Self {
        MultiVariableSolver {
            step_size: mode::DEFAULT_STEP_SIZE,
            method: mode::FiniteDifferenceMode::Central,
            step_size_multiplier: mode::DEFAULT_STEP_SIZE_MULTIPLIER,
        }
    }
}

impl MultiVariableSolver {
    /// @brief Returns the step size.
    pub fn get_step_size(&self) -> f64 {
        self.step_size
    }

    /// @brief Sets the step size.
    pub fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    /// @brief Returns the chosen method of differentiation.
    ///
    /// @note Possible choices: Forward, Backward, or Central.
    pub fn get_method(&self) -> mode::FiniteDifferenceMode {
        self.method
    }

    /// @brief Sets the method of differentiation.
    ///
    /// @note Possible choices: Forward, Backward, or Central.
    pub fn set_method(&mut self, method: mode::FiniteDifferenceMode) {
        self.method = method;
    }

    /// @brief Returns the chosen step size multiplier.
    pub fn get_step_size_multiplier(&self) -> f64 {
        self.step_size_multiplier
    }

    /// @brief Sets the step size multiplier.
    ///
    /// @note This parameter only matters for triple derivatives or higher.
    pub fn set_step_size_multiplier(&mut self, multiplier: f64) {
        self.step_size_multiplier = multiplier;
    }

    /// @brief Custom constructor for tuning parameters.
    ///
    /// @param step The desired step size.
    /// @param method The differentiation method (Forward, Backward, Central).
    /// @param multiplier The factor by which to multiply the step size on each iteration.
    ///
    /// @note Only matters for triple derivatives or higher.
    pub fn from_parameters(step: f64, method: mode::FiniteDifferenceMode, multiplier: f64) -> Self {
        MultiVariableSolver {
            step_size: step,
            method,
            step_size_multiplier: multiplier,
        }
    }

    /// @brief Returns the partial forward difference for multi-variable functions.
    ///
    /// Computes f'(X) = (f(X + h) - f(X)) / h, where h is the chosen step size.
    ///
    /// @param order The derivative order.
    /// @param func The multi-variable function.
    /// @param idx_to_derivate Array of variable indices to differentiate with respect to.
    /// @param point The evaluation point.
    /// @param step_size The step size.
    ///
    /// @return The computed numerical derivative.
    fn get_forward_difference_multi_variable<const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
        step_size: f64,
    ) -> f64 {
        if order == 1 {
            let f0_args = point;
            let mut f1_args = *point;
            f1_args[idx_to_derivate[0]] += step_size;

            let f0 = func(f0_args);
            let f1 = func(&f1_args);

            return (f1 - f0) / step_size;
        }

        let mut f1_args = *point;
        f1_args[idx_to_derivate[order - 1]] += step_size;

        let f0 = self.get_forward_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            point,
            self.step_size_multiplier * step_size,
        );
        let f1 = self.get_forward_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            &f1_args,
            self.step_size_multiplier * step_size,
        );

        (f1 - f0) / step_size
    }

    /// @brief Returns the partial backward difference for multi-variable functions.
    ///
    /// Computes f'(X) = (f(X) - f(X - h)) / h, where h is the chosen step size.
    ///
    /// @param order The derivative order.
    /// @param func The multi-variable function.
    /// @param idx_to_derivate Array of variable indices to differentiate with respect to.
    /// @param point The evaluation point.
    /// @param step_size The step size.
    ///
    /// @return The computed numerical derivative.
    fn get_backward_difference_multi_variable<const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
        step_size: f64,
    ) -> f64 {
        if order == 1 {
            let mut f0_args = *point;
            f0_args[idx_to_derivate[0]] -= step_size;
            let f1_args = point;

            let f0 = func(&f0_args);
            let f1 = func(f1_args);

            return (f1 - f0) / step_size;
        }

        let mut f0_args = *point;
        f0_args[idx_to_derivate[order - 1]] -= step_size;

        let f0 = self.get_backward_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            &f0_args,
            self.step_size_multiplier * step_size,
        );
        let f1 = self.get_backward_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            point,
            self.step_size_multiplier * step_size,
        );

        (f1 - f0) / step_size
    }

    /// @brief Returns the partial central difference for multi-variable functions.
    ///
    /// Computes f'(X) = (f(X + h) - f(X - h)) / (2h), where h is the chosen step size.
    ///
    /// @param order The derivative order.
    /// @param func The multi-variable function.
    /// @param idx_to_derivate Array of variable indices to differentiate with respect to.
    /// @param point The evaluation point.
    /// @param step_size The step size.
    ///
    /// @return The computed numerical derivative.
    fn get_central_difference_multi_variable<const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
        step_size: f64,
    ) -> f64 {
        if order == 1 {
            let mut f0_args = *point;
            f0_args[idx_to_derivate[0]] -= step_size;

            let mut f1_args = *point;
            f1_args[idx_to_derivate[0]] += step_size;

            let f0 = func(&f0_args);
            let f1 = func(&f1_args);

            return (f1 - f0) / (2.0 * step_size);
        }

        let mut f0_point = *point;
        f0_point[idx_to_derivate[order - 1]] -= step_size;

        let f0 = self.get_central_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            &f0_point,
            self.step_size_multiplier * step_size,
        );

        let mut f1_point = *point;
        f1_point[idx_to_derivate[order - 1]] += step_size;

        let f1 = self.get_central_difference_multi_variable(
            order - 1,
            func,
            idx_to_derivate,
            &f1_point,
            self.step_size_multiplier * step_size,
        );

        (f1 - f0) / (2.0 * step_size)
    }
}

impl DerivatorMultiVariable for MultiVariableSolver {
    /// @brief Returns the numerical differentiation value for a multi-variable function.
    ///
    /// @param order The number of times the equation should be differentiated.
    /// @param func The multi-variable function.
    /// @param idx_to_derivate The variable indices with respect to which differentiation occurs.
    /// @param point The point of interest where the derivative is computed.
    ///
    /// @return Result<f64, &'static str> The computed derivative or an error code.
    ///
    /// @note Possible error codes:
    /// - `NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO`: Step size is zero.
    /// - `DERIVATE_ORDER_CANNOT_BE_ZERO`: Order is zero.
    /// - `INDEX_TO_DERIVATE_ILL_FORMED`: Size of `idx_to_derivate` does not match `order`.
    /// - `INDEX_TO_DERIVATIVE_OUT_OF_RANGE`: Variable index exceeds the function dimension.
    ///
    /// @example
    /// ```
    /// use multicalc::numerical_derivative::derivator::*;
    /// use multicalc::numerical_derivative::finite_difference::*;
    ///
    /// let my_func = |args: &[f64; 3]| -> f64 {
    ///     args[1] * args[0].sin() + args[0] * args[1].cos() + args[0] * args[1] * args[2].exp()
    /// };
    /// let point = [1.0, 2.0, 3.0];
    /// let derivator = MultiVariableSolver::default();
    /// let idx: [usize; 2] = [0, 1];
    /// let val = derivator.get(2, &my_func, &idx, &point).unwrap();
    /// ```
    fn get<const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str> {
        if self.step_size == 0.0 {
            return Err(NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO);
        }
        if order == 0 {
            return Err(DERIVATE_ORDER_CANNOT_BE_ZERO);
        }
        if order != NUM_ORDER {
            return Err(INDEX_TO_DERIVATE_ILL_FORMED);
        }

        for &idx in idx_to_derivate {
            if idx >= point.len() {
                return Err(INDEX_TO_DERIVATIVE_OUT_OF_RANGE);
            }
        }

        match self.method {
            mode::FiniteDifferenceMode::Forward => Ok(self.get_forward_difference_multi_variable(
                order,
                func,
                idx_to_derivate,
                point,
                self.step_size,
            )),
            mode::FiniteDifferenceMode::Backward => Ok(self
                .get_backward_difference_multi_variable(
                    order,
                    func,
                    idx_to_derivate,
                    point,
                    self.step_size,
                )),
            mode::FiniteDifferenceMode::Central => Ok(self.get_central_difference_multi_variable(
                order,
                func,
                idx_to_derivate,
                point,
                self.step_size,
            )),
        }
    }
}
