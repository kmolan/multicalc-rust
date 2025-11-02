use crate::gaussian_tables::gauss_tables;
use crate::numerical_integration::integrator::*;
use crate::numerical_integration::mode::GaussianQuadratureMethod;
use crate::utils::error_codes::*;

pub const DEFAULT_QUADRATURE_ORDERS: usize = 4;

/// @brief Implements the Gaussian quadrature methods for numerical integration for single-variable functions.
#[derive(Clone, Copy)]
pub struct SingleVariableSolver {
    order: usize,
    integration_method: GaussianQuadratureMethod,
}

impl Default for SingleVariableSolver {
    /// @brief Default constructor, optimal for most generic polynomial equations.
    fn default() -> Self {
        SingleVariableSolver {
            order: DEFAULT_QUADRATURE_ORDERS,
            integration_method: GaussianQuadratureMethod::GaussLegendre,
        }
    }
}

impl SingleVariableSolver {
    /// @brief Returns the chosen number of nodes/order for quadrature.
    /// @return The number of nodes (order) used for quadrature.
    pub fn get_order(&self) -> usize {
        self.order
    }

    /// @brief Sets the number of nodes/order for quadrature.
    /// @param order The desired quadrature order (1–MAX_GAUSS_TABLE_ORDER).
    pub fn set_order(&mut self, order: usize) {
        self.order = order;
    }

    /// @brief Returns the chosen integration method.
    /// @note Possible choices are `GaussLegendre`, `GaussHermite`, and `GaussLaguerre`.
    pub fn get_integration_method(&self) -> GaussianQuadratureMethod {
        self.integration_method
    }

    /// @brief Sets the integration method.
    pub fn set_integration_method(&mut self, integration_method: GaussianQuadratureMethod) {
        self.integration_method = integration_method;
    }

    /// @brief Creates a solver with custom parameters.
    /// @param order The quadrature order (number of nodes).
    /// @param integration_method The numerical method (`GaussLegendre`, `GaussHermite`, or `GaussLaguerre`).
    /// @return A configured [`SingleVariableSolver`] instance.
    pub fn from_parameters(order: usize, integration_method: GaussianQuadratureMethod) -> Self {
        SingleVariableSolver {
            order,
            integration_method,
        }
    }

    /// @brief Validates the input parameters and integration limits.
    ///
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations The number of integrations to perform.
    /// @param integration_limit Integration limits for each integration stage.
    /// @return `Ok(())` if valid; otherwise an error string.
    fn check_for_errors<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<(), &'static str> {
        if !(1..=gauss_tables::MAX_GAUSS_TABLE_ORDER).contains(&self.order) {
            return Err(GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE);
        }

        for &limit in integration_limit {
            if limit[0] >= limit[1] {
                return Err(INTEGRATION_LIMITS_ILL_DEFINED);
            }
        }

        if NUM_INTEGRATIONS != number_of_integrations {
            return Err(INCORRECT_NUMBER_OF_INTEGRATION_LIMITS);
        }

        Ok(())
    }

    /// @brief Computes the integral using **Gauss–Legendre quadrature**.
    ///
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations Number of integrations to perform.
    /// @param func Function to integrate.
    /// @param integration_limit Integration limits for each round of integration.
    /// @return The computed integral value.
    fn get_gauss_legendre<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> f64 {
        if number_of_integrations == 1 {
            let mut ans = 0.0;
            let abscissa_coeff = (integration_limit[0][1] - integration_limit[0][0]) / 2.0;
            let intercept = (integration_limit[0][1] + integration_limit[0][0]) / 2.0;

            for iter in 0..self.order {
                let (weight, abscissa) = gauss_tables::get_weight_and_abscissa(
                    GaussianQuadratureMethod::GaussLegendre,
                    self.order,
                    iter,
                )
                .unwrap();

                let args = abscissa_coeff * abscissa + intercept;
                ans += weight * func(args);
            }

            return abscissa_coeff * ans;
        }

        let mut ans = 0.0;
        let abscissa_coeff = (integration_limit[number_of_integrations - 1][1]
            - integration_limit[number_of_integrations - 1][0])
            / 2.0;

        for iter in 0..self.order {
            let (weight, _) = gauss_tables::get_weight_and_abscissa(
                GaussianQuadratureMethod::GaussLegendre,
                self.order,
                iter,
            )
            .unwrap();

            ans += weight
                * self.get_gauss_legendre(number_of_integrations - 1, func, integration_limit);
        }

        abscissa_coeff * ans
    }

    /// @brief Computes the integral using **Gauss–Hermite quadrature**.
    ///
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations Number of integrations to perform.
    /// @param func Function to integrate.
    /// @param _integration_limit Ignored integration limits, currently only integrates from -infinity to +infinity.
    /// @return The computed integral value.
    fn get_gauss_hermite<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        _integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> f64 {
        if number_of_integrations == 1 {
            let mut ans = 0.0;

            for iter in 0..self.order {
                let (weight, abscissa) = gauss_tables::get_weight_and_abscissa(
                    GaussianQuadratureMethod::GaussHermite,
                    self.order,
                    iter,
                )
                .unwrap();

                ans += weight * func(abscissa);
            }

            return ans;
        }

        let mut ans = 0.0;

        for iter in 0..self.order {
            let (weight, _) = gauss_tables::get_weight_and_abscissa(
                GaussianQuadratureMethod::GaussHermite,
                self.order,
                iter,
            )
            .unwrap();

            ans += weight
                * self.get_gauss_hermite(number_of_integrations - 1, func, _integration_limit);
        }

        ans
    }

    /// @brief Computes the integral using **Gauss–Laguerre quadrature**.
    ///
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations Number of integrations to perform.
    /// @param func Function to integrate.
    /// @param _integration_limit Ignored integration limits, currently only integrates from -infinity to +infinity.
    /// @return The computed integral value.
    fn get_gauss_laguerre<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        _integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> f64 {
        if number_of_integrations == 1 {
            let mut ans = 0.0;

            for iter in 0..self.order {
                let (weight, abscissa) = gauss_tables::get_weight_and_abscissa(
                    GaussianQuadratureMethod::GaussLaguerre,
                    self.order,
                    iter,
                )
                .unwrap();

                ans += weight * func(abscissa);
            }

            return ans;
        }

        let mut ans = 0.0;

        for iter in 0..self.order {
            let (weight, _) = gauss_tables::get_weight_and_abscissa(
                GaussianQuadratureMethod::GaussLaguerre,
                self.order,
                iter,
            )
            .unwrap();

            ans +=
                weight * self.get_gauss_laguerre(number_of_integrations, func, _integration_limit);
        }

        ans
    }
}


impl IntegratorSingleVariable for SingleVariableSolver {
    /// @brief Computes the Gaussian quadrature numerical integration for a single-variable function.
    ///
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations Number of integrations to perform.
    /// @param func Function to integrate.
    /// @param integration_limit Integration bounds for each round of integration.
    /// @return Result containing the computed integral value, or an error message.
    ///
    /// @note Possible errors:
    /// - `GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE`: if the chosen order is unsupported.
    /// - `INTEGRATION_LIMITS_ILL_DEFINED`: if any limit has `a >= b`.
    /// - `INCORRECT_NUMBER_OF_INTEGRATION_LIMITS`: if bounds count ≠ integration order.
    ///
    /// @example Assume we want to differentiate f(x) = 4.0*x*x*x - 3.0*x*x. the function would be:
    /// ```
    ///    let my_func = | arg: f64 | -> f64
    ///    {
    ///        return 4.0*arg*arg*arg - 3.0*arg*arg;
    ///    };
    ///
    /// use multicalc::numerical_integration::integrator::*;
    /// use multicalc::numerical_integration::gaussian_integration;
    ///
    /// let integrator = gaussian_integration::SingleVariableSolver::default();
    /// let integration_limit = [[0.0, 2.0]; 1];
    /// let val = integrator.get(1, &my_func, &integration_limit).unwrap(); //single integration
    /// assert!(f64::abs(val - 8.0) < 1e-7);
    ///
    /// let integration_limit = [[0.0, 2.0], [-1.0, 1.0]];
    /// let val = integrator.get(2, &my_func, &integration_limit).unwrap(); //double integration
    /// assert!(f64::abs(val - 16.0) < 1e-7);
    ///```
    fn get<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        func: &dyn Fn(f64) -> f64,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<f64, &'static str> {
        self.check_for_errors(number_of_integrations, integration_limit)?;

        match self.integration_method {
            GaussianQuadratureMethod::GaussLegendre => {
                Ok(self.get_gauss_legendre(number_of_integrations, func, integration_limit))
            }
            GaussianQuadratureMethod::GaussHermite => {
                Ok(self.get_gauss_hermite(number_of_integrations, func, integration_limit))
            }
            GaussianQuadratureMethod::GaussLaguerre => {
                Ok(self.get_gauss_laguerre(number_of_integrations, func, integration_limit))
            }
        }
    }
}

/// @brief Implements the gaussian quadrature methods for numerical integration for multi variable functions.
#[derive(Clone, Copy)]
pub struct MultiVariableSolver {
    order: usize,
    integration_method: GaussianQuadratureMethod,
}

impl Default for MultiVariableSolver {
    /// @brief Default constructor, optimal for most generic polynomial equations.
    fn default() -> Self {
        MultiVariableSolver {
            order: DEFAULT_QUADRATURE_ORDERS,
            integration_method: GaussianQuadratureMethod::GaussLegendre,
        }
    }
}

impl MultiVariableSolver {
    /// @brief Returns the chosen number of nodes/order for quadrature.
    pub fn get_order(&self) -> usize {
        self.order
    }

    ///@ brief Sets the number of nodes/order for quadrature.
    pub fn set_order(&mut self, order: usize) {
        self.order = order;
    }

    /// @brief Returns the chosen integration method.
    /// @note Possible choices are GaussLegendre, GaussHermite and GaussLaguerre.
    pub fn get_integration_method(&self) -> GaussianQuadratureMethod {
        self.integration_method
    }

    /// @brief Sets the integration method.
    /// @note Possible choices are GaussLegendre, GaussHermite and GaussLaguerre.
    pub fn set_integration_method(&mut self, integration_method: GaussianQuadratureMethod) {
        self.integration_method = integration_method;
    }

    /// @brief Creates a solver with custom parameters.
    /// @param order The quadrature order (number of nodes).
    /// @param integration_method The numerical method (`GaussLegendre`, `GaussHermite`, or `GaussLaguerre`).
    /// @return A configured [`SingleVariableSolver`] instance.
    pub fn from_parameters(order: usize, integration_method: GaussianQuadratureMethod) -> Self {
        MultiVariableSolver {
            order,
            integration_method,
        }
    }

    /// @brief Validates the input parameters and integration limits.
    ///
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations The number of integrations to perform.
    /// @param integration_limit Integration limits for each integration stage.
    /// @return `Ok(())` if valid; otherwise an error string.
    fn check_for_errors<const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        integration_limit: &[[f64; 2]; NUM_INTEGRATIONS],
    ) -> Result<(), &'static str> {
        if !(1..=gauss_tables::MAX_GAUSS_TABLE_ORDER).contains(&self.order) {
            return Err(GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE);
        }

        for &limit in integration_limit {
            if limit[0] >= limit[1] {
                return Err(INTEGRATION_LIMITS_ILL_DEFINED);
            }
        }

        if NUM_INTEGRATIONS != number_of_integrations {
            return Err(INCORRECT_NUMBER_OF_INTEGRATION_LIMITS);
        }

        Ok(())
    }

    /// @brief Computes the integral using **Gauss–Legendre quadrature**.
    ///
    /// @tparam NUM_VARS Number of variables in the multivariable equation.
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations Number of integrations to perform.
    /// @param idx_to_integrate The index/indices of variable to integrate.
    /// @param func Function to integrate.
    /// @param integration_limit Integration limits for each round of integration.
    /// @param point For variables not being integrated, it is their constant value, otherwise it is 
    /// their final upper limit of integration.
    /// @return The computed integral value.
    fn get_gauss_legendre<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> f64 {
        if number_of_integrations == 1 {
            let mut ans = 0.0;
            let abcsissa_coeff = (integration_limits[0][1] - integration_limits[0][0]) / 2.0;
            let intercept = (integration_limits[0][1] + integration_limits[0][0]) / 2.0;

            let mut args = *point;

            for iter in 0..self.order {
                let (weight, abcsissa) = gauss_tables::get_weight_and_abscissa(
                    GaussianQuadratureMethod::GaussLegendre,
                    self.order,
                    iter,
                )
                .unwrap();

                args[idx_to_integrate[0]] = abcsissa_coeff * abcsissa + intercept;

                ans += weight * func(&args);
            }

            return abcsissa_coeff * ans;
        }

        let mut ans = 0.0;
        let abcsissa_coeff = (integration_limits[number_of_integrations - 1][1]
            - integration_limits[number_of_integrations - 1][0])
            / 2.0;

        let intercept = (integration_limits[number_of_integrations - 1][1]
            + integration_limits[number_of_integrations - 1][0])
            / 2.0;

        let mut args = *point;

        for iter in 0..self.order {
            let (weight, abcsissa) = gauss_tables::get_weight_and_abscissa(
                GaussianQuadratureMethod::GaussLegendre,
                self.order,
                iter,
            )
            .unwrap();

            args[idx_to_integrate[number_of_integrations - 1]] =
                abcsissa_coeff * abcsissa + intercept;

            ans += weight
                * self.get_gauss_legendre(
                    number_of_integrations - 1,
                    idx_to_integrate,
                    func,
                    integration_limits,
                    &args,
                );
        }

        abcsissa_coeff * ans
    }

    /// @brief Computes the integral using **Gauss-Hermite quadrature**.
    ///
    /// @tparam NUM_VARS Number of variables in the multivariable equation.
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations Number of integrations to perform.
    /// @param idx_to_integrate The index/indices of variable to integrate.
    /// @param func Function to integrate.
    /// @param integration_limit Integration limits for each round of integration.
    /// @param point For variables not being integrated, it is their constant value, otherwise it is 
    /// their final upper limit of integration.
    /// @return The computed integral value.
    fn get_gauss_hermite<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        _integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> f64 {
        if number_of_integrations == 1 {
            let mut ans = 0.0;

            let mut args = *point;

            for iter in 0..self.order {
                let (weight, abcsissa) = gauss_tables::get_weight_and_abscissa(
                    GaussianQuadratureMethod::GaussHermite,
                    self.order,
                    iter,
                )
                .unwrap();

                args[idx_to_integrate[0]] = abcsissa;

                ans += weight * func(&args);
            }

            return ans;
        }

        let mut ans = 0.0;

        let mut args = *point;

        for iter in 0..self.order {
            let (weight, abcsissa) = gauss_tables::get_weight_and_abscissa(
                GaussianQuadratureMethod::GaussHermite,
                self.order,
                iter,
            )
            .unwrap();

            args[idx_to_integrate[number_of_integrations - 1]] = abcsissa;

            ans += weight
                * self.get_gauss_hermite(
                    number_of_integrations - 1,
                    idx_to_integrate,
                    func,
                    _integration_limits,
                    &args,
                );
        }

        ans
    }

    /// @brief Computes the integral using **Gauss-Laguerre quadrature**.
    ///
    /// @tparam NUM_VARS Number of variables in the multivariable equation.
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations Number of integrations to perform.
    /// @param idx_to_integrate The index/indices of variable to integrate.
    /// @param func Function to integrate.
    /// @param integration_limit Integration limits for each round of integration.
    /// @param point For variables not being integrated, it is their constant value, otherwise it is 
    /// their final upper limit of integration.
    /// @return The computed integral value.
    fn get_gauss_laguerre<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        _integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> f64 {
        if number_of_integrations == 1 {
            let mut ans = 0.0;

            let mut args = *point;

            for iter in 0..self.order {
                let (weight, abcsissa) = gauss_tables::get_weight_and_abscissa(
                    GaussianQuadratureMethod::GaussLaguerre,
                    self.order,
                    iter,
                )
                .unwrap();

                args[idx_to_integrate[0]] = abcsissa;

                ans += weight * func(&args);
            }

            return ans;
        }

        let mut ans = 0.0;

        let mut args = *point;

        for iter in 0..self.order {
            let (weight, abcsissa) = gauss_tables::get_weight_and_abscissa(
                GaussianQuadratureMethod::GaussLaguerre,
                self.order,
                iter,
            )
            .unwrap();

            args[idx_to_integrate[number_of_integrations - 1]] = abcsissa;

            ans += weight
                * self.get_gauss_laguerre(
                    number_of_integrations - 1,
                    idx_to_integrate,
                    func,
                    _integration_limits,
                    &args,
                );
        }

        ans
    }
}

impl IntegratorMultiVariable for MultiVariableSolver {
    /// @brief Computes the Gaussian quadrature numerical integration for a multivariable function.
    /// 
    /// @tparam NUM_VARS Number of variables in the multivariable equation.
    /// @tparam NUM_INTEGRATIONS Number of nested integrations.
    /// @param number_of_integrations Number of integrations to perform.
    /// @param idx_to_integrate The index/indices of variable to integrate.
    /// @param func Function to integrate.
    /// @param integration_limit Integration limits for each round of integration.
    /// @param point For variables not being integrated, it is their constant value, otherwise it is 
    /// their final upper limit of integration.
    /// 
    /// @return Result containing the computed integral value, or an error message.
    ///
    /// @note Possible errors:
    /// - `GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE`: if the chosen order is unsupported.
    /// - `INTEGRATION_LIMITS_ILL_DEFINED`: if any limit has `a >= b`.
    ///
    /// @example Assume we want to differentiate f(x,y,z) = 2.0*x + y*z. the function would be:
    /// ```
    ///    let my_func = | args: &[f64; 3] | -> f64
    ///    {
    ///        return 2.0*args[0] + args[1]*args[2];
    ///    };
    ///
    /// use multicalc::numerical_integration::integrator::*;
    /// use multicalc::numerical_integration::gaussian_integration;
    ///
    /// let integrator = gaussian_integration::MultiVariableSolver::default();
    /// let point = [1.0, 2.0, 3.0];
    ///
    /// let integration_limit = [[0.0, 1.0]; 1];
    /// let val = integrator.get(1, [0; 1], &my_func, &integration_limit, &point).unwrap(); //single integration for x
    /// assert!(f64::abs(val - 7.0) < 1e-7);
    ///
    ///```
    fn get<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(
        &self,
        number_of_integrations: usize,
        idx_to_integrate: [usize; NUM_INTEGRATIONS],
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        integration_limits: &[[f64; 2]; NUM_INTEGRATIONS],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str> {
        self.check_for_errors(number_of_integrations, integration_limits)?;

        match self.integration_method {
            GaussianQuadratureMethod::GaussLegendre => Ok(self.get_gauss_legendre(
                number_of_integrations,
                idx_to_integrate,
                func,
                integration_limits,
                point,
            )),
            GaussianQuadratureMethod::GaussHermite => Ok(self.get_gauss_hermite(
                number_of_integrations,
                idx_to_integrate,
                func,
                integration_limits,
                point,
            )),
            GaussianQuadratureMethod::GaussLaguerre => Ok(self.get_gauss_laguerre(
                number_of_integrations,
                idx_to_integrate,
                func,
                integration_limits,
                point,
            )),
        }
    }
}
