/// @brief Base trait for single-variable numerical differentiation.
pub trait DerivatorSingleVariable: Default + Clone + Copy {
    /// @brief Compute the n-th derivative of a single-variable function.
    ///
    /// @param order The order of the derivative to compute (e.g., 1 for first derivative).
    /// @param func  A reference to the function to differentiate. Must take and return `f64`.
    /// @param point The point at which to evaluate the derivative.
    ///
    /// @return The computed derivative value as `Ok(f64)` or an error message as `Err(&'static str)`
    ///         if the computation fails.
    fn get(&self, order: usize, func: &dyn Fn(f64) -> f64, point: f64)
        -> Result<f64, &'static str>;

    /// @brief Convenience wrapper for computing the first derivative.
    ///
    /// @param func  A reference to the function to differentiate.
    /// @param point The point at which to evaluate the first derivative.
    ///
    /// @return The computed first derivative value or an error message.
    fn get_single(&self, func: &dyn Fn(f64) -> f64, point: f64) -> Result<f64, &'static str> {
        self.get(1, func, point)
    }

    /// @brief Convenience wrapper for computing the second derivative.
    ///
    /// @param func  A reference to the function to differentiate.
    /// @param point The point at which to evaluate the second derivative.
    ///
    /// @return The computed second derivative value or an error message.
    fn get_double(&self, func: &dyn Fn(f64) -> f64, point: f64) -> Result<f64, &'static str> {
        self.get(2, func, point)
    }
}

/// @brief Base trait for multi-variable numerical differentiation.
pub trait DerivatorMultiVariable: Default + Clone + Copy {
    /// @brief Compute the n-th order derivative of a multi-variable function.
    ///
    /// @tparam NUM_VARS  The number of variables in the function.
    /// @tparam NUM_ORDER The number of derivatives to take (order of differentiation).
    ///
    /// @param order            The order of the derivative to compute.
    /// @param func             A reference to the multi-variable function to differentiate.
    /// @param idx_to_derivate  Array of variable indices specifying differentiation order.
    ///                         For example, `[0, 1]` means ∂²f / ∂x₀∂x₁.
    /// @param point            The point at which to evaluate the derivative.
    ///
    /// @return The computed derivative value as `Ok(f64)` or an error message as `Err(&'static str)`.
    fn get<const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str>;

    /// @brief Convenience wrapper for computing a single partial derivative.
    ///
    /// @tparam NUM_VARS The number of variables in the function.
    ///
    /// @param func             A reference to the multi-variable function to differentiate.
    /// @param idx_to_derivate  The index of the variable with respect to which the derivative is taken.
    /// @param point            The point at which to evaluate the derivative.
    ///
    /// @return The computed partial derivative value or an error message.
    fn get_single_partial<const NUM_VARS: usize>(
        &self,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: usize,
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str> {
        self.get(1, func, &[idx_to_derivate], point)
    }

    /// @brief Convenience wrapper for computing a second-order partial derivative.
    ///
    /// @tparam NUM_VARS The number of variables in the function.
    ///
    /// @param func             A reference to the multi-variable function to differentiate.
    /// @param idx_to_derivate  Array specifying which variables to differentiate with respect to,
    ///                         for example `[0, 1]` for ∂²f / ∂x₀∂x₁.
    /// @param point            The point at which to evaluate the derivative.
    ///
    /// @return The computed second-order partial derivative value or an error message.
    fn get_double_partial<const NUM_VARS: usize>(
        &self,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; 2],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str> {
        self.get(2, func, idx_to_derivate, point)
    }
}
