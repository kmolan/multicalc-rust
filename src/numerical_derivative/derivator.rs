///Base trait for single variable numerical differentiation
pub trait DerivatorSingleVariable: Default + Clone + Copy {
    ///generic n-th derivative of a single variable function
    fn get(
        &self,
        order: usize,
        func: &dyn Fn(f64) -> f64,
        point: f64,
    ) -> Result<f64, &'static str>;

    ///convenience wrapper for a single derivative of a single variable function
    fn get_single(
        &self,
        func: &dyn Fn(f64) -> f64,
        point: f64,
    ) -> Result<f64, &'static str> {
        return self.get(1, func, point);
    }

    ///convenience wrapper for a double derivative of a single variable function
    fn get_double(
        &self,
        func: &dyn Fn(f64) -> f64,
        point: f64,
    ) -> Result<f64, &'static str> {
        return self.get(2, func, point);
    }
}

///Base trait for multi-variable numerical differentiation
pub trait DerivatorMultiVariable: Default + Clone + Copy {
    ///generic n-th derivative for a multivariable function of a multivariable function
    fn get<const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        order: usize,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; NUM_ORDER],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str>;

    ///convenience wrapper for a single partial derivative of a multivariable function
    fn get_single_partial<const NUM_VARS: usize>(
        &self,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: usize,
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str> {
        return self.get(1, func, &[idx_to_derivate], point);
    }

    ///convenience wrapper for a double partial derivative of a multivariable function
    fn get_double_partial<const NUM_VARS: usize>(
        &self,
        func: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        idx_to_derivate: &[usize; 2],
        point: &[f64; NUM_VARS],
    ) -> Result<f64, &'static str> {
        return self.get(2, func, idx_to_derivate, point);
    }
}
