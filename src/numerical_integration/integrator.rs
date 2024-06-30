
///Base trait for single variable numerical integration
pub trait IntegratorSingleVariable: Default + Clone + Copy
{
    ///generic n-th integration of a single variable function
    fn get<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> Result<f64, &'static str>;

    ///convenience wrapper for a single integral of a single variable function
    fn get_single(&self, func: &dyn Fn(f64) -> f64, integration_limit: &[f64; 2]) -> Result<f64, &'static str>
    {
        let new_limits: [[f64; 2]; 1] = [*integration_limit];

        return self.get(1, func, &new_limits);
    }

    ///convenience wrapper for a double integral of a single variable function
    fn get_double(&self, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; 2]) -> Result<f64, &'static str>
    {
        return self.get(2, func, integration_limit);
    }
}

///Base trait for multi-variable numerical integration
pub trait IntegratorMultiVariable : Default + Clone + Copy
{
    ///generic n-th partial integration of a multi variable function
    fn get<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[f64; NUM_VARS]) -> f64, integration_limits: &[[f64; 2]; NUM_INTEGRATIONS], point: &[f64; NUM_VARS]) -> Result<f64, &'static str>;

    ///convenience wrapper for a single partial integral of a multi variable function
    fn get_single_partial<const NUM_VARS: usize>(&self, func: &dyn Fn(&[f64; NUM_VARS]) -> f64, idx_to_integrate: usize, integration_limits: &[f64; 2], point: &[f64; NUM_VARS]) -> Result<f64, &'static str>
    {
        let new_limits: [[f64; 2]; 1] = [*integration_limits];
        let new_idx: [usize; 1] = [idx_to_integrate];

        return self.get(1, new_idx, func, &new_limits, point);
    }

    ///convenience wrapper for a double partial integral of a multi variable function
    fn get_double_partial<const NUM_VARS: usize>(&self, func: &dyn Fn(&[f64; NUM_VARS]) -> f64, idx_to_integrate: [usize; 2], integration_limits: &[[f64; 2]; 2], point: &[f64; NUM_VARS]) -> Result<f64, &'static str>
    {
        return self.get(2, idx_to_integrate, func, integration_limits, point);
    }
}