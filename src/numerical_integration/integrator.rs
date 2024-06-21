use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;

pub trait IntegratorSingleVariable: Default + Clone + Copy
{
    ///generic n-th integration of a single variable function
    fn get<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> Result<T, ErrorCode>;

    ///convenience wrapper for a single integral of a single variable function
    fn get_single<T:ComplexFloat>(&self, func: &dyn Fn(T) -> T, integration_limit: &[T; 2]) -> Result<T, ErrorCode>
    {
        let new_limits: [[T; 2]; 1] = [*integration_limit];

        return self.get(1, func, &new_limits);
    }

    ///convenience wrapper for a double integral of a single variable function
    fn get_double<T:ComplexFloat>(&self, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; 2]) -> Result<T, ErrorCode>
    {
        return self.get(2, func, integration_limit);
    }
}

pub trait IntegratorMultiVariable : Default + Clone + Copy
{
    ///generic n-th partial integration of a multi variable function
    fn get<T: ComplexFloat, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; NUM_INTEGRATIONS], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;

    ///convenience wrapper for a single partial integral of a multi variable function
    fn get_single_partial<T:ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limits: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let new_limits: [[T; 2]; 1] = [*integration_limits];
        let new_idx: [usize; 1] = [idx_to_integrate];

        return self.get(1, new_idx, func, &new_limits, point);
    }

    ///convenience wrapper for a double partial integral of a multi variable function
    fn get_double_partial<T:ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        return self.get(2, idx_to_integrate, func, integration_limits, point);
    }
}