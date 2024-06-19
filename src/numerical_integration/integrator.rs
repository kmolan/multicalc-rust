use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;

pub trait Integrator
{
    fn get_single_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limit: &[T; 2]) -> Result<T, ErrorCode>;
    fn get_single_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;
    
    fn get_double_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; 2]) -> Result<T, ErrorCode>;
    fn get_double_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;
}