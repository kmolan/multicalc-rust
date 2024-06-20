use num_complex::ComplexFloat;
use crate::utils::error_codes::ErrorCode;

pub trait DerivatorSingleVariable: Default + Clone + Copy
{
    ///generic n-th derivative for a single variable function
    fn get<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T) -> Result<T, ErrorCode>;
}

pub trait DerivatorMultiVariable: Default + Clone + Copy
{
    ///specialized version for a single partial derivative
    /// Specialized versions for most-used cases make for a smoother experience with API
    fn get_single_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: usize, point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;
    
    ///specialized version for a double partial derivative
    ///Specialized versions for most-used cases make for a smoother experience with API
    fn get_double_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;

    ///generic n-th derivative for a multivariable function
    fn get<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;
}