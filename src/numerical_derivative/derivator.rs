use num_complex::ComplexFloat;
use crate::utils::error_codes::ErrorCode;

pub trait DerivatorSingleVariable: Default + Clone + Copy
{
    ///generic n-th derivative of a single variable function
    fn get<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T) -> Result<T, ErrorCode>;

    ///convenience wrapper for a single derivative of a single variable function
    fn get_single<T: ComplexFloat>(&self, func: &dyn Fn(T) -> T, point: T) -> Result<T, ErrorCode>
    {
        return self.get(1, func, point);
    }

    ///convenience wrapper for a double derivative of a single variable function
    fn get_double<T: ComplexFloat>(&self, func: &dyn Fn(T) -> T, point: T) -> Result<T, ErrorCode>
    {
        return self.get(2, func, point);
    }
}

pub trait DerivatorMultiVariable: Default + Clone + Copy
{
    ///generic n-th derivative for a multivariable function of a multivariable function
    fn get<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;

    ///convenience wrapper for a single partial derivative of a multivariable function
    fn get_single_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: usize, point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        return self.get(1, func, &[idx_to_derivate], point);
    }
    
    ///convenience wrapper for a double partial derivative of a multivariable function
    fn get_double_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        return self.get(2, func, idx_to_derivate, point);
    }
}