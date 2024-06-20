use num_complex::ComplexFloat;
use crate::utils::error_codes::ErrorCode;

pub trait DerivatorSingleVariable: Default + Clone + Copy
{
    fn get<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T) -> Result<T, ErrorCode>;
}

pub trait DerivatorMultiVariable: Default + Clone + Copy
{
    fn get<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;
}