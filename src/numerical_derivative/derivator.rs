use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;


pub trait SingleTotalDerivator: Default + Clone + Copy
{
    fn get_single_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, point: T) -> Result<T, ErrorCode>;
}

pub trait SinglePartialDerivator: Default + Clone + Copy
{
    fn get_single_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: usize, point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;
}

pub trait DoubleTotalDerivator: Default + Clone + Copy
{
    fn get_double_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, point: T) -> Result<T, ErrorCode>;
}

pub trait DoublePartialDerivator: Default + Clone + Copy
{
    fn get_double_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>;
}