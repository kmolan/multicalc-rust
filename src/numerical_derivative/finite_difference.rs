use crate::numerical_derivative::mode as mode;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;
use crate::numerical_derivative::derivator::*;

#[derive(Clone, Copy)]
pub struct FiniteDifference
{
    step_size: f64,
    method: mode::FiniteDifferenceMode
}

impl Default for FiniteDifference
{
    fn default() -> Self 
    {
        return FiniteDifference 
        { 
            step_size: mode::DEFAULT_STEP_SIZE,
            method: mode::FiniteDifferenceMode::Central
        };
    }
}

impl FiniteDifference
{
    pub fn get_step_size(&self) -> f64
    {
        return self.step_size;
    }

    pub fn set_step_size(&mut self, step_size: f64) 
    {
        self.step_size = step_size;
    }

    pub fn get_method(&self) -> mode::FiniteDifferenceMode
    {
        return self.method;
    }

    pub fn set_method(&mut self, method: mode::FiniteDifferenceMode)
    {
        self.method = method;
    }

    pub fn from_parameters(step_size: f64, method: mode::FiniteDifferenceMode) -> Self 
    {
        FiniteDifference
        {
            step_size: step_size,
            method: method
        }    
    }

    fn get_forward_difference_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: usize, point: &[T; NUM_VARS]) -> T
    {
        let f0_args = point;

        let mut f1_args = *point;
        f1_args[idx_to_derivate] = f1_args[idx_to_derivate] + T::from(self.step_size).unwrap(); 

        let f0 = func(f0_args);
        let f1 = func(&f1_args);

        return (f1 - f0)/T::from(self.step_size).unwrap();
    }

    fn get_backward_difference_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: usize, point: &[T; NUM_VARS]) -> T
    {
        let mut f0_args = *point;
        f0_args[idx_to_derivate] = f0_args[idx_to_derivate] - T::from(self.step_size).unwrap(); 

        let f1_args = point;

        let f0 = func(&f0_args);
        let f1 = func(f1_args);

        return (f1 - f0)/T::from(self.step_size).unwrap();
    }

    fn get_central_difference_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: usize, point: &[T; NUM_VARS]) -> T
    {
        let mut f0_args = *point;
        f0_args[idx_to_derivate] = f0_args[idx_to_derivate] - T::from(self.step_size).unwrap();

        let mut f1_args = *point;
        f1_args[idx_to_derivate] = f1_args[idx_to_derivate] + T::from(self.step_size).unwrap(); 

        let f0 = func(&f0_args);
        let f1 = func(&f1_args);

        return (f1 - f0)/(T::from(2.0*self.step_size).unwrap());
    }

    fn get_forward_difference_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let f0 = self.get_single_partial(func, idx_to_derivate[1], point)?;

        let mut f1_point = *point;
        f1_point[idx_to_derivate[0]] = f1_point[idx_to_derivate[0]] + T::from(self.step_size).unwrap();
        let f1 = self.get_single_partial(func, idx_to_derivate[1], &f1_point)?;

        return Ok((f1 - f0)/T::from(self.step_size).unwrap());
    }

    fn get_backward_difference_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let mut f0_point = *point;
        f0_point[idx_to_derivate[0]] = f0_point[idx_to_derivate[0]] - T::from(self.step_size).unwrap();
        let f0 = self.get_single_partial(func, idx_to_derivate[1], &f0_point)?;

        let f1 = self.get_single_partial(func, idx_to_derivate[1], point)?;

        return Ok((f1 - f0)/T::from(self.step_size).unwrap());
    }

    fn get_central_difference_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let mut f0_point = *point;
        f0_point[idx_to_derivate[0]] = f0_point[idx_to_derivate[0]] - T::from(self.step_size).unwrap();
        let f0 = self.get_single_partial(func, idx_to_derivate[1], &f0_point)?;

        let mut f1_point = *point;
        f1_point[idx_to_derivate[0]] = f1_point[idx_to_derivate[0]] + T::from(self.step_size).unwrap();
        let f1 = self.get_single_partial(func, idx_to_derivate[1], &f1_point)?;

        return Ok((f1 - f0)/(T::from(2.0*self.step_size).unwrap()));
    }
}

impl SingleTotalDerivator for FiniteDifference
{
    fn get_single_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, point: T) -> Result<T, ErrorCode> 
    {
        if self.step_size == 0.0
        {
            return Err(ErrorCode::NumberOfStepsCannotBeZero);
        }

        let vec_point = [point; NUM_VARS];

        match self.method
        {
            mode::FiniteDifferenceMode::Forward => return Ok(self.get_forward_difference_1(func, 0, &vec_point)),
            mode::FiniteDifferenceMode::Backward => return Ok(self.get_backward_difference_1(func, 0, &vec_point)),
            mode::FiniteDifferenceMode::Central => return Ok(self.get_central_difference_1(func, 0, &vec_point)),
        }
    }
}

impl SinglePartialDerivator for FiniteDifference
{

    fn get_single_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: usize, point: &[T; NUM_VARS]) -> Result<T, ErrorCode> 
    {
        if self.step_size == 0.0
        {
            return Err(ErrorCode::NumberOfStepsCannotBeZero);
        }
        if idx_to_derivate >= NUM_VARS
        {
            return Err(ErrorCode::IndexToDerivativeOutOfRange);
        }

        match self.method
        {
            mode::FiniteDifferenceMode::Forward => return Ok(self.get_forward_difference_1(func, idx_to_derivate, &point)),
            mode::FiniteDifferenceMode::Backward => return Ok(self.get_backward_difference_1(func, idx_to_derivate, &point)),
            mode::FiniteDifferenceMode::Central => return Ok(self.get_central_difference_1(func, idx_to_derivate, &point)),
        }
    }
}

impl DoubleTotalDerivator for FiniteDifference
{
    fn get_double_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, point: T) -> Result<T, ErrorCode> 
    {
        if self.step_size == 0.0
        {
            return Err(ErrorCode::NumberOfStepsCannotBeZero);
        }

        let vec_point = [point; NUM_VARS];

        match self.method
        {
            mode::FiniteDifferenceMode::Forward => return self.get_forward_difference_2(func, &[0, 0], &vec_point),
            mode::FiniteDifferenceMode::Backward => return self.get_backward_difference_2(func, &[0, 0], &vec_point),
            mode::FiniteDifferenceMode::Central => return self.get_central_difference_2(func, &[0, 0], &vec_point) 
        }
    }
}

impl DoublePartialDerivator for FiniteDifference
{
    fn get_double_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode> 
    {
        if self.step_size == 0.0
        {
            return Err(ErrorCode::NumberOfStepsCannotBeZero);
        }
        
        match self.method
        {
            mode::FiniteDifferenceMode::Forward => return self.get_forward_difference_2(func, idx_to_derivate, &point),
            mode::FiniteDifferenceMode::Backward => return self.get_backward_difference_2(func, idx_to_derivate, &point),
            mode::FiniteDifferenceMode::Central => return self.get_central_difference_2(func, idx_to_derivate, &point),
        }
    }
}