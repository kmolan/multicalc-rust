use crate::numerical_derivative::mode; 
use crate::utils::error_codes::*;

use num_complex::ComplexFloat;
use crate::numerical_derivative::derivator::*;

#[derive(Clone, Copy)]
pub struct SingleVariableSolver
{
    step_size: f64,
    method: mode::FiniteDifferenceMode,

    //the step size will be multipled by this factor after every iteration
    step_size_multiplier: f64
}

impl Default for SingleVariableSolver
{
    fn default() -> Self 
    {
        return SingleVariableSolver 
        { 
            step_size: mode::DEFAULT_STEP_SIZE,
            method: mode::FiniteDifferenceMode::Central,
            step_size_multiplier: mode::DEFAULT_STEP_SIZE_MULTIPLIER
        };
    }
}

impl SingleVariableSolver
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

    pub fn get_step_size_multiplier(&self) -> f64
    {
        return self.step_size_multiplier;
    }

    pub fn set_step_size_multiplier(&mut self, multiplier: f64)
    {
        self.step_size_multiplier = multiplier;
    }

    pub fn from_parameters(step: f64, method: mode::FiniteDifferenceMode, multiplier: f64) -> Self 
    {
        SingleVariableSolver
        {
            step_size: step,
            method: method,
            step_size_multiplier: multiplier
        }    
    }

    fn get_forward_difference_single_variable<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T, step_size: f64) -> T
    {
        if order == 1
        {
            let f0 = func(point);
            let f1 = func(point + T::from(step_size).unwrap());
            return (f1 - f0)/(T::from(step_size).unwrap());
        }

        let f0_point = point;
        let f0 = self.get_forward_difference_single_variable(order - 1, func, f0_point, self.step_size_multiplier*step_size);

        let f1_point = point + T::from(step_size).unwrap();
        let f1 = self.get_forward_difference_single_variable(order - 1, func, f1_point, self.step_size_multiplier*step_size);

        return (f1 - f0)/(T::from(step_size).unwrap());
    }

    fn get_backward_difference_single_variable<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T, step_size: f64) -> T
    {
        if order == 1
        {
            let f0 = func(point - T::from(step_size).unwrap());
            let f1 = func(point);
            return (f1 - f0)/(T::from(step_size).unwrap());
        }

        let f0_point = point - T::from(step_size).unwrap();
        let f0 = self.get_backward_difference_single_variable(order - 1, func, f0_point, self.step_size_multiplier*step_size);

        let f1_point = point;
        let f1 = self.get_backward_difference_single_variable(order - 1, func, f1_point, self.step_size_multiplier*step_size);

        return (f1 - f0)/(T::from(step_size).unwrap());
    }


    fn get_central_difference_single_variable<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T, step_size: f64) -> T
    {
        if order == 1
        {
            let f0 = func(point - T::from(step_size).unwrap());
            let f1 = func(point + T::from(step_size).unwrap());
            return (f1 - f0)/(T::from(2.0*step_size).unwrap());
        }

        let f0_point = point - T::from(step_size).unwrap();
        let f0 = self.get_central_difference_single_variable(order - 1, func, f0_point, self.step_size_multiplier*step_size);

        let f1_point = point + T::from(step_size).unwrap();
        let f1 = self.get_central_difference_single_variable(order - 1, func, f1_point, self.step_size_multiplier*step_size);

        return (f1 - f0)/(T::from(2.0*step_size).unwrap());
    }
}

impl DerivatorSingleVariable for SingleVariableSolver
{
    fn get<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T) -> Result<T, &'static str>
    {
        if self.step_size == 0.0
        {
            return Err(NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO);
        }

        match self.method
        {
            mode::FiniteDifferenceMode::Forward => return Ok(self.get_forward_difference_single_variable(order, func, point, self.step_size)),
            mode::FiniteDifferenceMode::Backward => return Ok(self.get_backward_difference_single_variable(order, func, point, self.step_size)),
            mode::FiniteDifferenceMode::Central => return Ok(self.get_central_difference_single_variable(order, func, point, self.step_size)) 
        } 
    }
}

#[derive(Clone, Copy)]
pub struct MultiVariableSolver
{
    step_size: f64,
    method: mode::FiniteDifferenceMode,

    //the step size will be multipled by this factor after every iteration
    step_size_multiplier: f64
}

impl Default for MultiVariableSolver
{
    fn default() -> Self 
    {
        return MultiVariableSolver 
        { 
            step_size: mode::DEFAULT_STEP_SIZE,
            method: mode::FiniteDifferenceMode::Central,
            step_size_multiplier: mode::DEFAULT_STEP_SIZE_MULTIPLIER
        };
    }
}

impl MultiVariableSolver
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

    pub fn get_step_size_multiplier(&self) -> f64
    {
        return self.step_size_multiplier;
    }

    pub fn set_step_size_multiplier(&mut self, multiplier: f64)
    {
        self.step_size_multiplier = multiplier;
    }

    pub fn from_parameters(step: f64, method: mode::FiniteDifferenceMode, multiplier: f64) -> Self 
    {
        MultiVariableSolver
        {
            step_size: step,
            method: method,
            step_size_multiplier: multiplier
        }    
    }

    fn get_forward_difference_multi_variable<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS], step_size: f64) -> T
    {
        if order == 1
        {
            let f0_args = point;

            let mut f1_args = *point;
            f1_args[idx_to_derivate[0]] = f1_args[idx_to_derivate[0]] + T::from(step_size).unwrap(); 

            let f0 = func(f0_args);
            let f1 = func(&f1_args);

            return (f1 - f0)/T::from(step_size).unwrap();
        }

        let f0_args = point;

        let mut f1_args = *point;
        f1_args[idx_to_derivate[order - 1]] = f1_args[idx_to_derivate[order - 1]] + T::from(step_size).unwrap(); 

        let f0 = self.get_forward_difference_multi_variable(order - 1, func, idx_to_derivate, f0_args, self.step_size_multiplier*step_size);
        let f1 = self.get_forward_difference_multi_variable(order - 1, func, idx_to_derivate, &f1_args, self.step_size_multiplier*step_size);

        return (f1 - f0)/T::from(step_size).unwrap();
    }

    fn get_backward_difference_multi_variable<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS], step_size: f64) -> T
    {
        if order == 1
        {
            let mut f0_args = *point;
            f0_args[idx_to_derivate[0]] = f0_args[idx_to_derivate[0]] - T::from(step_size).unwrap(); 

            let f1_args = point;

            let f0 = func(&f0_args);
            let f1 = func(f1_args);

            return (f1 - f0)/T::from(step_size).unwrap();
        }

        let mut f0_args = *point;
        f0_args[idx_to_derivate[order - 1]] = f0_args[idx_to_derivate[order - 1]] - T::from(step_size).unwrap(); 

        let f1_args = point;

        let f0 = self.get_backward_difference_multi_variable(order - 1, func, idx_to_derivate, &f0_args, self.step_size_multiplier*step_size);
        let f1 = self.get_backward_difference_multi_variable(order - 1, func, idx_to_derivate, f1_args, self.step_size_multiplier*step_size);

        return (f1 - f0)/T::from(step_size).unwrap();
    }

    fn get_central_difference_multi_variable<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS], step_size: f64) -> T
    {
        if order == 1
        {
            let mut f0_args = *point;
            f0_args[idx_to_derivate[0]] = f0_args[idx_to_derivate[0]] - T::from(step_size).unwrap();

            let mut f1_args = *point;
            f1_args[idx_to_derivate[0]] = f1_args[idx_to_derivate[0]] + T::from(step_size).unwrap(); 

            let f0 = func(&f0_args);
            let f1 = func(&f1_args);

            return (f1 - f0)/(T::from(2.0*step_size).unwrap());
        }

        let mut f0_point = *point;
        f0_point[idx_to_derivate[order - 1]] = f0_point[idx_to_derivate[order - 1]] - T::from(step_size).unwrap();

        let f0 = self.get_central_difference_multi_variable(order - 1, func, idx_to_derivate, &f0_point, self.step_size_multiplier*step_size);

        let mut f1_point = *point;
        f1_point[idx_to_derivate[order - 1]] = f1_point[idx_to_derivate[order - 1]] + T::from(step_size).unwrap();

        let f1 = self.get_central_difference_multi_variable(order - 1, func, idx_to_derivate, &f1_point, self.step_size_multiplier*step_size);

        return (f1 - f0)/(T::from(2.0*step_size).unwrap());
    }
}

impl DerivatorMultiVariable for MultiVariableSolver
{
    fn get<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS]) -> Result<T, &'static str> 
    {
        if self.step_size == 0.0
        {
            return Err(NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO);
        }
        if order == 0
        {
            return Err(DERIVATE_ORDER_CANNOT_BE_ZERO);
        }
        if order != NUM_ORDER
        {
            return Err(INDEX_TO_DERIVATE_ILL_FORMED);
        }
        
        for iter in 0..idx_to_derivate.len()
        {
            if idx_to_derivate[iter] >= point.len()
            {
                return Err(INDEX_TO_DERIVATIVE_OUT_OF_RANGE);
            }
        }

        match self.method
        {
            mode::FiniteDifferenceMode::Forward => return Ok(self.get_forward_difference_multi_variable(order, func, idx_to_derivate, point, self.step_size)),
            mode::FiniteDifferenceMode::Backward => return Ok(self.get_backward_difference_multi_variable(order, func, idx_to_derivate, point, self.step_size)),
            mode::FiniteDifferenceMode::Central => return Ok(self.get_central_difference_multi_variable(order, func, idx_to_derivate, point, self.step_size)) 
        }    
    }
}