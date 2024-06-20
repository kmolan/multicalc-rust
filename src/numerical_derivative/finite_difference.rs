use crate::numerical_derivative::mode as mode;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;
use crate::numerical_derivative::derivator::*;

#[derive(Clone, Copy)]
pub struct SingleVariableSolver
{
    step_size: f64,
    method: mode::FiniteDifferenceMode
}

impl Default for SingleVariableSolver
{
    fn default() -> Self 
    {
        return SingleVariableSolver 
        { 
            step_size: mode::DEFAULT_STEP_SIZE,
            method: mode::FiniteDifferenceMode::Central
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

    pub fn from_parameters(step_size: f64, method: mode::FiniteDifferenceMode) -> Self 
    {
        SingleVariableSolver
        {
            step_size: step_size,
            method: method
        }    
    }

    fn get_forward_difference_single_variable<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T) -> T
    {
        if order == 1
        {
            let f0 = func(point);
            let f1 = func(point + T::from(self.step_size).unwrap());
            return (f1 - f0)/(T::from(self.step_size).unwrap());
        }

        let f0_point = point;
        let f0 = self.get_forward_difference_single_variable(order - 1, func, f0_point);

        let f1_point = point + T::from(self.step_size).unwrap();
        let f1 = self.get_forward_difference_single_variable(order - 1, func, f1_point);

        return (f1 - f0)/(T::from(self.step_size).unwrap());
    }

    fn get_backward_difference_single_variable<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T) -> T
    {
        if order == 1
        {
            let f0 = func(point - T::from(self.step_size).unwrap());
            let f1 = func(point);
            return (f1 - f0)/(T::from(self.step_size).unwrap());
        }

        let f0_point = point - T::from(self.step_size).unwrap();
        let f0 = self.get_backward_difference_single_variable(order - 1, func, f0_point);

        let f1_point = point;
        let f1 = self.get_backward_difference_single_variable(order - 1, func, f1_point);

        return (f1 - f0)/(T::from(self.step_size).unwrap());
    }


    fn get_central_difference_single_variable<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T) -> T
    {
        if order == 1
        {
            let f0 = func(point - T::from(self.step_size).unwrap());
            let f1 = func(point + T::from(self.step_size).unwrap());
            return (f1 - f0)/(T::from(2.0*self.step_size).unwrap());
        }

        let f0_point = point - T::from(self.step_size).unwrap();
        let f0 = self.get_central_difference_single_variable(order - 1, func, f0_point);

        let f1_point = point + T::from(self.step_size).unwrap();
        let f1 = self.get_central_difference_single_variable(order - 1, func, f1_point);

        return (f1 - f0)/(T::from(2.0*self.step_size).unwrap());
    }
}

impl DerivatorSingleVariable for SingleVariableSolver
{
    fn get<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T) -> Result<T, ErrorCode>
    {
        if self.step_size == 0.0
        {
            return Err(ErrorCode::NumberOfStepsCannotBeZero);
        }

        match self.method
        {
            mode::FiniteDifferenceMode::Forward => return Ok(self.get_forward_difference_single_variable(order, func, point)),
            mode::FiniteDifferenceMode::Backward => return Ok(self.get_backward_difference_single_variable(order, func, point)),
            mode::FiniteDifferenceMode::Central => return Ok(self.get_central_difference_single_variable(order, func, point)) 
        } 
    }
}

#[derive(Clone, Copy)]
pub struct MultiVariableSolver
{
    step_size: f64,
    method: mode::FiniteDifferenceMode
}

impl Default for MultiVariableSolver
{
    fn default() -> Self 
    {
        return MultiVariableSolver 
        { 
            step_size: mode::DEFAULT_STEP_SIZE,
            method: mode::FiniteDifferenceMode::Central
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

    pub fn from_parameters(step_size: f64, method: mode::FiniteDifferenceMode) -> Self 
    {
        MultiVariableSolver
        {
            step_size: step_size,
            method: method
        }    
    }

    fn get_forward_difference_multi_variable<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS]) -> T
    {
        if order == 1
        {
            let f0_args = point;

            let mut f1_args = *point;
            f1_args[idx_to_derivate[0]] = f1_args[idx_to_derivate[0]] + T::from(self.step_size).unwrap(); 

            let f0 = func(f0_args);
            let f1 = func(&f1_args);

            return (f1 - f0)/T::from(self.step_size).unwrap();
        }

        let f0_args = point;

        let mut f1_args = *point;
        f1_args[idx_to_derivate[order - 2]] = f1_args[idx_to_derivate[order - 2]] + T::from(self.step_size).unwrap(); 

        let f0 = self.get_forward_difference_multi_variable(order - 1, func, idx_to_derivate, f0_args);
        let f1 = self.get_forward_difference_multi_variable(order - 1, func, idx_to_derivate, &f1_args);

        return (f1 - f0)/T::from(self.step_size).unwrap();
    }

    fn get_backward_difference_multi_variable<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS]) -> T
    {
        if order == 1
        {
            let mut f0_args = *point;
            f0_args[idx_to_derivate[0]] = f0_args[idx_to_derivate[0]] - T::from(self.step_size).unwrap(); 

            let f1_args = point;

            let f0 = func(&f0_args);
            let f1 = func(f1_args);

            return (f1 - f0)/T::from(self.step_size).unwrap();
        }

        let mut f0_args = *point;
        f0_args[idx_to_derivate[order - 2]] = f0_args[idx_to_derivate[order - 2]] - T::from(self.step_size).unwrap(); 

        let f1_args = point;

        let f0 = self.get_backward_difference_multi_variable(order - 1, func, idx_to_derivate, &f0_args);
        let f1 = self.get_backward_difference_multi_variable(order - 1, func, idx_to_derivate, f1_args);

        return (f1 - f0)/T::from(self.step_size).unwrap();
    }

    fn get_central_difference_multi_variable<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS]) -> T
    {
        if order == 1
        {
            let mut f0_args = *point;
            f0_args[idx_to_derivate[0]] = f0_args[idx_to_derivate[0]] - T::from(self.step_size).unwrap();

            let mut f1_args = *point;
            f1_args[idx_to_derivate[0]] = f1_args[idx_to_derivate[0]] + T::from(self.step_size).unwrap(); 

            let f0 = func(&f0_args);
            let f1 = func(&f1_args);

            return (f1 - f0)/(T::from(2.0*self.step_size).unwrap());
        }

        let mut f0_point = *point;
        f0_point[idx_to_derivate[order - 2]] = f0_point[idx_to_derivate[order - 2]] - T::from(self.step_size).unwrap();

        let f0 = self.get_central_difference_multi_variable(order - 1, func, idx_to_derivate, &f0_point);

        let mut f1_point = *point;
        f1_point[idx_to_derivate[order - 2]] = f1_point[idx_to_derivate[order - 2]] + T::from(self.step_size).unwrap();

        let f1 = self.get_central_difference_multi_variable(order - 1, func, idx_to_derivate, &f1_point);

        return (f1 - f0)/(T::from(2.0*self.step_size).unwrap());
    }
}

impl DerivatorMultiVariable for MultiVariableSolver
{
    fn get<T: ComplexFloat, const NUM_VARS: usize, const NUM_ORDER: usize>(&self, order: usize, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; NUM_ORDER], point: &[T; NUM_VARS]) -> Result<T, ErrorCode> 
    {
        if self.step_size == 0.0
        {
            return Err(ErrorCode::NumberOfStepsCannotBeZero);
        }
        if order == 0
        {
            return Err(ErrorCode::DerivateOrderCannotBeZero);
        }
        if order >= NUM_ORDER
        {
            return Err(ErrorCode::DerivateOrderOutOfrange);
        }
        if NUM_ORDER > NUM_VARS
        {
            return Err(ErrorCode::IndexToDerivativeOutOfRange);
        }

        match self.method
        {
            mode::FiniteDifferenceMode::Forward => return Ok(self.get_forward_difference_multi_variable(order, func, idx_to_derivate, point)),
            mode::FiniteDifferenceMode::Backward => return Ok(self.get_backward_difference_multi_variable(order, func, idx_to_derivate, point)),
            mode::FiniteDifferenceMode::Central => return Ok(self.get_central_difference_multi_variable(order, func, idx_to_derivate, point)) 
        }    
    }
}