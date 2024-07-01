use crate::numerical_derivative::mode; 
use crate::utils::error_codes::*;

use num_complex::ComplexFloat;
use crate::numerical_derivative::derivator::*;

///Implements the finite difference method for numerical differentation for single variable functions
#[derive(Clone, Copy)]
pub struct SingleVariableSolver
{
    step_size: f64,
    method: mode::FiniteDifferenceMode,

    //the step size will be multipled by this factor after every iteration. Only matters for triple derivatives or higher
    step_size_multiplier: f64
}

impl Default for SingleVariableSolver
{
    ///default constructor, choose this for optimal results for most generic equations
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
    ///Returns the step size
    pub fn get_step_size(&self) -> f64
    {
        return self.step_size;
    }

    ///Sets the step size
    pub fn set_step_size(&mut self, step_size: f64) 
    {
        self.step_size = step_size;
    }

    ///Returns the chosen method of differentiation
    ///Possible choices are: Forward step, backward step and central step
    pub fn get_method(&self) -> mode::FiniteDifferenceMode
    {
        return self.method;
    }

    ///Sets the method of differentiation
    ///Possible choices are: Forward step, backward step and central step
    pub fn set_method(&mut self, method: mode::FiniteDifferenceMode)
    {
        self.method = method;
    }

    ///Returns the chosen step size multiplier
    pub fn get_step_size_multiplier(&self) -> f64
    {
        return self.step_size_multiplier;
    }

    ///Sets the chosen step size multiplier. The step size will
    /// be multiplied by this factor after every iteration
    /// This parameter only matters if you are interested in triple derivatives or higher
    pub fn set_step_size_multiplier(&mut self, multiplier: f64)
    {
        self.step_size_multiplier = multiplier;
    }

    ///custom constructor, choose this for tweaking parameters if computing solutions for complex equations
    /// step: the desired step size for each iteration
    /// method: the desired method of differentiation: forward step, backward step or central step
    /// multiplier: default is 10.0, this is the factor by which we multiply the step size with on each iteration.
    ///             Only matters for triple derivatives or higher
    pub fn from_parameters(step: f64, method: mode::FiniteDifferenceMode, multiplier: f64) -> Self 
    {
        SingleVariableSolver
        {
            step_size: step,
            method: method,
            step_size_multiplier: multiplier
        }    
    }

    ///Returns the forward difference numerical differentiation for single variable functions
    ///computes f'(x) = (f(x + h) - f(x))/h, where h is the chosen step size
    /// you can control how many times to differentiate using the "order" parameter
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

    ///Returns the backward difference numerical differentiation for single variable functions
    ///computes f'(x) = (f(x) - f(x - h))/h, where h is the chosen step size
    /// you can control how many times to differentiate using the "order" parameter
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


    ///Returns the central difference numerical differentiation for single variable functions
    ///computes f'(x) = (f(x + h) - f(x - h))/2h, where h is the chosen step size
    /// you can control how many times to differentiate using the "order" parameter
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
    /// Returns the numerical differentiation value for a single variable function
    /// order: number of times the equation should be differentiated
    /// func: the single variable function
    /// point: the point of interest around which we want to differentiate
    /// 
    /// NOTE: Returns a Result<T, &'static str>
    /// Possible &'static str are:
    /// NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO -> if the step size value is zero
    /// DERIVATE_ORDER_CANNOT_BE_ZERO -> if the 'order' argument is zero
    /// 
    /// assume we want to differentiate f(x) = x^3. the function would be:
    /// ```
    ///    let my_func = | arg: f64 | -> f64 
    ///    { 
    ///        return arg*arg*arg;
    ///    };
    ///
    /// let point = 2.0; //the point at which we want to differentiate
    /// 
    /// use multicalc::numerical_derivative::derivator::*;
    /// use multicalc::numerical_derivative::finite_difference::*;
    /// 
    /// let derivator = SingleVariableSolver::default();
    /// let val = derivator.get(1, &my_func, point).unwrap(); //single derivative
    /// assert!(f64::abs(val - 12.0) < 1e-7);
    /// let val = derivator.get(2, &my_func, point).unwrap(); //double derivative
    /// assert!(f64::abs(val - 12.0) < 1e-5);
    /// let val = derivator.get(3, &my_func, point).unwrap(); //triple derivative
    /// assert!(f64::abs(val - 6.0) < 1e-3);
    /// 
    ///```
    ///// Note that the accuracy of approximations fall with every derivative. This can be fine-tuned for each case 
    /// using an appropriate starting step size and a step size multiplier
    fn get<T: ComplexFloat>(&self, order: usize, func: &dyn Fn(T) -> T, point: T) -> Result<T, &'static str>
    {
        if order == 0
        {
            return Err(DERIVATE_ORDER_CANNOT_BE_ZERO);
        }

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

///Implements the finite difference method for numerical differentation for multi-variable functions
#[derive(Clone, Copy)]
pub struct MultiVariableSolver
{
    step_size: f64,
    method: mode::FiniteDifferenceMode,

    //the step size will be multiplied by this factor after every iteration. Only matters for triple derivatives or higher
    step_size_multiplier: f64
}

impl Default for MultiVariableSolver
{
    ///default constructor, choose this for optimal results for most generic equations
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
    ///Returns the step size
    pub fn get_step_size(&self) -> f64
    {
        return self.step_size;
    }

    ///Sets the step size
    pub fn set_step_size(&mut self, step_size: f64) 
    {
        self.step_size = step_size;
    }

    ///Returns the chosen method of differentiation
    ///Possible choices are: Forward step, backward step and central step
    pub fn get_method(&self) -> mode::FiniteDifferenceMode
    {
        return self.method;
    }

    ///Sets the method of differentiation
    ///Possible choices are: Forward step, backward step and central step
    pub fn set_method(&mut self, method: mode::FiniteDifferenceMode)
    {
        self.method = method;
    }

    ///Returns the chosen step size multiplier.
    pub fn get_step_size_multiplier(&self) -> f64
    {
        return self.step_size_multiplier;
    }

    ///Sets the chosen step size multiplier. The step size will
    /// be multiplied by this factor after every iteration
    /// This parameter only matters if you are interested in triple derivatives or higher
    pub fn set_step_size_multiplier(&mut self, multiplier: f64)
    {
        self.step_size_multiplier = multiplier;
    }

    ///custom constructor, choose this for tweaking parameters if computing solutions for complex equations
    /// step: the desired step size for each iteration
    /// method: the desired method of differentiation: forward step, backward step or central step
    /// multiplier: default is 10.0, this is the factor by which we multiply the step size with on each iteration.
    ///             Only matters for triple derivatives or higher
    pub fn from_parameters(step: f64, method: mode::FiniteDifferenceMode, multiplier: f64) -> Self 
    {
        MultiVariableSolver
        {
            step_size: step,
            method: method,
            step_size_multiplier: multiplier
        }    
    }

    ///Returns the partial forward difference numerical differentiation for multi variable functions
    ///computes f'(X) = (f(X + h) - f(X))/h, where h is the chosen step size
    /// you can control how many times to differentiate using the "order" parameter
    /// you can specify the variable(s) whose respect to the equation needs to be differentiated using the 'idx_to_derivate' parameter
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

    ///Returns the partial backward difference numerical differentiation for multi variable functions
    ///computes f'(X) = (f(X) - f(X - h))/h, where h is the chosen step size
    /// you can control how many times to differentiate using the "order" parameter
    /// you can specify the variable(s) whose respect to the equation needs to be differentiated using the 'idx_to_derivate' parameter
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

    ///Returns the partial central difference numerical differentiation for multi variable functions
    ///computes f'(X) = (f(X + h) - f(X - h))/2h, where h is the chosen step size
    /// you can control how many times to differentiate using the "order" parameter
    /// you can specify the variable(s) whose respect to the equation needs to be differentiated using the 'idx_to_derivate' parameter
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
    /// Returns the numerical differentiation value for a multi variable function
    /// order: number of times the equation should be differentiated
    /// func: the multi variable function
    /// idx_to_derivate: The variable index/indices whose respect to we want to differentiate
    /// point: the point of interest around which we want to differentiate
    /// 
    /// NOTE: Returns a Result<T, &'static str>
    /// Possible &'static str are:
    /// NUMBER_OF_DERIVATIVE_STEPS_CANNOT_BE_ZERO -> if the step size value is zero
    /// DERIVATE_ORDER_CANNOT_BE_ZERO -> if the 'order' argument is zero
    /// INDEX_TO_DERIVATE_ILL_FORMED -> if size of 'idx_to_derivate' argument is not equal to the 'order' argument
    /// 
    /// assume we want to differentiate f(x,y,z) = y*sin(x) + x*cos(y) + x*y*e^z. the function would be:
    /// ```
    ///    let my_func = | args: &[f64; 3] | -> f64 
    ///    { 
    ///        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    ///    };
    ///
    /// let point = [1.0, 2.0, 3.0]; //the point at which we want to differentiate
    /// 
    /// 
    /// use multicalc::numerical_derivative::derivator::*;
    /// use multicalc::numerical_derivative::finite_difference::*;
    /// 
    /// let derivator = MultiVariableSolver::default();
    /// 
    /// let idx: [usize; 2] = [0, 1]; //mixed partial double derivate d(df/dx)/dy
    /// let val = derivator.get(2, &my_func, &idx, &point).unwrap();
    /// let expected_value = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
    /// assert!(f64::abs(val - expected_value) < 0.001);
    /// 
    /// let idx: [usize; 2] = [1, 1]; //partial double derivative d(df/dy)/dy 
    ///let val = derivator.get(2, &my_func, &idx, &point).unwrap();
    ///let expected_value = -1.0*f64::cos(2.0);
    ///assert!(f64::abs(val - expected_value) < 0.0001);
    ///```
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