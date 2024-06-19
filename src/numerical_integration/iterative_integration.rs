use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;
use crate::numerical_integration::integrator::Integrator;
use crate::numerical_integration::mode;

pub struct Iterative
{
    total_iterations: u64,
    integration_method: mode::IterativeMethod
}

impl Default for Iterative
{
    fn default() -> Self 
    {
        return Iterative { total_iterations: mode::DEFAULT_TOTAL_ITERATIONS, integration_method: mode::IterativeMethod::Trapezoidal };
    }
}

impl Iterative
{
    pub fn get_total_iterations(&self) -> u64
    {
        return self.total_iterations;
    }

    pub fn set_total_iterations(&mut self, total_iterations: u64) 
    {
        self.total_iterations = total_iterations;
    }

    pub fn get_integration_method(&self) -> mode::IterativeMethod
    {
        return self.integration_method;
    }

    pub fn set_integration_method(&mut self, integration_method: mode::IterativeMethod)
    {
        self.integration_method = integration_method;
    }

    pub fn with_parameters(total_iterations: u64, integration_method: mode::IterativeMethod) -> Self 
    {
        Iterative
        {
            total_iterations: total_iterations,
            integration_method: integration_method
        }    
    }

    fn check_for_errors<T: ComplexFloat>(&self, integration_limit: &[T; 2]) -> Result<(), ErrorCode> 
    {
        if integration_limit[0].abs() >= integration_limit[1].abs()
        {
            return Err(ErrorCode::IntegrationLimitsIllDefined);
        }
        if self.total_iterations == 0
        {
            return Err(ErrorCode::NumberOfStepsCannotBeZero);
        }

        return Ok(());        
    }

    fn get_booles_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let mut current_vec = *point;
        current_vec[idx_to_integrate] = integration_limit[0];

        let mut ans = T::from(7.0).unwrap()*func(&current_vec);
        let delta = (integration_limit[1] - integration_limit[0])/(T::from(self.total_iterations).unwrap());

        let mut multiplier = T::from(32.0).unwrap();

        for iter in 0..self.total_iterations-1
        {
            current_vec[idx_to_integrate] = current_vec[idx_to_integrate] + delta;
            ans = ans + multiplier*func(&current_vec);
            
            if (iter + 2) % 2 != 0
            {
                multiplier = T::from(32.0).unwrap();
            }
            else if (iter + 2) % 4 == 0
            {
                multiplier = T::from(14.0).unwrap();
            }
            else
            {
                multiplier = T::from(12.0).unwrap();
            }
        }

        current_vec[idx_to_integrate] = integration_limit[1];

        ans = ans + T::from(7.0).unwrap()*func(&current_vec);

        return Ok(T::from(2.0).unwrap()*delta*ans/T::from(45.0).unwrap());
    }


    //TODO: add the 1/3 rule also
    //the 3/8 rule, better than the 1/3 rule
    fn get_simpsons_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let mut current_vec = *point;
        current_vec[idx_to_integrate] = integration_limit[0];

        let mut ans = func(&current_vec);
        let delta = (integration_limit[1] - integration_limit[0])/(T::from(self.total_iterations).unwrap());

        let mut multiplier = T::from(3.0).unwrap();

        for iter in 0..self.total_iterations-1
        {
            current_vec[idx_to_integrate] = current_vec[idx_to_integrate] + delta;
            ans = ans + multiplier*func(&current_vec);        

            if (iter + 2) % 3 == 0
            {
                multiplier = T::from(2.0).unwrap();
            }
            else
            {
                multiplier = T::from(3.0).unwrap();
            }
        }

        current_vec[idx_to_integrate] = integration_limit[1];

        ans = ans + func(&current_vec);

        return Ok(T::from(3.0).unwrap()*delta*ans/T::from(8.0).unwrap());
    }

    fn get_trapezoidal_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let mut current_vec = *point;
        current_vec[idx_to_integrate] = integration_limit[0];

        let mut ans = func(&current_vec);
        let delta = (integration_limit[1] - integration_limit[0])/(T::from(self.total_iterations).unwrap());

        for _ in 0..self.total_iterations-1
        {
            current_vec[idx_to_integrate] = current_vec[idx_to_integrate] + delta;
            ans = ans + T::from(2.0).unwrap()*func(&current_vec);        
        }
        
        current_vec[idx_to_integrate] = integration_limit[1];

        ans = ans + func(&current_vec);

        return Ok(T::from(0.5).unwrap()*delta*ans);
    }


    fn get_booles_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let total_iterations = self.get_total_iterations();

        let mut current_vec = *point;
        current_vec[idx_to_integrate[0]] = integration_limits[0][0];

        let mut ans = T::from(7.0).unwrap()*self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], &current_vec)?;

        let delta = (integration_limits[0][1] - integration_limits[0][0])/(T::from(total_iterations).unwrap());
        let mut multiplier = T::from(32.0).unwrap();

        for iter in 0..total_iterations-1
        {
            current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
            ans = ans + multiplier*self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], &current_vec)?;
            
            if (iter + 2) % 2 != 0
            {
                multiplier = T::from(32.0).unwrap();
            }
            else if (iter + 2) % 4 == 0
            {
                multiplier = T::from(14.0).unwrap();
            }
            else
            {
                multiplier = T::from(12.0).unwrap();
            }
        }

        current_vec[idx_to_integrate[0]] = integration_limits[0][1];

        ans = ans + T::from(7.0).unwrap()*self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], &current_vec)?;

        return Ok(T::from(2.0).unwrap()*delta*ans/T::from(45.0).unwrap());
    }


    fn get_simpsons_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let total_iterations = self.get_total_iterations();

        let mut current_vec = *point;
        current_vec[idx_to_integrate[0]] = integration_limits[0][0];

        let mut ans = self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], &current_vec)?;
        let delta = (integration_limits[0][1] - integration_limits[0][0])/(T::from(total_iterations).unwrap());

        let mut multiplier = T::from(3.0).unwrap();

        for iter in 0..total_iterations-1
        {
            current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
            ans = ans + multiplier*self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], &current_vec)?;        

            if (iter + 2) % 3 == 0
            {
                multiplier = T::from(2.0).unwrap();
            }
            else
            {
                multiplier = T::from(3.0).unwrap();
            }
        }

        current_vec[idx_to_integrate[0]] = integration_limits[0][1];

        ans = ans + self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], &current_vec)?;

        return Ok(T::from(3.0).unwrap()*delta*ans/T::from(8.0).unwrap());
    }

    fn get_trapezoidal_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        let total_iterations = self.get_total_iterations();

        let mut current_vec = *point;
        current_vec[idx_to_integrate[0]] = integration_limits[0][0];

        let mut ans = self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], &current_vec)?;

        let delta = (integration_limits[0][1] - integration_limits[0][0])/(T::from(total_iterations).unwrap());

        for _ in 0..total_iterations-1
        {
            current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
            ans = ans + T::from(2.0).unwrap()*self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], &current_vec)?;  
        }

        current_vec[idx_to_integrate[0]] = integration_limits[0][1];

        ans = ans + self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], &current_vec)?;

        return Ok(T::from(0.5).unwrap()*delta*ans);
    }
}

impl Integrator for Iterative
{
    fn get_single_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limit: &[T; 2]) -> Result<T, ErrorCode>
    {
        self.check_for_errors(integration_limit)?;

        let point = [integration_limit[1]; NUM_VARS];

        match self.integration_method
        {
            mode::IterativeMethod::Booles        => return self.get_booles_1(func, 0, integration_limit, &point),
            mode::IterativeMethod::Simpsons      => return self.get_simpsons_1(func, 0, integration_limit, &point),
            mode::IterativeMethod::Trapezoidal   => return self.get_trapezoidal_1(func, 0, integration_limit, &point)
        }
    }

    fn get_single_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        self.check_for_errors(integration_limit)?;

        match self.integration_method
        {
            mode::IterativeMethod::Booles        => return self.get_booles_1(func, idx_to_integrate, integration_limit, &point),
            mode::IterativeMethod::Simpsons      => return self.get_simpsons_1(func, idx_to_integrate, integration_limit, &point),
            mode::IterativeMethod::Trapezoidal   => return self.get_trapezoidal_1(func, idx_to_integrate, integration_limit, &point)
        }
    }

    fn get_double_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; 2]) -> Result<T, ErrorCode> 
    {
        let point = [integration_limits[0][1]; NUM_VARS];

        match self.integration_method
        {
            mode::IterativeMethod::Booles        => return self.get_booles_2(func, [0, 0], integration_limits, &point),
            mode::IterativeMethod::Simpsons      => return self.get_simpsons_2(func, [0, 0], integration_limits, &point),
            mode::IterativeMethod::Trapezoidal   => return self.get_trapezoidal_2(func, [0, 0], integration_limits, &point)
        }
    }

    fn get_double_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode> 
    {
        match self.integration_method
        {
            mode::IterativeMethod::Booles        => return self.get_booles_2(func, idx_to_integrate, integration_limits, point),
            mode::IterativeMethod::Simpsons      => return self.get_simpsons_2(func, idx_to_integrate, integration_limits, point),
            mode::IterativeMethod::Trapezoidal   => return self.get_trapezoidal_2(func, idx_to_integrate, integration_limits, point)
        }
    }
}