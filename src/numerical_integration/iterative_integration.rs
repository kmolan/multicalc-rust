
use num_complex::ComplexFloat;
use crate::numerical_integration::integrator::*;
use crate::numerical_integration::mode::IterativeMethod;
use crate::utils::error_codes::*;

pub const DEFAULT_TOTAL_ITERATIONS: u64 = 50;

#[derive(Clone, Copy)]
pub struct SingleVariableSolver
{
    total_iterations: u64,
    integration_method: IterativeMethod
}

impl Default for SingleVariableSolver
{
    fn default() -> Self 
    {
        return SingleVariableSolver { total_iterations: DEFAULT_TOTAL_ITERATIONS, integration_method: IterativeMethod::Booles };
    }
}

impl SingleVariableSolver
{
    pub fn get_total_iterations(&self) -> u64
    {
        return self.total_iterations;
    }

    pub fn set_total_iterations(&mut self, total_iterations: u64) 
    {
        self.total_iterations = total_iterations;
    }

    pub fn get_integration_method(&self) -> IterativeMethod
    {
        return self.integration_method;
    }

    pub fn set_integration_method(&mut self, integration_method: IterativeMethod)
    {
        self.integration_method = integration_method;
    }

    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self 
    {
        SingleVariableSolver
        {
            total_iterations: total_iterations,
            integration_method: integration_method
        }    
    }

    fn check_for_errors<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> Result<(), &'static str> 
    {
        if self.total_iterations == 0
        {
            return Err(INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS);
        }

        for iter in 0..integration_limit.len()
        {
            if integration_limit[iter][0].abs() >= integration_limit[iter][1].abs()
            {
                return Err(INTEGRATION_LIMITS_ILL_DEFINED);
            }
        }

        if NUM_INTEGRATIONS != number_of_integrations
        {
            return Err(INCORRECT_NUMBER_OF_INTEGRATION_LIMITS)
        }        

        return Ok(());        
    }

    fn get_booles<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut current_point = integration_limit[0][0];

            let mut ans = T::from(7.0).unwrap()*func(current_point);
            let delta = (integration_limit[0][1] - integration_limit[0][0])/(T::from(self.total_iterations).unwrap());

            let mut multiplier = T::from(32.0).unwrap();

            for iter in 0..self.total_iterations-1
            {
                current_point = current_point + delta;
                ans = ans + multiplier*func(current_point);
                
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

            current_point = integration_limit[0][1];

            ans = ans + T::from(7.0).unwrap()*func(current_point);

            return T::from(2.0).unwrap()*delta*ans/T::from(45.0).unwrap();
        }

        let mut current_point = integration_limit[number_of_integrations - 1][0];

        let mut ans = T::from(7.0).unwrap()*self.get_booles(number_of_integrations - 1, func, integration_limit);
        let delta = (integration_limit[number_of_integrations - 1][1] - integration_limit[number_of_integrations - 1][0])/(T::from(self.total_iterations).unwrap());

        let mut multiplier = T::from(32.0).unwrap();

        for iter in 0..self.total_iterations-1
        {
            current_point = current_point + delta;
            ans = ans + multiplier*self.get_booles(number_of_integrations - 1, func, integration_limit);
            
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

        //current_point = integration_limit[1];

        ans = ans + T::from(7.0).unwrap()*self.get_booles(number_of_integrations - 1, func, integration_limit);

        return T::from(2.0).unwrap()*delta*ans/T::from(45.0).unwrap();
    }


    //TODO: add the 1/3 rule also
    //the 3/8 rule, better than the 1/3 rule
    fn get_simpsons<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut current_point = integration_limit[0][0];

            let mut ans = func(current_point);
            let delta = (integration_limit[0][1] - integration_limit[0][0])/(T::from(self.total_iterations).unwrap());

            let mut multiplier = T::from(3.0).unwrap();

            for iter in 0..self.total_iterations-1
            {
                current_point = current_point + delta;
                ans = ans + multiplier*func(current_point);        

                if (iter + 2) % 3 == 0
                {
                    multiplier = T::from(2.0).unwrap();
                }
                else
                {
                    multiplier = T::from(3.0).unwrap();
                }
            }

            current_point = integration_limit[0][1];

            ans = ans + func(current_point);

            return T::from(3.0).unwrap()*delta*ans/T::from(8.0).unwrap();
        }

        let mut current_point = integration_limit[number_of_integrations - 1][0];

        let mut ans = self.get_simpsons(number_of_integrations - 1, func, integration_limit);
        let delta = (integration_limit[number_of_integrations - 1][1] - integration_limit[number_of_integrations - 1][0])/(T::from(self.total_iterations).unwrap());

        let mut multiplier = T::from(3.0).unwrap();

        for iter in 0..self.total_iterations-1
        {
            current_point = current_point + delta;
            ans = ans + multiplier*self.get_simpsons(number_of_integrations - 1, func, integration_limit);     

            if (iter + 2) % 3 == 0
            {
                multiplier = T::from(2.0).unwrap();
            }
            else
            {
                multiplier = T::from(3.0).unwrap();
            }
        }

        //current_point = integration_limit[1];

        ans = ans + self.get_simpsons(number_of_integrations - 1, func, integration_limit);

        return T::from(3.0).unwrap()*delta*ans/T::from(8.0).unwrap();
    }

    fn get_trapezoidal<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut current_point = integration_limit[0][0];

            let mut ans = func(current_point);
            let delta = (integration_limit[0][1] - integration_limit[0][0])/(T::from(self.total_iterations).unwrap());

            for _ in 0..self.total_iterations-1
            {
                current_point = current_point + delta;
                ans = ans + T::from(2.0).unwrap()*func(current_point);        
            }
            
            current_point = integration_limit[0][1];

            ans = ans + func(current_point);

            return T::from(0.5).unwrap()*delta*ans;
        }

        let mut current_point = integration_limit[number_of_integrations - 1][0];

        let mut ans = self.get_trapezoidal(number_of_integrations - 1, func, integration_limit);
        let delta = (integration_limit[number_of_integrations - 1][1] - integration_limit[number_of_integrations - 1][0])/(T::from(self.total_iterations).unwrap());

        for _ in 0..self.total_iterations-1
        {
            current_point = current_point + delta;
            ans = ans + T::from(2.0).unwrap()*self.get_trapezoidal(number_of_integrations - 1, func, integration_limit);        
        }
        
        //current_point = integration_limit[1];

        ans = ans + self.get_trapezoidal(number_of_integrations - 1, func, integration_limit);

        return T::from(0.5).unwrap()*delta*ans;
    }
}

impl IntegratorSingleVariable for SingleVariableSolver
{
    fn get<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> Result<T, &'static str> 
    {
        self.check_for_errors(number_of_integrations, integration_limit)?;

        match self.integration_method
        {
            IterativeMethod::Booles        => return Ok(self.get_booles(number_of_integrations, func, integration_limit)),
            IterativeMethod::Simpsons      => return Ok(self.get_simpsons(number_of_integrations, func, integration_limit)),
            IterativeMethod::Trapezoidal   => return Ok(self.get_trapezoidal(number_of_integrations, func, integration_limit))
        }        
    }
}

#[derive(Clone, Copy)]
pub struct MultiVariableSolver
{
    total_iterations: u64,
    integration_method: IterativeMethod
}

impl Default for MultiVariableSolver
{
    fn default() -> Self 
    {
        return MultiVariableSolver { total_iterations: DEFAULT_TOTAL_ITERATIONS, integration_method: IterativeMethod::Booles };
    }
}

impl MultiVariableSolver
{
    pub fn get_total_iterations(&self) -> u64
    {
        return self.total_iterations;
    }

    pub fn set_total_iterations(&mut self, total_iterations: u64) 
    {
        self.total_iterations = total_iterations;
    }

    pub fn get_integration_method(&self) -> IterativeMethod
    {
        return self.integration_method;
    }

    pub fn set_integration_method(&mut self, integration_method: IterativeMethod)
    {
        self.integration_method = integration_method;
    }

    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self 
    {
        MultiVariableSolver
        {
            total_iterations: total_iterations,
            integration_method: integration_method
        }    
    }

    fn check_for_errors<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> Result<(), &'static str> 
    {
        if self.total_iterations == 0
        {
            return Err(INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS);
        }

        for iter in 0..integration_limit.len()
        {
            if integration_limit[iter][0].abs() >= integration_limit[iter][1].abs()
            {
                return Err(INTEGRATION_LIMITS_ILL_DEFINED);
            }
        }

        if NUM_INTEGRATIONS != number_of_integrations
        {
            return Err(INCORRECT_NUMBER_OF_INTEGRATION_LIMITS)
        }        

        return Ok(());        
    }


    fn get_booles<T: ComplexFloat, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; NUM_INTEGRATIONS], point: &[T; NUM_VARS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut current_vec = *point;
            current_vec[idx_to_integrate[0]] = integration_limits[0][0];

            let mut ans = T::from(7.0).unwrap()*func(&current_vec);
            let delta = (integration_limits[0][1] - integration_limits[0][0])/(T::from(self.total_iterations).unwrap());

            let mut multiplier = T::from(32.0).unwrap();

            for iter in 0..self.total_iterations-1
            {
                current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
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

            current_vec[idx_to_integrate[0]] = integration_limits[0][1];

            ans = ans + T::from(7.0).unwrap()*func(&current_vec);

            return T::from(2.0).unwrap()*delta*ans/T::from(45.0).unwrap();
        }

        let mut current_vec = *point;
        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][0];

        let mut ans = T::from(7.0).unwrap()*self.get_booles(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);
        let delta = (integration_limits[number_of_integrations - 1][1] - integration_limits[number_of_integrations - 1][0])/(T::from(self.total_iterations).unwrap());

        let mut multiplier = T::from(32.0).unwrap();

        for iter in 0..self.total_iterations-1
        {
            current_vec[idx_to_integrate[number_of_integrations - 1]] = current_vec[idx_to_integrate[number_of_integrations - 1]] + delta;
            ans = ans + multiplier*self.get_booles(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);
            
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

        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][1];

        ans = ans + T::from(7.0).unwrap()*self.get_booles(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);

        return T::from(2.0).unwrap()*delta*ans/T::from(45.0).unwrap();
    }

    //TODO: add the 1/3 rule also
    //the 3/8 rule, better than the 1/3 rule
    fn get_simpsons<T: ComplexFloat, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; NUM_INTEGRATIONS], point: &[T; NUM_VARS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut current_vec = *point;
            current_vec[idx_to_integrate[0]] = integration_limits[0][0];

            let mut ans = func(&current_vec);
            let delta = (integration_limits[0][1] - integration_limits[0][0])/(T::from(self.total_iterations).unwrap());

            let mut multiplier = T::from(3.0).unwrap();

            for iter in 0..self.total_iterations-1
            {
                current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
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

            current_vec[idx_to_integrate[0]] = integration_limits[0][1];

            ans = ans + func(&current_vec);

            return T::from(3.0).unwrap()*delta*ans/T::from(8.0).unwrap();
        }

        let mut current_vec = *point;
        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][0];

        let mut ans = self.get_simpsons(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);
        let delta = (integration_limits[number_of_integrations - 1][1] - integration_limits[number_of_integrations - 1][0])/(T::from(self.total_iterations).unwrap());

        let mut multiplier = T::from(3.0).unwrap();

        for iter in 0..self.total_iterations-1
        {
            current_vec[idx_to_integrate[number_of_integrations - 1]] = current_vec[idx_to_integrate[number_of_integrations - 1]] + delta;
            ans = ans + multiplier*self.get_simpsons(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);        

            if (iter + 2) % 3 == 0
            {
                multiplier = T::from(2.0).unwrap();
            }
            else
            {
                multiplier = T::from(3.0).unwrap();
            }
        }

        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][1];

        ans = ans + self.get_simpsons(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);

        return T::from(3.0).unwrap()*delta*ans/T::from(8.0).unwrap();
    }

    fn get_trapezoidal<T: ComplexFloat, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; NUM_INTEGRATIONS], point: &[T; NUM_VARS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut current_vec = *point;
            current_vec[idx_to_integrate[0]] = integration_limits[0][0];

            let mut ans = func(&current_vec);
            let delta = (integration_limits[0][1] - integration_limits[0][0])/(T::from(self.total_iterations).unwrap());

            for _ in 0..self.total_iterations-1
            {
                current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
                ans = ans + T::from(2.0).unwrap()*func(&current_vec);        
            }
            
            current_vec[idx_to_integrate[0]] = integration_limits[0][1];

            ans = ans + func(&current_vec);

            return T::from(0.5).unwrap()*delta*ans;
        }

        let mut current_vec = *point;
        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][0];

        let mut ans = self.get_trapezoidal(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);
        let delta = (integration_limits[number_of_integrations - 1][1] - integration_limits[number_of_integrations - 1][0])/(T::from(self.total_iterations).unwrap());

        for _ in 0..self.total_iterations-1
        {
            current_vec[idx_to_integrate[number_of_integrations - 1]] = current_vec[idx_to_integrate[number_of_integrations - 1]] + delta;
            ans = ans + T::from(2.0).unwrap()*self.get_trapezoidal(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);        
        }
        
        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][1];

        ans = ans + self.get_trapezoidal(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);

        return T::from(0.5).unwrap()*delta*ans;
    }
}

impl IntegratorMultiVariable for MultiVariableSolver
{
    fn get<T: ComplexFloat, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; NUM_INTEGRATIONS], point: &[T; NUM_VARS]) -> Result<T, &'static str> 
    {
        self.check_for_errors(number_of_integrations, integration_limits)?;

        match self.integration_method
        {
            IterativeMethod::Booles =>      return Ok(self.get_booles(number_of_integrations, idx_to_integrate, func, integration_limits, point)),
            IterativeMethod::Simpsons =>    return Ok(self.get_simpsons(number_of_integrations, idx_to_integrate, func, integration_limits, point)),
            IterativeMethod::Trapezoidal => return Ok(self.get_trapezoidal(number_of_integrations, idx_to_integrate, func, integration_limits, point))
        }
    }
}

/* 
pub struct Iterative
{
    total_iterations: u64,
    integration_method: mode::IterativeMethod
}

impl Default for Iterative
{
    fn default() -> Self 
    {
        return Iterative { total_iterations: DEFAULT_TOTAL_ITERATIONS, integration_method: mode::IterativeMethod::Trapezoidal };
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

    fn check_for_errors<T: ComplexFloat>(&self, integration_limit: &[T; 2]) -> Result<(), &'static str> 
    {
        if integration_limit[0].abs() >= integration_limit[1].abs()
        {
            return Err(&'static str::IntegrationLimitsIllDefined);
        }
        if self.total_iterations == 0
        {
            return Err(&'static str::NumberOfStepsCannotBeZero);
        }

        return Ok(());        
    }

    fn get_booles_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
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
    fn get_simpsons_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
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

    fn get_trapezoidal_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
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


    fn get_booles_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
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


    fn get_simpsons_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
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

    fn get_trapezoidal_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
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
    fn get_single_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limit: &[T; 2]) -> Result<T, &'static str>
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

    fn get_single_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
    {
        self.check_for_errors(integration_limit)?;

        match self.integration_method
        {
            mode::IterativeMethod::Booles        => return self.get_booles_1(func, idx_to_integrate, integration_limit, &point),
            mode::IterativeMethod::Simpsons      => return self.get_simpsons_1(func, idx_to_integrate, integration_limit, &point),
            mode::IterativeMethod::Trapezoidal   => return self.get_trapezoidal_1(func, idx_to_integrate, integration_limit, &point)
        }
    }

    fn get_double_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; 2]) -> Result<T, &'static str> 
    {
        let point = [integration_limits[0][1]; NUM_VARS];

        match self.integration_method
        {
            mode::IterativeMethod::Booles        => return self.get_booles_2(func, [0, 0], integration_limits, &point),
            mode::IterativeMethod::Simpsons      => return self.get_simpsons_2(func, [0, 0], integration_limits, &point),
            mode::IterativeMethod::Trapezoidal   => return self.get_trapezoidal_2(func, [0, 0], integration_limits, &point)
        }
    }

    fn get_double_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str> 
    {
        match self.integration_method
        {
            mode::IterativeMethod::Booles        => return self.get_booles_2(func, idx_to_integrate, integration_limits, point),
            mode::IterativeMethod::Simpsons      => return self.get_simpsons_2(func, idx_to_integrate, integration_limits, point),
            mode::IterativeMethod::Trapezoidal   => return self.get_trapezoidal_2(func, idx_to_integrate, integration_limits, point)
        }
    }
}
*/