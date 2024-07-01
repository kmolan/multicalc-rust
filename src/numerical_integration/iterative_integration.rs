
use crate::numerical_integration::integrator::*;
use crate::numerical_integration::mode::IterativeMethod;
use crate::utils::error_codes::*;

pub const DEFAULT_TOTAL_ITERATIONS: u64 = 100;

///Implements the iterative methods for numerical integration for single variable functions
#[derive(Clone, Copy)]
pub struct SingleVariableSolver
{
    total_iterations: u64,
    integration_method: IterativeMethod
}

impl Default for SingleVariableSolver
{
    ///default constructor, optimal for most generic equations
    fn default() -> Self 
    {
        return SingleVariableSolver { total_iterations: DEFAULT_TOTAL_ITERATIONS, integration_method: IterativeMethod::Booles };
    }
}

impl SingleVariableSolver
{
    ///returns the total nuber of iterations
    pub fn get_total_iterations(&self) -> u64
    {
        return self.total_iterations;
    }

    ///sets the total nuber of iterations
    pub fn set_total_iterations(&mut self, total_iterations: u64) 
    {
        self.total_iterations = total_iterations;
    }

    ///returns the chosen integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn get_integration_method(&self) -> IterativeMethod
    {
        return self.integration_method;
    }

    ///sets the integration method
    ///choices are: Booles, Simpsons and Trapezoidal
    pub fn set_integration_method(&mut self, integration_method: IterativeMethod)
    {
        self.integration_method = integration_method;
    }

    ///custom constructor. Optimal for fine-tuning for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self 
    {
        SingleVariableSolver
        {
            total_iterations: total_iterations,
            integration_method: integration_method
        }    
    }

    ///Helper method to check if inputs are well defined
    fn check_for_errors<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> Result<(), &'static str> 
    {
        if self.total_iterations == 0
        {
            return Err(INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS);
        }

        for iter in 0..integration_limit.len()
        {
            if integration_limit[iter][0] >= integration_limit[iter][1]
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

    ///returns the numerical integration via Booles' method
    ///number_of_integrations: number of times the equation needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    fn get_booles<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut current_point = integration_limit[0][0];

            let mut ans = 7.0*func(current_point);
            let delta = (integration_limit[0][1] - integration_limit[0][0])/(self.total_iterations as f64);

            let mut multiplier = 32.0;

            for iter in 0..self.total_iterations-1
            {
                current_point = current_point + delta;
                ans = ans + multiplier*func(current_point);
                
                if (iter + 2) % 2 != 0
                {
                    multiplier = 32.0;
                }
                else if (iter + 2) % 4 == 0
                {
                    multiplier = 14.0;
                }
                else
                {
                    multiplier = 12.0;
                }
            }

            current_point = integration_limit[0][1];

            ans = ans + 7.0*func(current_point);

            return 2.0*delta*ans/45.0;
        }

        let mut current_point = integration_limit[number_of_integrations - 1][0];

        let mut ans = 7.0*self.get_booles(number_of_integrations - 1, func, integration_limit);
        let delta = (integration_limit[number_of_integrations - 1][1] - integration_limit[number_of_integrations - 1][0])/(self.total_iterations as f64);

        let mut multiplier = 32.0;

        for iter in 0..self.total_iterations-1
        {
            current_point = current_point + delta;
            ans = ans + multiplier*self.get_booles(number_of_integrations - 1, func, integration_limit);
            
            if (iter + 2) % 2 != 0
            {
                multiplier = 32.0;
            }
            else if (iter + 2) % 4 == 0
            {
                multiplier = 14.0;
            }
            else
            {
                multiplier = 12.0
            }
        }

        //current_point = integration_limit[1];

        ans = ans + 7.0*self.get_booles(number_of_integrations - 1, func, integration_limit);

        return 2.0*delta*ans/45.0;
    }


    ///returns the numerical integration via Simsons 3/8th method
    ///number_of_integrations: number of times the equation needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    fn get_simpsons<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut current_point = integration_limit[0][0];

            let mut ans = func(current_point);
            let delta = (integration_limit[0][1] - integration_limit[0][0])/(self.total_iterations as f64);

            let mut multiplier = 3.0;

            for iter in 0..self.total_iterations-1
            {
                current_point = current_point + delta;
                ans = ans + multiplier*func(current_point);        

                if (iter + 2) % 3 == 0
                {
                    multiplier = 2.0;
                }
                else
                {
                    multiplier = 3.0;
                }
            }

            current_point = integration_limit[0][1];

            ans = ans + func(current_point);

            return 3.0*delta*ans/8.0;
        }

        let mut current_point = integration_limit[number_of_integrations - 1][0];

        let mut ans = self.get_simpsons(number_of_integrations - 1, func, integration_limit);
        let delta = (integration_limit[number_of_integrations - 1][1] - integration_limit[number_of_integrations - 1][0])/(self.total_iterations as f64);

        let mut multiplier = 3.0;

        for iter in 0..self.total_iterations-1
        {
            current_point = current_point + delta;
            ans = ans + multiplier*self.get_simpsons(number_of_integrations - 1, func, integration_limit);     

            if (iter + 2) % 3 == 0
            {
                multiplier = 2.0;
            }
            else
            {
                multiplier = 3.0;
            }
        }

        //current_point = integration_limit[1];

        ans = ans + self.get_simpsons(number_of_integrations - 1, func, integration_limit);

        return 3.0*delta*ans/8.0;
    }

    ///returns the numerical integration via Trapezoidal method
    ///number_of_integrations: number of times the equation needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    fn get_trapezoidal<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut current_point = integration_limit[0][0];

            let mut ans = func(current_point);
            let delta = (integration_limit[0][1] - integration_limit[0][0])/(self.total_iterations as f64);

            for _ in 0..self.total_iterations-1
            {
                current_point = current_point + delta;
                ans = ans + 2.0*func(current_point);        
            }
            
            current_point = integration_limit[0][1];

            ans = ans + func(current_point);

            return 0.5*delta*ans;
        }

        let mut current_point = integration_limit[number_of_integrations - 1][0];

        let mut ans = self.get_trapezoidal(number_of_integrations - 1, func, integration_limit);
        let delta = (integration_limit[number_of_integrations - 1][1] - integration_limit[number_of_integrations - 1][0])/(self.total_iterations as f64);

        for _ in 0..self.total_iterations-1
        {
            current_point = current_point + delta;
            ans = ans + 2.0*self.get_trapezoidal(number_of_integrations - 1, func, integration_limit);        
        }
        
        //current_point = integration_limit[1];

        ans = ans + self.get_trapezoidal(number_of_integrations - 1, func, integration_limit);

        return 0.5*delta*ans;
    }
}

impl IntegratorSingleVariable for SingleVariableSolver
{
    ///returns the numeric integration value for a single variable function
    ///number_of_integrations: number of times the equation needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// 
    /// NOTE: Returns a Result<f64, &'static str>,
    /// where possible Err are:
    /// INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS -> if number_of_integrations is zero
    /// INTEGRATION_LIMITS_ILL_DEFINED -> if any integration_limit[i][0] >= integration_limit[i][1] for all possible i
    /// INCORRECT_NUMBER_OF_INTEGRATION_LIMITS -> if size of integration_limit is not equal to number_of_integrations
    /// 
    /// assume we want to integrate 2*x . the function would be:
    /// ```
    ///    let my_func = | arg: f64 | -> f64 
    ///    { 
    ///        return 2.0*arg;
    ///    };
    /// 
    /// use crate::multicalc::numerical_integration::integrator::*;
    /// use multicalc::numerical_integration::iterative_integration;
    ///
    /// let integrator = iterative_integration::SingleVariableSolver::default();  
    /// 
    /// let integration_limit = [[0.0, 2.0]; 1]; //desired integration limit 
    /// let val = integrator.get(1, &my_func, &integration_limit).unwrap(); //single integration
    /// assert!(f64::abs(val - 4.0) < 1e-6);
    /// 
    /// let integration_limit = [[0.0, 2.0], [-1.0, 1.0]]; //desired integration limits
    /// let val = integrator.get(2, &my_func, &integration_limit).unwrap(); //double integration
    /// assert!(f64::abs(val - 8.0) < 1e-6);
    ///
    /// let integration_limit = [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]; //desired integration limits
    /// let val = integrator.get(3, &my_func, &integration_limit).unwrap(); //triple integration
    /// assert!(f64::abs(val - 16.0) < 1e-6);
    ///```
    fn get<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> Result<f64, &'static str> 
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


///Implements the iterative methods for numerical integration for multi variable functions
#[derive(Clone, Copy)]
pub struct MultiVariableSolver
{
    total_iterations: u64,
    integration_method: IterativeMethod
}

impl Default for MultiVariableSolver
{
    ///default constructor, optimal for most generic equations
    fn default() -> Self 
    {
        return MultiVariableSolver { total_iterations: DEFAULT_TOTAL_ITERATIONS, integration_method: IterativeMethod::Booles };
    }
}

impl MultiVariableSolver
{
    ///returns the total number of iterations
    pub fn get_total_iterations(&self) -> u64
    {
        return self.total_iterations;
    }

    ///sets the total number of iterations
    pub fn set_total_iterations(&mut self, total_iterations: u64) 
    {
        self.total_iterations = total_iterations;
    }

    ///returns the chosen integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn get_integration_method(&self) -> IterativeMethod
    {
        return self.integration_method;
    }

    ///sets the integration method
    /// choices are: Booles, Simpsons and Trapezoidal
    pub fn set_integration_method(&mut self, integration_method: IterativeMethod)
    {
        self.integration_method = integration_method;
    }

    ///custom constructor, optimal for fine-tuning the integrator for more complex equations
    pub fn from_parameters(total_iterations: u64, integration_method: IterativeMethod) -> Self 
    {
        MultiVariableSolver
        {
            total_iterations: total_iterations,
            integration_method: integration_method
        }    
    }

    ///Helper method to check if inputs are well defined
    fn check_for_errors<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> Result<(), &'static str> 
    {
        if self.total_iterations == 0
        {
            return Err(INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS);
        }

        for iter in 0..integration_limit.len()
        {
            if integration_limit[iter][0] >= integration_limit[iter][1]
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


    ///returns the numerical integration via Booles' method
    ///number_of_integrations: number of times the equation needs to be integrated
    /// idx_to_integrate: the variables' index/indices that needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    fn get_booles<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[f64; NUM_VARS]) -> f64, integration_limits: &[[f64; 2]; NUM_INTEGRATIONS], point: &[f64; NUM_VARS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut current_vec = *point;
            current_vec[idx_to_integrate[0]] = integration_limits[0][0];

            let mut ans = 7.0*func(&current_vec);
            let delta = (integration_limits[0][1] - integration_limits[0][0])/(self.total_iterations as f64);

            let mut multiplier = 32.0;

            for iter in 0..self.total_iterations-1
            {
                current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
                ans = ans + multiplier*func(&current_vec);
                
                if (iter + 2) % 2 != 0
                {
                    multiplier = 32.0;
                }
                else if (iter + 2) % 4 == 0
                {
                    multiplier = 14.0;
                }
                else
                {
                    multiplier = 12.0;
                }
            }

            current_vec[idx_to_integrate[0]] = integration_limits[0][1];

            ans = ans + 7.0*func(&current_vec);

            return 2.0*delta*ans/45.0;
        }

        let mut current_vec = *point;
        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][0];

        let mut ans = 7.0*self.get_booles(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);
        let delta = (integration_limits[number_of_integrations - 1][1] - integration_limits[number_of_integrations - 1][0])/(self.total_iterations as f64);

        let mut multiplier = 32.0;

        for iter in 0..self.total_iterations-1
        {
            current_vec[idx_to_integrate[number_of_integrations - 1]] = current_vec[idx_to_integrate[number_of_integrations - 1]] + delta;
            ans = ans + multiplier*self.get_booles(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);
            
            if (iter + 2) % 2 != 0
            {
                multiplier = 32.0;
            }
            else if (iter + 2) % 4 == 0
            {
                multiplier = 14.0;
            }
            else
            {
                multiplier = 12.0;
            }
        }

        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][1];

        ans = ans + 7.0*self.get_booles(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);

        return 2.0*delta*ans/45.0;
    }

    ///returns the numerical integration via Simsons' 3/8th method
    ///number_of_integrations: number of times the equation needs to be integrated
    /// idx_to_integrate: the variables' index/indices that needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    fn get_simpsons<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[f64; NUM_VARS]) -> f64, integration_limits: &[[f64; 2]; NUM_INTEGRATIONS], point: &[f64; NUM_VARS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut current_vec = *point;
            current_vec[idx_to_integrate[0]] = integration_limits[0][0];

            let mut ans = func(&current_vec);
            let delta = (integration_limits[0][1] - integration_limits[0][0])/(self.total_iterations as f64);

            let mut multiplier = 3.0;

            for iter in 0..self.total_iterations-1
            {
                current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
                ans = ans + multiplier*func(&current_vec);        

                if (iter + 2) % 3 == 0
                {
                    multiplier = 2.0;
                }
                else
                {
                    multiplier = 3.0;
                }
            }

            current_vec[idx_to_integrate[0]] = integration_limits[0][1];

            ans = ans + func(&current_vec);

            return 3.0*delta*ans/8.0;
        }

        let mut current_vec = *point;
        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][0];

        let mut ans = self.get_simpsons(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);
        let delta = (integration_limits[number_of_integrations - 1][1] - integration_limits[number_of_integrations - 1][0])/(self.total_iterations as f64);

        let mut multiplier = 3.0;

        for iter in 0..self.total_iterations-1
        {
            current_vec[idx_to_integrate[number_of_integrations - 1]] = current_vec[idx_to_integrate[number_of_integrations - 1]] + delta;
            ans = ans + multiplier*self.get_simpsons(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);        

            if (iter + 2) % 3 == 0
            {
                multiplier = 2.0;
            }
            else
            {
                multiplier = 3.0;
            }
        }

        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][1];

        ans = ans + self.get_simpsons(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);

        return 3.0*delta*ans/8.0;
    }

    ///returns the numerical integration via Trapezoidal method
    /// number_of_integrations: number of times the equation needs to be integrated
    /// idx_to_integrate: the variables' index/indices that needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    fn get_trapezoidal<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[f64; NUM_VARS]) -> f64, integration_limits: &[[f64; 2]; NUM_INTEGRATIONS], point: &[f64; NUM_VARS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut current_vec = *point;
            current_vec[idx_to_integrate[0]] = integration_limits[0][0];

            let mut ans = func(&current_vec);
            let delta = (integration_limits[0][1] - integration_limits[0][0])/(self.total_iterations as f64);

            for _ in 0..self.total_iterations-1
            {
                current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
                ans = ans + 2.0*func(&current_vec);        
            }
            
            current_vec[idx_to_integrate[0]] = integration_limits[0][1];

            ans = ans + func(&current_vec);

            return 0.5*delta*ans;
        }

        let mut current_vec = *point;
        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][0];

        let mut ans = self.get_trapezoidal(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);
        let delta = (integration_limits[number_of_integrations - 1][1] - integration_limits[number_of_integrations - 1][0])/(self.total_iterations as f64);

        for _ in 0..self.total_iterations-1
        {
            current_vec[idx_to_integrate[number_of_integrations - 1]] = current_vec[idx_to_integrate[number_of_integrations - 1]] + delta;
            ans = ans + 2.0*self.get_trapezoidal(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);        
        }
        
        current_vec[idx_to_integrate[number_of_integrations - 1]] = integration_limits[number_of_integrations - 1][1];

        ans = ans + self.get_trapezoidal(number_of_integrations-1, idx_to_integrate, func, integration_limits, &current_vec);

        return 0.5*delta*ans;
    }
}

impl IntegratorMultiVariable for MultiVariableSolver
{
    ///returns the numeric integration value for a multi-variable function
    /// number_of_integrations: number of times the equation needs to be integrated
    /// func: The function to integrate
    /// idx_to_integrate: the variables' index/indices that needs to be integrated
    /// func: The function to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    /// 
    /// NOTE: Returns a Result<f64, &'static str>,
    /// where possible Err are:
    /// INTEGRATION_CANNOT_HAVE_ZERO_ITERATIONS -> if number_of_integrations is zero
    /// INTEGRATION_LIMITS_ILL_DEFINED -> if any integration_limit[i][0] >= integration_limit[i][1] for all possible i
    /// INCORRECT_NUMBER_OF_INTEGRATION_LIMITS -> if size of integration_limit is not equal to number_of_integrations
    /// 
    /// assume we want to integrate 2.0*x + y*z . the function would be:
    /// ```
    /// let func = | args: &[f64; 3] | -> f64 
    ///{ 
    ///    return 2.0*args[0] + args[1]*args[2];
    ///};
    /// let point = [1.0, 2.0, 3.0];
    /// 
    /// use crate::multicalc::numerical_integration::integrator::*;
    /// use multicalc::numerical_integration::iterative_integration;
    /// 
    /// let integrator = iterative_integration::MultiVariableSolver::default();
    /// 
    /// let integration_limit = [[0.0, 1.0]; 1]; //desired integation limit
    /// let val = integrator.get(1, [0; 1], &func, &integration_limit, &point).unwrap();
    /// assert!(f64::abs(val - 7.0) < 1e-6);
    /// ```
    fn get<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[f64; NUM_VARS]) -> f64, integration_limits: &[[f64; 2]; NUM_INTEGRATIONS], point: &[f64; NUM_VARS]) -> Result<f64, &'static str> 
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