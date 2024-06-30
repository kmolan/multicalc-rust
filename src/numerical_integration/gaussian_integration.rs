use crate::numerical_integration::mode::GaussianQuadratureMethod;
use crate::utils::{gh_table, gl_table, gauss_laguerre_table};
use crate::numerical_integration::integrator::*;
use crate::utils::error_codes::*;


pub const DEFAULT_QUADRATURE_ORDERS: usize = 4;

///Implements the gaussian quadrature methods for numerical integration for single variable functions
#[derive(Clone, Copy)]
pub struct SingleVariableSolver
{
    order: usize,
    integration_method: GaussianQuadratureMethod
}

impl Default for SingleVariableSolver
{
    ///default constructor, optimal for most generic polynomial equations
    fn default() -> Self 
    {
        return SingleVariableSolver { order: DEFAULT_QUADRATURE_ORDERS, integration_method: GaussianQuadratureMethod::GaussLegendre };
    }
}

impl SingleVariableSolver
{
    ///returns the chosen number of nodes/order for quadrature
    pub fn get_order(&self) -> usize
    {
        return self.order;
    }

    ///sets the number of nodes/order for quadrature
    pub fn set_order(&mut self, order: usize) 
    {
        self.order = order;
    }

    ///returns the chosen integration method
    /// possible choices are GaussLegendre, GaussHermite and GaussLaguerre
    pub fn get_integration_method(&self) -> GaussianQuadratureMethod
    {
        return self.integration_method;
    }

    /// sets the integration method
    /// possible choices are GaussLegendre, GaussHermite and GaussLaguerre
    pub fn set_integration_method(&mut self, integration_method: GaussianQuadratureMethod)
    {
        self.integration_method = integration_method;
    }

    ///custom constructor, optimal for fine-tuning for specific cases
    pub fn from_parameters(order: usize, integration_method: GaussianQuadratureMethod) -> Self 
    {
        SingleVariableSolver
        {
            order: order,
            integration_method: integration_method
        }    
    }

    ///Helper method to check if inputs are well defined
    fn check_for_errors<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> Result<(), &'static str> 
    {
        //TODO
        if !(1..=gl_table::MAX_GL_ORDER).contains(&self.order)
        {
            return Err(GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE);
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

    ///returns the gauss legendre numerical integral for a given equation
    /// number_of_integrations: number of integrations to perform on the equation
    /// func: the equation to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    fn get_gauss_legendre<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut ans = 0.0;
            let abcsissa_coeff = (integration_limit[0][1] - integration_limit[0][0])/2.0;
            let intercept = (integration_limit[0][1] + integration_limit[0][0])/2.0;

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

                let args = abcsissa_coeff*abcsissa + intercept;

                ans = ans + weight*func(args);
            }

            return abcsissa_coeff*ans;
        }

        let mut ans = 0.0;
        let abcsissa_coeff = (integration_limit[number_of_integrations-1][1] - integration_limit[number_of_integrations-1][0])/2.0;
        //let intercept = (integration_limit[number_of_integrations-1][1] + integration_limit[number_of_integrations-1][0])/2.0;

        for iter in 0..self.order
        {
            let (_, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

            //let args = abcsissa_coeff*abcsissa + intercept;

            ans = ans + weight*self.get_gauss_legendre(number_of_integrations-1, func, integration_limit);
        }

        return abcsissa_coeff*ans;
    }

    ///returns the gauss hermite numerical integral for a given equation
    /// number_of_integrations: number of integrations to perform on the equation
    /// func: the equation to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    fn get_gauss_hermite<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut ans = 0.0;

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gh_table::get_gh_weights_and_abscissae(self.order, iter).unwrap();

                ans = ans + weight*func(abcsissa)*f64::exp(abcsissa*abcsissa);
            }

            return ans;
        }

        let mut ans = 0.0;

        for iter in 0..self.order
        {
            let (_, weight) = gh_table::get_gh_weights_and_abscissae(self.order, iter).unwrap();

            ans = ans + weight*self.get_gauss_hermite(number_of_integrations-1, func, integration_limit);
        }

        return ans;
    }

    ///returns the gauss laguerre numerical integral for a given equation
    /// number_of_integrations: number of integrations to perform on the equation
    /// func: the equation to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    fn get_gauss_laguerre<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut ans = 0.0;

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gauss_laguerre_table::get_gauss_laguerre_weights_and_abscissae(self.order, iter).unwrap();
                ans = ans + (weight*func(abcsissa)*f64::exp(abcsissa));
            }

            return ans;
        }

        let mut ans = 0.0;

        for iter in 0..self.order
        {
            let (_, weight) = gauss_laguerre_table::get_gauss_laguerre_weights_and_abscissae(self.order, iter).unwrap();

            //let args = (integration_limit[0][0] - integration_limit[0][1])*T::log(abcsissa - integration_limit[0][1], T::abs(T::exp(T::one()))) - abcsissa;

            ans = ans + weight*self.get_gauss_laguerre(number_of_integrations, func, integration_limit);
        }

        return ans;
    }
}

impl IntegratorSingleVariable for SingleVariableSolver
{
    ///returns the gaussian quadrature numerical integration for a single variable equation
    /// number_of_integrations: number of integrations to perform on the equation
    /// func: the equation to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// 
    /// NOTE: Returns a Result<f64, &'static str>, where possible Err are:
    /// GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE -> if the chosen numer of nodes/order is out of supported range
    /// INTEGRATION_LIMITS_ILL_DEFINED -> if any integration_limit[i][0] >= integration_limit[i][1] for all possible i
    /// INCORRECT_NUMBER_OF_INTEGRATION_LIMITS -> if number_of_integrations is not equal to the size of integration_limit
    /// 
    /// assume we want to differentiate f(x) = 4.0*x*x*x - 3.0*x*x. the function would be:
    /// ```
    ///    let my_func = | arg: f64 | -> f64 
    ///    { 
    ///        return 4.0*arg*arg*arg - 3.0*arg*arg;
    ///    };
    /// 
    /// use multicalc::numerical_integration::integrator::*;
    /// use multicalc::numerical_integration::gaussian_integration;
    /// 
    /// let integrator = gaussian_integration::SingleVariableSolver::default();
    /// let integration_limit = [[0.0, 2.0]; 1];
    /// let val = integrator.get(1, &my_func, &integration_limit).unwrap(); //single integration
    /// assert!(f64::abs(val - 8.0) < 1e-7);
    /// 
    /// let integration_limit = [[0.0, 2.0], [-1.0, 1.0]];
    /// let val = integrator.get(2, &my_func, &integration_limit).unwrap(); //double integration
    /// assert!(f64::abs(val - 16.0) < 1e-7);
    ///```
    fn get<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(f64) -> f64, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> Result<f64, &'static str> 
    {
        self.check_for_errors(number_of_integrations, integration_limit)?;

        match self.integration_method
        {
            GaussianQuadratureMethod::GaussLegendre => return Ok(self.get_gauss_legendre(number_of_integrations, func, integration_limit)),
            GaussianQuadratureMethod::GaussHermite => return Ok(self.get_gauss_hermite(number_of_integrations, func, integration_limit)),
            GaussianQuadratureMethod::GaussLaguerre => return Ok(self.get_gauss_laguerre(number_of_integrations, func, integration_limit))
        }
    }
}


///Implements the gaussian quadrature methods for numerical integration for multi variable functions
#[derive(Clone, Copy)]
pub struct MultiVariableSolver
{
    order: usize,
    integration_method: GaussianQuadratureMethod
}

impl Default for MultiVariableSolver
{
    ///default constructor, optimal for most generic polynomial equations
    fn default() -> Self 
    {
        return MultiVariableSolver { order: DEFAULT_QUADRATURE_ORDERS, integration_method: GaussianQuadratureMethod::GaussLegendre };
    }
}

impl MultiVariableSolver
{
    ///returns the chosen number of nodes/order for quadrature
    pub fn get_order(&self) -> usize
    {
        return self.order;
    }

    ///sets the number of nodes/order for quadrature
    pub fn set_order(&mut self, order: usize) 
    {
        self.order = order;
    }

    ///returns the chosen integration method
    /// possible choices are GaussLegendre, GaussHermite and GaussLaguerre
    pub fn get_integration_method(&self) -> GaussianQuadratureMethod
    {
        return self.integration_method;
    }

    ///sets the integration method
    /// possible choices are GaussLegendre, GaussHermite and GaussLaguerre
    pub fn set_integration_method(&mut self, integration_method: GaussianQuadratureMethod)
    {
        self.integration_method = integration_method;
    }

    ///custom constructor, optimal for fine-tuning for specific cases
    pub fn from_parameters(order: usize, integration_method: GaussianQuadratureMethod) -> Self 
    {
        MultiVariableSolver
        {
            order,
            integration_method
        }    
    }

    ///Helper method to check if inputs are well defined
    fn check_for_errors<const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, integration_limit: &[[f64; 2]; NUM_INTEGRATIONS]) -> Result<(), &'static str> 
    {
        //TODO
        if !(1..=gl_table::MAX_GL_ORDER).contains(&self.order)
        {
            return Err(GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE);
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

    /// returns the gauss legendre numerical integral for a given equation
    /// number_of_integrations: number of integrations to perform on the equation
    /// idx_to_integrate: the index/indices of variable to integrate
    /// func: the equation to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    fn get_gauss_legendre<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[f64; NUM_VARS]) -> f64, integration_limits: &[[f64; 2]; NUM_INTEGRATIONS], point: &[f64; NUM_VARS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut ans = 0.0;
            let abcsissa_coeff = (integration_limits[0][1] - integration_limits[0][0])/2.0;
            let intercept = (integration_limits[0][1] + integration_limits[0][0])/2.0;

            let mut args = *point;

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

                args[idx_to_integrate[0]] = abcsissa_coeff*abcsissa + intercept;

                ans = ans + weight*func(&args);
            }

            return abcsissa_coeff*ans;
        }

        let mut ans = 0.0;
        let abcsissa_coeff = (integration_limits[number_of_integrations-1][1] - integration_limits[number_of_integrations-1][0])/2.0;
        let intercept = (integration_limits[number_of_integrations-1][1] + integration_limits[number_of_integrations-1][0])/2.0;

        let mut args = *point;

        for iter in 0..self.order
        {
            let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

            args[idx_to_integrate[number_of_integrations-1]] = abcsissa_coeff*abcsissa + intercept;

            ans = ans + weight*self.get_gauss_legendre(number_of_integrations-1, idx_to_integrate, func, integration_limits, &args);
        }

        return abcsissa_coeff*ans;
    }

    /// returns the gauss hermite numerical integral for a given equation
    /// number_of_integrations: number of integrations to perform on the equation
    /// idx_to_integrate: the index/indices of variable to integrate
    /// func: the equation to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    fn get_gauss_hermite<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[f64; NUM_VARS]) -> f64, integration_limits: &[[f64; 2]; NUM_INTEGRATIONS], point: &[f64; NUM_VARS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut ans = 0.0;

            let mut args = *point;

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

                args[idx_to_integrate[0]] = abcsissa;

                ans = ans + weight*func(&args);
            }

            return ans;
        }

        let mut ans = 0.0;

        let mut args = *point;

        for iter in 0..self.order
        {
            let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

            args[idx_to_integrate[number_of_integrations-1]] = abcsissa;

            ans = ans + weight*self.get_gauss_legendre(number_of_integrations-1, idx_to_integrate, func, integration_limits, &args);
        }

        return ans;
    }

    /// returns the gauss laguerre numerical integral for a given equation
    /// number_of_integrations: number of integrations to perform on the equation
    /// idx_to_integrate: the index/indices of variable to integrate
    /// func: the equation to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// point: for variables not being integrated, it is their constant value, otherwise it is their final upper limit of integration
    fn get_gauss_laguerre<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[f64; NUM_VARS]) -> f64, integration_limits: &[[f64; 2]; NUM_INTEGRATIONS], point: &[f64; NUM_VARS]) -> f64
    {
        if number_of_integrations == 1
        {
            let mut ans = 0.0;

            let mut args = *point;

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

                args[idx_to_integrate[0]] = abcsissa;

                ans = ans + weight*func(&args);
            }

            return ans;
        }

        let mut ans = 0.0;

        let mut args = *point;

        for iter in 0..self.order
        {
            let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

            args[idx_to_integrate[number_of_integrations-1]] = abcsissa;

            ans = ans + weight*self.get_gauss_legendre(number_of_integrations-1, idx_to_integrate, func, integration_limits, &args);
        }

        return ans;
    }
}

impl IntegratorMultiVariable for MultiVariableSolver
{
    ///returns the gaussian quadrature numerical integration for a single variable equation
    /// number_of_integrations: number of integrations to perform on the equation
    /// func: the equation to integrate
    /// integration_limit: the integration bound(s) for each round of integration
    /// 
    /// NOTE: Returns a Result<f64, &'static str>, where possible Err are:
    /// GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE -> if the chosen numer of nodes/order is out of supported range
    /// INTEGRATION_LIMITS_ILL_DEFINED -> if any integration_limit[i][0] >= integration_limit[i][1] for all possible i
    /// INCORRECT_NUMBER_OF_INTEGRATION_LIMITS -> if number_of_integrations is not equal to the size of integration_limit
    /// 
    /// assume we want to differentiate f(x,y,z) = 2.0*x + y*z. the function would be:
    /// ```
    ///    let my_func = | args: &[f64; 3] | -> f64 
    ///    { 
    ///        return 2.0*args[0] + args[1]*args[2];
    ///    };
    /// 
    /// use multicalc::numerical_integration::integrator::*;
    /// use multicalc::numerical_integration::gaussian_integration;
    /// 
    /// let integrator = gaussian_integration::MultiVariableSolver::default();
    /// let point = [1.0, 2.0, 3.0];
    /// 
    /// let integration_limit = [[0.0, 1.0]; 1];
    /// let val = integrator.get(1, [0; 1], &my_func, &integration_limit, &point).unwrap(); //single integration for x
    /// assert!(f64::abs(val - 7.0) < 1e-7);
    /// 
    ///```
    fn get<const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[f64; NUM_VARS]) -> f64, integration_limits: &[[f64; 2]; NUM_INTEGRATIONS], point: &[f64; NUM_VARS]) -> Result<f64, &'static str> 
    {
        self.check_for_errors(number_of_integrations, integration_limits)?;

        match self.integration_method
        {
            GaussianQuadratureMethod::GaussLegendre => return Ok(self.get_gauss_legendre(number_of_integrations, idx_to_integrate, func, integration_limits, point)),
            GaussianQuadratureMethod::GaussHermite => return Ok(self.get_gauss_hermite(number_of_integrations, idx_to_integrate, func, integration_limits, point)),
            GaussianQuadratureMethod::GaussLaguerre => return Ok(self.get_gauss_laguerre(number_of_integrations, idx_to_integrate, func, integration_limits, point))
        }
    }
}