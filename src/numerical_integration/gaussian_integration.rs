use crate::numerical_integration::mode::GaussianQuadratureMethod;
use num_complex::ComplexFloat;
use crate::utils::{gh_table, gl_table, gauss_laguerre_table};
use crate::numerical_integration::integrator::*;
use crate::utils::error_codes::*;


pub const DEFAULT_QUADRATURE_ORDERS: usize = 4;

#[derive(Clone, Copy)]
pub struct SingleVariableSolver
{
    order: usize,
    integration_method: GaussianQuadratureMethod
}

impl Default for SingleVariableSolver
{
    fn default() -> Self 
    {
        return SingleVariableSolver { order: DEFAULT_QUADRATURE_ORDERS, integration_method: GaussianQuadratureMethod::GaussLegendre };
    }
}

impl SingleVariableSolver
{
    pub fn get_order(&self) -> usize
    {
        return self.order;
    }

    pub fn set_order(&mut self, order: usize) 
    {
        self.order = order;
    }

    pub fn get_integration_method(&self) -> GaussianQuadratureMethod
    {
        return self.integration_method;
    }

    pub fn set_integration_method(&mut self, integration_method: GaussianQuadratureMethod)
    {
        self.integration_method = integration_method;
    }

    pub fn from_parameters(order: usize, integration_method: GaussianQuadratureMethod) -> Self 
    {
        SingleVariableSolver
        {
            order: order,
            integration_method: integration_method
        }    
    }

    fn check_for_errors<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> Result<(), &'static str> 
    {
        //TODO
        if !(1..=gl_table::MAX_GL_ORDER).contains(&self.order)
        {
            return Err(GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE);
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

    fn get_gauss_legendre<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut ans = T::zero();
            let abcsissa_coeff = (integration_limit[0][1] - integration_limit[0][0])/T::from(2.0).unwrap();
            let intercept = (integration_limit[0][1] + integration_limit[0][0])/T::from(2.0).unwrap();

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

                let args = abcsissa_coeff*T::from(abcsissa).unwrap() + intercept;

                ans = ans + T::from(weight).unwrap()*func(args);
            }

            return abcsissa_coeff*ans;
        }

        let mut ans = T::zero();
        let abcsissa_coeff = (integration_limit[number_of_integrations-1][1] - integration_limit[number_of_integrations-1][0])/T::from(2.0).unwrap();
        //let intercept = (integration_limit[number_of_integrations-1][1] + integration_limit[number_of_integrations-1][0])/T::from(2.0).unwrap();

        for iter in 0..self.order
        {
            let (_, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

            //let args = abcsissa_coeff*T::from(abcsissa).unwrap() + intercept;

            ans = ans + T::from(weight).unwrap()*self.get_gauss_legendre(number_of_integrations-1, func, integration_limit);
        }

        return abcsissa_coeff*ans;
    }

    fn get_gauss_hermite_transformation<T: ComplexFloat>(&self, func: &dyn Fn(T) -> T, args: T) -> T
    {
        //https://math.stackexchange.com/questions/180447/can-i-only-apply-the-gauss-hermite-routine-with-an-infinite-interval-or-can-i-tr
        return func(args)*T::exp(args*args)/(T::cosh(args)*T::cosh(args));
    }

    fn get_gauss_hermite<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut ans = T::zero();

            //limits transformation logic from https://math.stackexchange.com/questions/180447/can-i-only-apply-the-gauss-hermite-routine-with-an-infinite-interval-or-can-i-tr
            let abcsissa_coeff = (integration_limit[0][1] - integration_limit[0][0])/T::from(2.0).unwrap();
            let intercept = (integration_limit[0][1] + integration_limit[0][0])/T::from(2.0).unwrap();

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gh_table::get_gh_weights_and_abscissae(self.order, iter).unwrap();

                let args = abcsissa_coeff*T::tanh(T::from(abcsissa).unwrap()) + intercept;

                ans = ans + T::from(weight).unwrap()*self.get_gauss_hermite_transformation(func, args);
            }

            return abcsissa_coeff*ans;
        }

        let mut ans = T::zero();
        let abcsissa_coeff = (integration_limit[number_of_integrations-1][1] - integration_limit[number_of_integrations-1][0])/T::from(2.0).unwrap();
        //let intercept = (integration_limit[number_of_integrations-1][1] + integration_limit[number_of_integrations-1][0])/T::from(2.0).unwrap();

        for iter in 0..self.order
        {
            let (_, weight) = gh_table::get_gh_weights_and_abscissae(self.order, iter).unwrap();

            //let args = abcsissa_coeff*T::tanh(T::from(abcsissa).unwrap()) + intercept;

            ans = ans + T::from(weight).unwrap()*self.get_gauss_hermite(number_of_integrations-1, func, integration_limit);
        }

        return abcsissa_coeff*ans;
    }

    fn get_gauss_laguerre<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut ans = T::zero();

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gauss_laguerre_table::get_gauss_laguerre_weights_and_abscissae(self.order, iter).unwrap();

                let args = (integration_limit[0][0] - integration_limit[0][1])*T::log(T::from(abcsissa).unwrap() - integration_limit[0][1], T::abs(T::exp(T::one()))) - T::from(abcsissa).unwrap();

                ans = ans + T::from(weight).unwrap()*func(args);
            }

            return ans;
        }

        let mut ans = T::zero();

        for iter in 0..self.order
        {
            let (_, weight) = gauss_laguerre_table::get_gauss_laguerre_weights_and_abscissae(self.order, iter).unwrap();

            //let args = (integration_limit[0][0] - integration_limit[0][1])*T::log(T::from(abcsissa).unwrap() - integration_limit[0][1], T::abs(T::exp(T::one()))) - T::from(abcsissa).unwrap();

            ans = ans + T::from(weight).unwrap()*self.get_gauss_laguerre(number_of_integrations, func, integration_limit);
        }

        return ans;
    }
}

impl IntegratorSingleVariable for SingleVariableSolver
{
    fn get<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, func: &dyn Fn(T) -> T, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> Result<T, &'static str> 
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

#[derive(Clone, Copy)]
pub struct MultiVariableSolver
{
    order: usize,
    integration_method: GaussianQuadratureMethod
}

impl Default for MultiVariableSolver
{
    fn default() -> Self 
    {
        return MultiVariableSolver { order: DEFAULT_QUADRATURE_ORDERS, integration_method: GaussianQuadratureMethod::GaussLegendre };
    }
}

impl MultiVariableSolver
{
    pub fn get_order(&self) -> usize
    {
        return self.order;
    }

    pub fn set_order(&mut self, order: usize) 
    {
        self.order = order;
    }

    pub fn get_integration_method(&self) -> GaussianQuadratureMethod
    {
        return self.integration_method;
    }

    pub fn set_integration_method(&mut self, integration_method: GaussianQuadratureMethod)
    {
        self.integration_method = integration_method;
    }

    pub fn from_parameters(order: usize, integration_method: GaussianQuadratureMethod) -> Self 
    {
        MultiVariableSolver
        {
            order: order,
            integration_method: integration_method
        }    
    }

    fn check_for_errors<T: ComplexFloat, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, integration_limit: &[[T; 2]; NUM_INTEGRATIONS]) -> Result<(), &'static str> 
    {
        //TODO
        if !(1..=gl_table::MAX_GL_ORDER).contains(&self.order)
        {
            return Err(GAUSSIAN_QUADRATURE_ORDER_OUT_OF_RANGE);
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

    fn get_gauss_legendre<T: ComplexFloat, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; NUM_INTEGRATIONS], point: &[T; NUM_VARS]) -> T
    {
        if number_of_integrations == 1
        {
            let mut ans = T::zero();
            let abcsissa_coeff = (integration_limits[0][1] - integration_limits[0][0])/T::from(2.0).unwrap();
            let intercept = (integration_limits[0][1] + integration_limits[0][0])/T::from(2.0).unwrap();

            let mut args = *point;

            for iter in 0..self.order
            {
                let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

                args[idx_to_integrate[0]] = abcsissa_coeff*T::from(abcsissa).unwrap() + intercept;

                ans = ans + T::from(weight).unwrap()*func(&args);
            }

            return abcsissa_coeff*ans;
        }

        let mut ans = T::zero();
        let abcsissa_coeff = (integration_limits[number_of_integrations-1][1] - integration_limits[number_of_integrations-1][0])/T::from(2.0).unwrap();
        let intercept = (integration_limits[number_of_integrations-1][1] + integration_limits[number_of_integrations-1][0])/T::from(2.0).unwrap();

        let mut args = *point;

        for iter in 0..self.order
        {
            let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter).unwrap();

            args[idx_to_integrate[number_of_integrations-1]] = abcsissa_coeff*T::from(abcsissa).unwrap() + intercept;

            ans = ans + T::from(weight).unwrap()*self.get_gauss_legendre(number_of_integrations-1, idx_to_integrate, func, integration_limits, &args);
        }

        return abcsissa_coeff*ans;
    }
}

impl IntegratorMultiVariable for MultiVariableSolver
{
    fn get<T: ComplexFloat, const NUM_VARS: usize, const NUM_INTEGRATIONS: usize>(&self, number_of_integrations: usize, idx_to_integrate: [usize; NUM_INTEGRATIONS], func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; NUM_INTEGRATIONS], point: &[T; NUM_VARS]) -> Result<T, &'static str> 
    {
        self.check_for_errors(number_of_integrations, integration_limits)?;

        match self.integration_method
        {
            GaussianQuadratureMethod::GaussLegendre => return Ok(self.get_gauss_legendre(number_of_integrations, idx_to_integrate, func, integration_limits, point)),
            GaussianQuadratureMethod::GaussHermite => return Ok(T::zero()),
            GaussianQuadratureMethod::GaussLaguerre => return Ok(T::zero())
        }
    }
}


/*
pub struct GaussianQuadrature
{
    order: usize,
    method_type: mode::GaussianQuadratureMethod
}

impl GaussianQuadrature
{
    pub fn get_order(&self) -> usize
    {
        return self.order;
    }

    pub fn set_order(&mut self, order: usize) 
    {
        self.order = order;
    }

    pub fn get_integration_method(&self) -> mode::GaussianQuadratureMethod
    {
        return self.method_type;
    }

    pub fn set_integration_method(&mut self, integration_method: mode::GaussianQuadratureMethod)
    {
        self.method_type = integration_method;
    }

    pub fn with_parameters(order: usize, integration_method: mode::GaussianQuadratureMethod) -> Self 
    {
        GaussianQuadrature
        {
            order: order,
            method_type: integration_method
        }    
    }

    fn check_for_errors<T: ComplexFloat>(&self, integration_limit: &[T; 2]) -> Result<(), &'static str> 
    {
        if !(1..=gl_table::MAX_GL_ORDER).contains(&self.order)
        {
            return Err(&'static str::GaussianQuadratureOrderOutOfRange);
        }

        if integration_limit[0].abs() >= integration_limit[1].abs()
        {
            return Err(&'static str::IntegrationLimitsIllDefined);
        }

        return Ok(());        
    }

    //must know the highest order of the equation
    fn get_gauss_legendre_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
    {
        self.check_for_errors(integration_limit)?;

        let mut ans = T::zero();
        let abcsissa_coeff = (integration_limit[1] - integration_limit[0])/T::from(2.0).unwrap();
        let intercept = (integration_limit[1] + integration_limit[0])/T::from(2.0).unwrap();

        let mut args = *point;

        for iter in 0..self.order
        {
            let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter)?;

            args[idx_to_integrate] = abcsissa_coeff*T::from(abcsissa).unwrap() + intercept;

            ans = ans + T::from(weight).unwrap()*func(&args);
        }

        return Ok(abcsissa_coeff*ans);
    }

    fn get_gauss_legendre_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
    {
        let mut ans = T::zero();
        let abcsissa_coeff = (integration_limits[0][1] - integration_limits[0][0])/T::from(2.0).unwrap();
        let intercept = (integration_limits[0][1] + integration_limits[0][0])/T::from(2.0).unwrap();

        let mut args = *point;

        for iter in 0..self.order
        {
            let (abcsissa, weight) = gl_table::get_gl_weights_and_abscissae(self.order, iter)?;

            args[idx_to_integrate[0]] = abcsissa_coeff*T::from(abcsissa).unwrap() + intercept;

            ans = ans + T::from(weight).unwrap()*self.get_single_partial(func, idx_to_integrate[1], &integration_limits[1], point)?;
        }

        return Ok(abcsissa_coeff*ans);
    }
}

impl Integrator for GaussianQuadrature
{
    fn get_single_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limit: &[T; 2]) -> Result<T, &'static str>
    {
        self.check_for_errors(integration_limit)?;

        let point = [integration_limit[1]; NUM_VARS];

        match self.method_type
        {
            mode::GaussianQuadratureMethod::GaussLegendre  => return self.get_gauss_legendre_1(func, 0, integration_limit, &point)
        }
    }

    fn get_single_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
    {
        self.check_for_errors(integration_limit)?;

        match self.method_type
        {
            mode::GaussianQuadratureMethod::GaussLegendre  => return self.get_gauss_legendre_1(func, idx_to_integrate, integration_limit, &point)
        }
    }

    fn get_double_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; 2]) -> Result<T, &'static str> 
    {
        let point = [integration_limits[0][1]; NUM_VARS];

        match self.method_type
        {
            mode::GaussianQuadratureMethod::GaussLegendre  => return self.get_gauss_legendre_2(func, [0, 0], integration_limits, &point)
        }
    }

    fn get_double_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str> 
    {
        match self.method_type
        {
            mode::GaussianQuadratureMethod::GaussLegendre => return self.get_gauss_legendre_2(func, idx_to_integrate, integration_limits, point)
        }
    }
}
*/