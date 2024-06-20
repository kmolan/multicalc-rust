use crate::numerical_integration::mode;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;
use crate::utils::gl_table as gl_table;
use crate::numerical_integration::integrator::Integrator;

pub struct GaussianQuadrature
{
    order: usize,
    method_type: mode::GaussianMethod
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

    pub fn get_integration_method(&self) -> mode::GaussianMethod
    {
        return self.method_type;
    }

    pub fn set_integration_method(&mut self, integration_method: mode::GaussianMethod)
    {
        self.method_type = integration_method;
    }

    pub fn with_parameters(order: usize, integration_method: mode::GaussianMethod) -> Self 
    {
        GaussianQuadrature
        {
            order: order,
            method_type: integration_method
        }    
    }

    fn check_for_errors<T: ComplexFloat>(&self, integration_limit: &[T; 2]) -> Result<(), ErrorCode> 
    {
        if !(1..=gl_table::MAX_GL_ORDER).contains(&self.order)
        {
            return Err(ErrorCode::GaussianQuadratureOrderOutOfRange);
        }

        if integration_limit[0].abs() >= integration_limit[1].abs()
        {
            return Err(ErrorCode::IntegrationLimitsIllDefined);
        }

        return Ok(());        
    }

    //must know the highest order of the equation
    fn get_gauss_legendre_1<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
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

    fn get_gauss_legendre_2<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
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
    fn get_single_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limit: &[T; 2]) -> Result<T, ErrorCode>
    {
        self.check_for_errors(integration_limit)?;

        let point = [integration_limit[1]; NUM_VARS];

        match self.method_type
        {
            mode::GaussianMethod::GaussLegendre  => return self.get_gauss_legendre_1(func, 0, integration_limit, &point)
        }
    }

    fn get_single_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode>
    {
        self.check_for_errors(integration_limit)?;

        match self.method_type
        {
            mode::GaussianMethod::GaussLegendre  => return self.get_gauss_legendre_1(func, idx_to_integrate, integration_limit, &point)
        }
    }

    fn get_double_total<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; 2]) -> Result<T, ErrorCode> 
    {
        let point = [integration_limits[0][1]; NUM_VARS];

        match self.method_type
        {
            mode::GaussianMethod::GaussLegendre  => return self.get_gauss_legendre_2(func, [0, 0], integration_limits, &point)
        }
    }

    fn get_double_partial<T: ComplexFloat, const NUM_VARS: usize>(&self, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS]) -> Result<T, ErrorCode> 
    {
        match self.method_type
        {
            mode::GaussianMethod::GaussLegendre => return self.get_gauss_legendre_2(func, idx_to_integrate, integration_limits, point)
        }
    }
}