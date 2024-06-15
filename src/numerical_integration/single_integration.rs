use crate::numerical_integration::mode::IntegrationMethod;
use num_complex::ComplexFloat;

#[cfg(feature = "std")]
use crate::utils::gl_table as gl_table;

/// Returns the total single integration value for a given function
/// Only ideal for single variable functions
/// 
/// assume we want to integrate 2*x . the function would be:
/// ```
///    let my_func = | args: &[f64; 1] | -> f64 
///    { 
///        return 2.0*args[0];
///    };
///
/////where args[0] = x. We also need to define the integration limit around which we want to integrate. Assuming our limit is [0.0, 2.0]
///
/// let integration_limit = [0.0, 2.0];
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::single_integration;
///
/// let val = single_integration::get_total(IntegrationMethod::Trapezoidal,  //<- The method for integration we want to use
///                                          &my_func,                       //<- our closure                 
///                                          &integration_limit,             //<- The integration limit needed                          
///                                          10);                            //<- number of steps
/// 
/// assert!(f64::abs(val - 4.0) < 0.00001);
///```
/// 
/// the above method example can also be extended to complex numbers:
/// ```
///    let my_func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
///    { 
///        return 2.0*args[0];
///    };
///
/////where args[0] = x. Assuming our integration limit is (0.0 + 0.0i) to (2.0 + 2.0i)
///
/// let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)];
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::single_integration;
///
/// let val = single_integration::get_total(IntegrationMethod::Trapezoidal,  //<- The method for integration we want to use
///                                          &my_func,                       //<- our closure                 
///                                          &integration_limit,             //<- The integration limit needed                          
///                                          10);                            //<- number of steps
/// 
/// assert!(num_complex::ComplexFloat::abs(val.re - 0.0) < 0.00001);
/// assert!(num_complex::ComplexFloat::abs(val.im - 8.0) < 0.00001);
///```
/// 
/// Note: The argument 'n' denotes the number of steps to be used. However, for [`IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// 
pub fn get_total<T: ComplexFloat, const NUM_VARS: usize>(integration_method: IntegrationMethod, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limit: &[T; 2], n: u64) -> T
{
    let point = [integration_limit[1]; NUM_VARS];

    match integration_method
    {
        IntegrationMethod::Booles        => return get_booles(func, 0, integration_limit, &point, n),
        IntegrationMethod::GaussLegendre => 
        {
            #[cfg(feature = "std")]
            return get_gauss_legendre(func, 0, integration_limit, &point, n as usize);

            #[cfg(not(feature = "std"))]
            panic!("enable std context to use!");
        }
        IntegrationMethod::Simpsons      => return get_simpsons(func, 0, integration_limit, &point, n),
        IntegrationMethod::Trapezoidal   => return get_trapezoidal(func, 0, integration_limit, &point, n)
    }
}


/// Returns the partial single integration value for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// assume we want to partially integrate for y for the equation 2.0*x + y*z. The function would be:
/// ```
///    let my_func = | args: &[f64; 3] | -> f64 
///    { 
///        return 2.0*args[0] + args[1]*args[2];
///    };
///
/////where args[0] = x, args[1] = y and args[2] = z. We also need to define the integration limit around which we want to integrate. Assuming our limit is [0.0, 2.0]
///
/// let integration_limit = [0.0, 2.0];
/// 
///// For partial integration to work, we also need to define the static values for the remaining variables. 
///// Assuming x = 1.0, z = 3.0 and we want to integrate over y:
/// 
/// let point = [1.0, 2.0, 3.0];
///
///// Note above that the point vector has the same number of elements as the number of elements my_func expects. 
///// The element to integrate, y, has index = 1. We MUST therefore make the point vector's 1st element the same as the integration intervals's upper limit which is 2.0
/// 
///// if we then want to integrate this function over y with 100 steps, we would use:
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::single_integration;
///
/// let val = single_integration::get_partial(IntegrationMethod::Trapezoidal,  //<- The method for integration we want to use
///                                           &my_func,                        //<- our closure   
///                                           1,                               //<- index of variable we want to integrate, in this case "y", which is 1 
///                                           &integration_limit,              //<- The integration limit needed 
///                                           &point,                          //<- The final point with all x,y,z values
///                                           10);                             //<- number of steps
/// 
/// assert!(f64::abs(val - 10.0) < 0.00001);
///```
/// 
/// the above method example can also be extended to complex numbers:
/// ```
///    let my_func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
///    { 
///        return 2.0*args[0] + args[1]*args[2];
///    };
///
/////where args[0] = x, args[1] = y and args[2] = z. Assuming our integration limit is (0.0 + 0.0i) to (2.0 + 0.0i)
///
/// let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 0.0)];
/// 
///// For partial integration to work, we also need to define the static values for the remaining variables. 
///// Assuming x = 1.0 + 1.0i, z = 3.0 + 0.5i and we want to integrate over y:
/// 
/// let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];
///
///// Note above that the point vector has the same number of elements as the number of elements my_func expects. 
///// The element to integrate, y, has index = 1. We MUST therefore make the point vector's 1st element the same as the integration intervals's upper limit which is 2.0 + 0.0i
/// 
///// if we then want to integrate this function over y with 100 steps], we would use:
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::single_integration;
///
/// let val = single_integration::get_partial(IntegrationMethod::Trapezoidal,  //<- The method for integration we want to use
///                                           &my_func,                        //<- our closure   
///                                           1,                               //<- index of variable we want to integrate, in this case "y", which is 1 
///                                           &integration_limit,              //<- The integration limit needed 
///                                           &point,                          //<- The final point with all x,y,z values
///                                           10);                             //<- number of steps
/// 
/// assert!(num_complex::ComplexFloat::abs(val.re - 10.0) < 0.00001);
/// assert!(num_complex::ComplexFloat::abs(val.im - 5.0) < 0.00001);
///```
/// Note: The argument 'n' denotes the number of steps to be used. However, for [`IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// 
pub fn get_partial<T: ComplexFloat, const NUM_VARS: usize>(integration_method: IntegrationMethod, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS], n: u64) -> T
{
    match integration_method
    {
        IntegrationMethod::Booles        => return get_booles(func, idx_to_integrate, integration_limit, point, n),
        IntegrationMethod::GaussLegendre => 
        {
            #[cfg(feature = "std")]
            return get_gauss_legendre(func, idx_to_integrate, integration_limit, point, n as usize);

            #[cfg(not(feature = "std"))]
            panic!("enable std context to use!");            
        }
        IntegrationMethod::Simpsons      => return get_simpsons(func, idx_to_integrate, integration_limit, point, n),
        IntegrationMethod::Trapezoidal   => return get_trapezoidal(func, idx_to_integrate, integration_limit, point, n)
    }
}

fn get_booles<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS], steps: u64) -> T
{
    check_for_errors(integration_limit, steps);

    let mut current_vec = *point;
    current_vec[idx_to_integrate] = integration_limit[0];

    let mut ans = T::from(7.0).unwrap()*func(&current_vec);
    let delta = (integration_limit[1] - integration_limit[0])/(T::from(steps).unwrap());

    let mut multiplier = T::from(32.0).unwrap();

    for iter in 0..steps-1
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

    return T::from(2.0).unwrap()*delta*ans/T::from(45.0).unwrap();
}


//must know the highest order of the equation
#[cfg(feature = "std")]
fn get_gauss_legendre<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS], order: usize) -> T
{
    assert!(order < gl_table::MAX_GL_ORDER, "Legendre quadrature is limited up to n = {}", gl_table::MAX_GL_ORDER);
    
    check_for_errors(integration_limit, 100); //no 'steps' argument for this method

    let mut ans = T::zero();
    let abcsissa_coeff = (integration_limit[1] - integration_limit[0])/T::from(2.0).unwrap();
    let intercept = (integration_limit[1] + integration_limit[0])/T::from(2.0).unwrap();

    let (weight, abcsissa) = gl_table::get_gl_weights_and_abscissae(order);

    let mut args = *point;

    for iter in 0..order
    {
        args[idx_to_integrate] = abcsissa_coeff*T::from(abcsissa[iter]).unwrap() + intercept;

        ans = ans + T::from(weight[iter]).unwrap()*func(&args);
    }

    return abcsissa_coeff*ans;
}

//TODO: add the 1/3 rule also
//the 3/8 rule, better than the 1/3 rule
fn get_simpsons<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS], steps: u64) -> T
{
    check_for_errors(integration_limit, steps);

    let mut current_vec = *point;
    current_vec[idx_to_integrate] = integration_limit[0];

    let mut ans = func(&current_vec);
    let delta = (integration_limit[1] - integration_limit[0])/(T::from(steps).unwrap());

    let mut multiplier = T::from(3.0).unwrap();

    for iter in 0..steps-1
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

    return T::from(3.0).unwrap()*delta*ans/T::from(8.0).unwrap();
}

fn get_trapezoidal<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS], steps: u64) -> T
{
    check_for_errors(integration_limit, steps);

    let mut current_vec = *point;
    current_vec[idx_to_integrate] = integration_limit[0];

    let mut ans = func(&current_vec);
    let delta = (integration_limit[1] - integration_limit[0])/(T::from(steps).unwrap());

    for _ in 0..steps-1
    {
        current_vec[idx_to_integrate] = current_vec[idx_to_integrate] + delta;
        ans = ans + T::from(2.0).unwrap()*func(&current_vec);        
    }
    
    current_vec[idx_to_integrate] = integration_limit[1];

    ans = ans + func(&current_vec);

    return T::from(0.5).unwrap()*delta*ans;
}

fn check_for_errors<T: ComplexFloat>(integration_limit: &[T; 2], steps: u64)
{
    assert!(integration_limit[0].abs() < integration_limit[1].abs(), "lower end of integration interval value must be lower than the higher value");
    assert!(steps != 0, "number of steps cannot be zero");
}