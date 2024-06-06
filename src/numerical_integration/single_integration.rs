use crate::numerical_integration::mode::IntegrationMethod;
use crate::utils::gl_table as gl_table;


/// Returns the total single integration value for a given function
/// Only ideal for single variable functions
/// 
/// assume we want to integrate 2*x . the function would be:
/// ```
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return 2.0*args[0];
///    };
///
////where args[0] = x. We also need to define the intervals around which we want to integrate. Assuming our interval is [0.0, 2.0]
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
/// Note: The argument 'n' denotes the number of steps to be used. However, for [`IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// 
pub fn get_total(integration_method: IntegrationMethod, func: &dyn Fn(&Vec<f64>) -> f64, integration_limit: &[f64; 2], n: u64) -> f64
{
    let point = vec![integration_limit[1]];

    match integration_method
    {
        IntegrationMethod::Booles        => return get_booles(func, 0, integration_limit, &point, n),
        IntegrationMethod::GaussLegendre => return get_gauss_legendre(func, 0, integration_limit, &point, n as usize),
        IntegrationMethod::Simpsons      => return get_simpsons(func, 0, integration_limit, &point, n),
        IntegrationMethod::Trapezoidal   => return get_trapezoidal(func, 0, integration_limit, &point, n)
    }
}


/// Returns the partial single integration value for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// assume we want to partially integrate for y for the equation 2.0*x + y*z. The function would be:
/// ```
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return 2.0*args[0] + args[1]*args[2];
///    };
///
////where args[0] = x, args[1] = y and args[2] = z. We also need to define the intervals around which we want to integrate. Assuming our interval is [0.0, 2.0]
///
/// let integration_limit = [0.0, 2.0];
/// 
//// For partial integration to work, we also need to define the static values for the remaining variables. 
//// Assuming x = 1.0, z = 3.0 and we want to integrate over y:
/// 
/// let point = vec![1.0, 2.0, 3.0];
///
//// Note above that the point vector has the same number of elements as the number of elements my_func expects. 
//// The element to integrate, y, has index = 1. We MUST therefore make the point vector's 1st element the same as the integration intervals's upper limit which is 2.0
/// 
//// if we then want to integrate this function over y with 100 steps over the limit [0.0, 2.0], we would use:
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
/// Note: The argument 'n' denotes the number of steps to be used. However, for [`IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// 
pub fn get_partial(integration_method: IntegrationMethod, func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: usize, integration_limit: &[f64; 2], point: &Vec<f64>, n: u64) -> f64
{
    match integration_method
    {
        IntegrationMethod::Booles        => return get_booles(func, idx_to_integrate, integration_limit, point, n),
        IntegrationMethod::GaussLegendre => return get_gauss_legendre(func, idx_to_integrate, integration_limit, point, n as usize),
        IntegrationMethod::Simpsons      => return get_simpsons(func, idx_to_integrate, integration_limit, point, n),
        IntegrationMethod::Trapezoidal   => return get_trapezoidal(func, idx_to_integrate, integration_limit, point, n)
    }
}

fn get_booles(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: usize, integration_limit: &[f64; 2], point: &Vec<f64>, steps: u64) -> f64
{
    check_for_errors(integration_limit, steps);

    let mut current_vec = point.clone();
    current_vec[idx_to_integrate] = integration_limit[0];

    let mut ans = 7.0*func(&current_vec);
    let delta = (integration_limit[1] - integration_limit[0])/(steps as f64);

    let mut multiplier: f64 = 32.0;

    for iter in 0..steps-1
    {
        current_vec[idx_to_integrate] += delta;
        ans += multiplier*func(&current_vec);
        
        if (iter + 2) % 2 != 0
        {
            multiplier = 32.0;
        }
        else
        {
            if (iter + 2) % 4 == 0
            {
                multiplier = 14.0;
            }
            else
            {
                multiplier = 12.0;
            }
        }
    }

    current_vec[idx_to_integrate] = integration_limit[1];

    ans += 7.0*func(&current_vec);

    return 2.0*delta*ans/45.0;
}


//must know the highest order of the equation
fn get_gauss_legendre(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: usize, integration_limit: &[f64; 2], point: &Vec<f64>, order: usize) -> f64
{
    assert!(order < gl_table::MAX_GL_ORDER, "Legendre quadrature is limited up to n = {}", gl_table::MAX_GL_ORDER);
    
    check_for_errors(integration_limit, 100); //no 'steps' argument for this method

    let mut ans = 0.0;
    let abcsissa_coeff = (integration_limit[1] - integration_limit[0])/2.0;
    let intercept = (integration_limit[1] + integration_limit[0])/2.0;

    let (weight, abcsissa) = gl_table::get_gl_weights_and_abscissae(order);

    let mut args = point.clone();

    for iter in 0..order
    {
        args[idx_to_integrate] = abcsissa_coeff*abcsissa[iter] + intercept;

        ans += weight[iter]*func(&args);
    }

    return abcsissa_coeff*ans;
}

//TODO: add the 1/3 rule also
//the 3/8 rule, better than the 1/3 rule
fn get_simpsons(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: usize, integration_limit: &[f64; 2], point: &Vec<f64>, steps: u64) -> f64
{
    check_for_errors(integration_limit, steps);

    let mut current_vec = point.clone();
    current_vec[idx_to_integrate] = integration_limit[0];

    let mut ans = func(&current_vec);
    let delta = (integration_limit[1] - integration_limit[0])/(steps as f64);

    let mut multiplier = 3.0;

    for iter in 0..steps-1
    {
        current_vec[idx_to_integrate] += delta;
        ans += multiplier*func(&current_vec);        

        if (iter + 2) % 3 == 0
        {
            multiplier = 2.0;
        }
        else
        {
            multiplier = 3.0;
        }
    }

    current_vec[idx_to_integrate] = integration_limit[1];

    ans += func(&current_vec);

    return 3.0*delta*ans/8.0;
}

fn get_trapezoidal(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: usize, integration_limit: &[f64; 2], point: &Vec<f64>, steps: u64) -> f64
{
    check_for_errors(integration_limit, steps);

    let mut current_vec = point.clone();
    current_vec[idx_to_integrate] = integration_limit[0];

    let mut ans = func(&current_vec);
    let delta = (integration_limit[1] - integration_limit[0])/(steps as f64);

    for _ in 0..steps-1
    {
        current_vec[idx_to_integrate] += delta;
        ans += 2.0*func(&current_vec);        
    }
    
    current_vec[idx_to_integrate] = integration_limit[1];

    ans += func(&current_vec);

    return 0.5*delta*ans;
}

fn check_for_errors(integration_limit: &[f64; 2], steps: u64)
{
    assert!(integration_limit[0] < integration_limit[1], "lower end of integration interval value must be lower than the higher value");
    assert!(steps != 0, "number of steps cannot be zero");
}