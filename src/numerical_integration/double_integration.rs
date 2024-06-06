use crate::numerical_integration::mode::IntegrationMethod;
use crate::numerical_integration::single_integration;
use crate::utils::gl_table as gl_table;

/// Returns the total double integration value for a given function
/// Only ideal for single variable functions
/// 
/// assume we want to integrate 6*x. The function would be:
/// ```
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return 6.0*args[0];
///    };
///
//// where args[0] = x. We also need to define the intervals around which we want to integrate.
//// Assuming we first we want to integrate over [0.0, 1.0] and then over [0.0, 3.0]
///
/// let integration_intervals = [[0.0, 1.0], [0.0, 3.0]];
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::double_integration;
///
/// let val = double_integration::get_total(IntegrationMethod::Trapezoidal,   //<- The method for integration we want to use
///                                          &my_func,                        //<- our closure                 
///                                          &integration_intervals,          //<- The integration interval needed                          
///                                          10);                             //<- number of steps
/// 
/// assert!(f64::abs(val - 27.0) < 0.00001);
///```
/// 
/// Note: The argument 'n' denotes the number of steps to be used. However, for [`IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// 
pub fn get_total(integration_method: IntegrationMethod, func: &dyn Fn(&Vec<f64>) -> f64, integration_intervals: &[[f64; 2]; 2], n: u64) -> f64 
{
    let point = vec![integration_intervals[0][1], integration_intervals[1][1]];

    match integration_method
    {
        IntegrationMethod::Booles        => return get_booles(func, [0, 0], integration_intervals, &point, n),
        IntegrationMethod::GaussLegendre => return get_gauss_legendre(func, [0, 0], integration_intervals, &point, n as usize),
        IntegrationMethod::Simpsons      => return get_simpsons(func, [0, 0], integration_intervals, &point, n),
        IntegrationMethod::Trapezoidal   => return get_trapezoidal(func, [0, 0], integration_intervals, &point, n)
    }
}


/// Returns the partial double integration value for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// assume we want to partially integrate first for x then y for the equation 2.0*x + y*z. The function would be:
/// ```
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return 2.0*args[0] + args[1]*args[2];
///    };
///
//// where args[0] = x, args[1] = y and args[2] = z. We also need to define the intervals around which we want to integrate.
//// Assuming we first we want to integrate over [0.0, 1.0] and then over [0.0, 1.0]
///
/// let integration_intervals = [[0.0, 1.0], [0.0, 1.0]];
/// 
//// For partial integration to work, we also need to define the static values for the remaining variables. 
//// Assuming z = 3.0:
/// 
/// let point = vec![1.0, 2.0, 3.0];
/// 
//// Note above that the point vector has the same number of elements as the number of elements my_func expects. 
//// Note above that in the point vector, the indexes of the variables to integrate, 0 and 1, 
//// MUST have same value as those variables' integration interval's upper limit, which is 1.0 and 2.0 respectively 
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::double_integration;
///
/// let val = double_integration::get_partial(IntegrationMethod::Trapezoidal,  //<- The method for integration we want to use
///                                           &my_func,                        //<- our closure   
///                                           [0, 1],                          //<- index of variables we want to differentiate, in this case x and y              
///                                           &integration_intervals,          //<- The integration interval needed
///                                           &point,                          //<- The final point with all x,y,z values                          
///                                           10);                             //<- number of steps
/// 
/// assert!(f64::abs(val - 2.50) < 0.00001);
///```
/// 
/// Note: The argument 'n' denotes the number of steps to be used. However, for [`IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// 
pub fn get_partial(integration_method: IntegrationMethod, func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: [usize; 2], integration_intervals: &[[f64; 2]; 2], point: &Vec<f64>, n: u64) -> f64 
{
    match integration_method
    {
        IntegrationMethod::Booles        => return get_booles(func, idx_to_integrate, integration_intervals, point, n),
        IntegrationMethod::GaussLegendre => return get_gauss_legendre(func, idx_to_integrate, integration_intervals, point, n as usize),
        IntegrationMethod::Simpsons      => return get_simpsons(func, idx_to_integrate, integration_intervals, point, n),
        IntegrationMethod::Trapezoidal   => return get_trapezoidal(func, idx_to_integrate, integration_intervals, point, n)
    }
}

fn get_booles(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: [usize; 2], integration_intervals: &[[f64; 2]; 2], point: &Vec<f64>, steps: u64) -> f64
{
    let mut current_vec = point.clone();
    current_vec[idx_to_integrate[0]] = integration_intervals[0][0];

    let mut ans = 7.0*single_integration::get_partial(IntegrationMethod::Booles, func, idx_to_integrate[1], &integration_intervals[1], &current_vec, steps);

    let delta = (integration_intervals[0][1] - integration_intervals[0][0])/(steps as f64);
    let mut multiplier: f64 = 32.0;

    for iter in 0..steps-1
    {
        current_vec[idx_to_integrate[0]] += delta;
        ans += multiplier*single_integration::get_partial(IntegrationMethod::Booles, func, idx_to_integrate[1], &integration_intervals[1], &current_vec, steps);
        
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

    current_vec[idx_to_integrate[0]] = integration_intervals[0][1];

    ans += 7.0*single_integration::get_partial(IntegrationMethod::Booles, func, idx_to_integrate[1], &integration_intervals[1], &current_vec, steps);

    return 2.0*delta*ans/45.0;
}

fn get_gauss_legendre(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: [usize; 2], integration_intervals: &[[f64; 2]; 2], point: &Vec<f64>, order: usize) -> f64
{
    let mut ans = 0.0;
    let abcsissa_coeff = (integration_intervals[0][1] - integration_intervals[0][0])/2.0;
    let intercept = (integration_intervals[0][1] + integration_intervals[0][0])/2.0;

    let (weight, abcsissa) = gl_table::get_gl_weights_and_abscissae(order);

    let mut args = point.clone();

    for iter in 0..order
    {
        args[idx_to_integrate[0]] = abcsissa_coeff*abcsissa[iter] + intercept;

        ans += weight[iter]*single_integration::get_partial(IntegrationMethod::GaussLegendre, func, idx_to_integrate[1], &integration_intervals[1], point, order as u64)
    }

    return abcsissa_coeff*ans;
}

fn get_simpsons(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: [usize; 2], integration_intervals: &[[f64; 2]; 2], point: &Vec<f64>, steps: u64) -> f64
{
    let mut current_vec = point.clone();
    current_vec[idx_to_integrate[0]] = integration_intervals[0][0];

    let mut ans = single_integration::get_partial(IntegrationMethod::Simpsons, func, idx_to_integrate[1], &integration_intervals[1], &current_vec, steps);
    let delta = (integration_intervals[0][1] - integration_intervals[0][0])/(steps as f64);

    let mut multiplier = 3.0;

    for iter in 0..steps-1
    {
        current_vec[idx_to_integrate[0]] += delta;
        ans += multiplier*single_integration::get_partial(IntegrationMethod::Simpsons, func, idx_to_integrate[1], &integration_intervals[1], &current_vec, steps);        

        if (iter + 2) % 3 == 0
        {
            multiplier = 2.0;
        }
        else
        {
            multiplier = 3.0;
        }
    }

    current_vec[idx_to_integrate[0]] = integration_intervals[0][1];

    ans += single_integration::get_partial(IntegrationMethod::Simpsons, func, idx_to_integrate[1], &integration_intervals[1], &current_vec, steps);

    return 3.0*delta*ans/8.0;
}

fn get_trapezoidal(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_integrate: [usize; 2], integration_intervals: &[[f64; 2]; 2], point: &Vec<f64>, steps: u64) -> f64
{
    let mut current_vec = point.clone();
    current_vec[idx_to_integrate[0]] = integration_intervals[0][0];

    let mut ans = single_integration::get_partial(IntegrationMethod::Trapezoidal, func, idx_to_integrate[1], &integration_intervals[1], &current_vec, steps);

    let delta = (integration_intervals[0][1] - integration_intervals[0][0])/(steps as f64);

    for _ in 0..steps-1
    {
        current_vec[idx_to_integrate[0]] += delta;
        ans += 2.0*single_integration::get_partial(IntegrationMethod::Trapezoidal, func, idx_to_integrate[1], &integration_intervals[1], &current_vec, steps);  
    }

    current_vec[idx_to_integrate[0]] = integration_intervals[0][1];

    ans += single_integration::get_partial(IntegrationMethod::Trapezoidal, func, idx_to_integrate[1], &integration_intervals[1], &current_vec, steps);

    return 0.5*delta*ans;
}