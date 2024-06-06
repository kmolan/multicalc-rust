use crate::numerical_derivative::single_derivative;
use crate::numerical_derivative::mode as mode;

/// Returns the double derivative value for a given function
/// Only ideal for single variable functions
/// 
/// assume we want to differentiate x*Sin(x) . the function would be:
/// ```
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return args[0]*args[0].sin();
///    };
/// 
//// where args[0] = x
///
//// We also need to define the point at which we want to differentiate. Assuming our point x = 5.0
//// if we then want to differentiate this function first over x with a step size of 0.001, we would use:
///
/// use multicalc::numerical_derivative::double_derivative;
/// use multicalc::numerical_derivative::mode::DiffMode;
/// 
/// let val = double_derivative::get_simple(&my_func,     //<- our closure                                           
///                                         5.0,          //<- point around which we want to differentiate
///                                         0.001);       //<- required step size
/// 
/// let expected_val = 2.0*f64::cos(5.0) - 5.0*f64::sin(5.0);
/// assert!(f64::abs(val - expected_val) < 0.00001);
/// ```
/// 
pub fn get_simple(func: &dyn Fn(&Vec<f64>) -> f64, point: f64, step: f64) -> f64
{
    return get_simple_custom(func, point, step, &mode::DiffMode::CentralFixedStep);
}


///same as [get_simple()] but with the option to change the differentiation mode used
pub fn get_simple_custom(func: &dyn Fn(&Vec<f64>) -> f64, point: f64, step: f64, mode: &mode::DiffMode) -> f64
{
    let vec_point = vec![point];

    match mode
    {
        mode::DiffMode::ForwardFixedStep => return get_forward_difference(func, &[0, 0], &vec_point, step),
        mode::DiffMode::BackwardFixedStep => return get_backward_difference(func, &[0, 0], &vec_point, step),
        mode::DiffMode::CentralFixedStep => return get_central_difference(func, &[0, 0], &vec_point, step) 
    }
}


/// Returns the double derivative value for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// assume we want to differentiate y*sin(x) + x*cos(y) + x*y*e^z . the function would be:
/// ```
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
///    };
/// 
//// where args[0] = x, args[1] = y and args[2] = z. Also, we know our function must accept 3 arguments.
///
//// We also need to define the point at which we want to differentiate. Assuming our point is (1.0, 2.0, 3.0)
///
/// let point = vec![1.0, 2.0, 3.0];
///
//// For double differentiation, we can choose which variables we want to differentiate over
//// For a simple double differentiation over x, idx = [0, 0] since 'x' is the 0th variable in our function
//// For a partial mixed differentiation, say first over x and then y, idx = [0, 1] since 'x' is the 0th, and 'y' the 1st in our function
///
//// if we then want to partially differentiate this function first over x then y, for (x, y, z) = (1.0, 2.0, 3.0) with a step size of 0.001, we would use:
///
/// use multicalc::numerical_derivative::double_derivative;
/// use multicalc::numerical_derivative::mode::DiffMode;
/// 
/// let val = double_derivative::get_partial(&my_func,   //<- our closure                
///                                          &[0, 1],    //<- idx, index of variables we want to differentiate                            
///                                          &point,     //<- point around which we want to differentiate
///                                          0.001);     //<- required step size
/// 
/// let expected_value = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
/// assert!(f64::abs(val - expected_value) < 0.001);
/// ```
/// 
pub fn get_partial(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: &[usize; 2], point: &Vec<f64>, step: f64) -> f64
{
    return get_partial_custom(func, idx_to_derivate, point, step, &mode::DiffMode::CentralFixedStep);
}

///same as [get_partial()] but with the option to change the differentiation mode used
pub fn get_partial_custom(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: &[usize; 2], point: &Vec<f64>, step: f64, mode: &mode::DiffMode) -> f64
{
    match mode
    {
        mode::DiffMode::ForwardFixedStep => return get_forward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::BackwardFixedStep => return get_backward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::CentralFixedStep => return get_central_difference(func, idx_to_derivate, point, step) 
    }
}



pub fn get_forward_difference(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: &[usize; 2], point: &Vec<f64>, step: f64) -> f64
{
    let f0 = single_derivative::get_partial_custom(func, idx_to_derivate[1], point, step, &mode::DiffMode::ForwardFixedStep);

    let mut f1_point = point.clone();
    f1_point[idx_to_derivate[0]] += step;
    let f1 = single_derivative::get_partial_custom(func, idx_to_derivate[1], &f1_point, step, &mode::DiffMode::ForwardFixedStep);

    return (f1 - f0)/step;
}

pub fn get_backward_difference(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: &[usize; 2], point: &Vec<f64>, step: f64) -> f64
{
    let mut f0_point = point.clone();
    f0_point[idx_to_derivate[0]] -= step;
    let f0 = single_derivative::get_partial_custom(func, idx_to_derivate[1], &f0_point, step, &mode::DiffMode::BackwardFixedStep);

    let f1 = single_derivative::get_partial_custom(func, idx_to_derivate[1], point, step, &mode::DiffMode::BackwardFixedStep);

    return (f1 - f0)/step;
}

pub fn get_central_difference(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: &[usize; 2], point: &Vec<f64>, step: f64) -> f64
{
    let mut f0_point = point.clone();
    f0_point[idx_to_derivate[0]] -= step;
    let f0 = single_derivative::get_partial_custom(func, idx_to_derivate[1], &f0_point, step, &mode::DiffMode::CentralFixedStep);

    let mut f1_point = point.clone();
    f1_point[idx_to_derivate[0]] += step;
    let f1 = single_derivative::get_partial_custom(func, idx_to_derivate[1], &f1_point, step, &mode::DiffMode::CentralFixedStep);

    return (f1 - f0)/(2.0*step);
}

