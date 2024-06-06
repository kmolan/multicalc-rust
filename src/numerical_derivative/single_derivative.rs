use crate::numerical_derivative::mode as mode;


/// Returns the single total derivative value for a given function
/// Only ideal for single variable functions
/// 
/// assume we want to differentiate 2*x*x the function would be:
/// ```
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return 2.0*args[0]*args[0];
///    };
///
//// where args[0] = x
///
//// We also need to define the point at which we want to differentiate. Assuming our point is x = 1.0
//// if we then want to differentiate this function over x with a step size of 0.001, we would use:
/// 
/// use multicalc::numerical_derivative::single_derivative;
///
/// let val = single_derivative::get_total(&my_func,    //<- our closure                                          
///                                         1.0,        //<- point around which we want to differentiate
///                                         0.001);     //<- required step size
/// 
/// assert!(f64::abs(val - 4.0) < 0.00001);
///```
///
pub fn get_total(func: &dyn Fn(&Vec<f64>) -> f64, point: f64, step: f64) -> f64
{
    return get_total_custom(func, point, step, mode::DiffMode::CentralFixedStep);
}


///same as [get_total()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_total_custom(func: &dyn Fn(&Vec<f64>) -> f64, point: f64, step: f64, mode: mode::DiffMode) -> f64
{
    assert!(step != 0.0, "step size cannot be zero");

    let vec_point = vec![point];

    match mode
    {
        mode::DiffMode::ForwardFixedStep => return get_forward_difference(func, 0, &vec_point, step),
        mode::DiffMode::BackwardFixedStep => return get_backward_difference(func, 0, &vec_point, step),
        mode::DiffMode::CentralFixedStep => return get_central_difference(func, 0, &vec_point, step),
    }
}


/// Returns the single partial derivative value for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// assume we want to differentiate y*sin(x) + x*cos(y) + x*y*e^z. the function would be:
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
//// if we then want to differentiate this function over x with a step size of 0.001, we would use:
/// 
/// use multicalc::numerical_derivative::single_derivative;
///
/// let val = single_derivative::get_partial(&my_func,    //<- our closure                 
///                                          0,           //<- index of variable we want to differentiate, in this case "x", which is 0                           
///                                          &point,      //<- point around which we want to differentiate
///                                          0.001);      //<- required step size
/// 
/// let expected_value = 2.0*f64::cos(1.0) + f64::cos(2.0) + 2.0*f64::exp(3.0);
/// assert!(f64::abs(val - expected_value) < 0.00001);
///```
/// 
pub fn get_partial(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: usize, point: &Vec<f64>, step: f64) -> f64
{
    return get_partial_custom(func, idx_to_derivate, point, step, mode::DiffMode::CentralFixedStep);
}


///same as [get_partial()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_partial_custom(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: usize, point: &Vec<f64>, step: f64, mode: mode::DiffMode) -> f64
{
    assert!(step != 0.0, "step size cannot be zero");

    let num_variables = point.len();

    assert!(num_variables != 0, "list of points cannot be empty");
    assert!(idx_to_derivate < num_variables, "idx value is greater than the points vector length");

    match mode
    {
        mode::DiffMode::ForwardFixedStep => return get_forward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::BackwardFixedStep => return get_backward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::CentralFixedStep => return get_central_difference(func, idx_to_derivate, point, step),
    }
}

fn get_forward_difference(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: usize, point: &Vec<f64>, step: f64) -> f64
{
    let f0_args = point;

    let mut f1_args = point.clone();
    f1_args[idx_to_derivate] += step; 

    let f0: f64 = func(&f0_args);
    let f1: f64 = func(&f1_args);

    return (f1 - f0)/step;
}

fn get_backward_difference(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: usize, point: &Vec<f64>, step: f64) -> f64
{
    let mut f0_args = point.clone();
    f0_args[idx_to_derivate] -= step; 

    let f1_args = point;

    let f0: f64 = func(&f0_args);
    let f1: f64 = func(&f1_args);

    return (f1 - f0)/step;
}

fn get_central_difference(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: usize, point: &Vec<f64>, step: f64) -> f64
{
    let mut f0_args = point.clone();
    f0_args[idx_to_derivate] -= step;

    let mut f1_args = point.clone();
    f1_args[idx_to_derivate] += step; 

    let f0: f64 = func(&f0_args);
    let f1: f64 = func(&f1_args);

    return (f1 - f0)/(2.0*step);
}