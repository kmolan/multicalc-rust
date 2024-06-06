use crate::numerical_derivative::double_derivative;
use crate::numerical_derivative::mode as mode;

/*
assume we want to differentiate y*sin(x) + x*cos(y) + x*y*e^z . the function would be:
    let func = | args: &Vec<f64> | -> f64 
    { 
        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
    };
where args[0] = x, args[1] = y and args[2] = z. Also, we know our function must accept 3 arguments.

We also need to define the point at which we want to differentiate. Assuming our point is (1.0, 2.0, 3.0)

let point = !vec[1.0, 2.0, 3.0];

For double differentiation, we can choose which variables we want to differentiate over
For a simple double differentiation over x, idx = [0, 0, 0] since 'x' is the 0th variable in our function
For a partial mixed differentiation, say first over x, then y, then z, idx = [0, 1, 2] since 'x' is 0th, 'y' the 1st and 'z' the 2nd in our function

if we then want to partially differentiate this function (x, y, z) = (1.0, 2.0, 3.0) with a step size of 0.001, we would use:

triple_derivative::get(&my_func,                    <- our closure                
                       &idx,                        <- index of variables we want to differentiate                            
                       &point,                      <- point around which we want to differentiate
                       0.001,                       <- required step size
                       &DiffMode::CentralFixedStep  <- required method of differentiation
*/
pub fn get(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: &[usize; 3], point: &Vec<f64>, step: f64, mode: &mode::DiffMode) -> f64
{
    match mode
    {
        mode::DiffMode::ForwardFixedStep => return get_forward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::BackwardFixedStep => return get_backward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::CentralFixedStep => return get_central_difference(func, idx_to_derivate, point, step) 
    }
}

pub fn get_forward_difference(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: &[usize; 3], point: &Vec<f64>, step: f64) -> f64
{
    let f0 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], point, step, &mode::DiffMode::ForwardFixedStep);

    let mut f1_point = point.clone();
    f1_point[idx_to_derivate[0]] += step;
    let f1 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], &f1_point, step, &mode::DiffMode::ForwardFixedStep);

    return (f1 - f0)/step;    
}

pub fn get_backward_difference(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: &[usize; 3], point: &Vec<f64>, step: f64) -> f64
{
    let mut f0_point = point.clone();
    f0_point[idx_to_derivate[0]] -= step;
    let f0 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], &f0_point, step, &mode::DiffMode::BackwardFixedStep);

    let f1 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], point, step, &mode::DiffMode::BackwardFixedStep);

    return (f1 - f0)/step;
}

pub fn get_central_difference(func: &dyn Fn(&Vec<f64>) -> f64, idx_to_derivate: &[usize; 3], point: &Vec<f64>, step: f64) -> f64
{
    let mut f0_point = point.clone();
    f0_point[idx_to_derivate[0]] -= step;
    let f0 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], &f0_point, step, &mode::DiffMode::CentralFixedStep);

    let mut f1_point = point.clone();
    f1_point[idx_to_derivate[0]] += step;
    let f1 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], &f1_point, step, &mode::DiffMode::CentralFixedStep);

    return (f1 - f0)/(2.0*step);
}
