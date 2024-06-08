use crate::numerical_derivative::mode as mode;
use crate::numerical_derivative::single_derivative;


pub fn get_3d(vector_field: &[Box<&dyn Fn(&Vec<f64>) -> f64>; 3], point: &Vec<f64>) -> [f64; 3]
{
    return get_3d_custom(vector_field, point, 0.00001, mode::DiffMode::CentralFixedStep);
}

pub fn get_3d_custom(vector_field: &[Box<&dyn Fn(&Vec<f64>) -> f64>; 3], point: &Vec<f64>, step_size: f64, mode: mode::DiffMode) -> [f64; 3]
{
    let mut ans = [0.0, 0.0, 0.0];

    ans[0] = single_derivative::get_partial_custom(vector_field[2].as_ref(), 1, point, step_size, mode) - single_derivative::get_partial_custom(vector_field[1].as_ref(), 2, point, step_size, mode);
    ans[1] = single_derivative::get_partial_custom(vector_field[0].as_ref(), 2, point, step_size, mode) - single_derivative::get_partial_custom(vector_field[2].as_ref(), 0, point, step_size, mode);
    ans[2] = single_derivative::get_partial_custom(vector_field[1].as_ref(), 0, point, step_size, mode) - single_derivative::get_partial_custom(vector_field[0].as_ref(), 1, point, step_size, mode);

    return ans;
}

pub fn get_2d(vector_field: &[Box<&dyn Fn(&Vec<f64>) -> f64>; 2], point: &Vec<f64>) -> f64
{
    return get_2d_custom(vector_field, point, 0.00001, mode::DiffMode::CentralFixedStep);
}

pub fn get_2d_custom(vector_field: &[Box<&dyn Fn(&Vec<f64>) -> f64>; 2], point: &Vec<f64>, step_size: f64, mode: mode::DiffMode) -> f64
{
    return single_derivative::get_partial_custom(vector_field[1].as_ref(), 0, point, step_size, mode)
         - single_derivative::get_partial_custom(vector_field[0].as_ref(), 1, point, step_size, mode);
}