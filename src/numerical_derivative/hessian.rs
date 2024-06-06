use crate::numerical_derivative::double_derivative as double_derivative;
use crate::numerical_derivative::mode as mode;

pub fn get(function: &dyn Fn(&Vec<f64>) -> f64, vector_of_points: &Vec<f64>) -> Vec<Vec<f64>>
{
    return get_custom(function, vector_of_points, 0.00001, mode::DiffMode::CentralFixedStep);
}

pub fn get_custom(function: &dyn Fn(&Vec<f64>) -> f64, vector_of_points: &Vec<f64>, step_size: f64, mode: mode::DiffMode) -> Vec<Vec<f64>>
{
    assert!(vector_of_points.len() > 0, "points cannot be empty");

    let mut result = vec![vec![f64::NAN; vector_of_points.len()]; vector_of_points.len()];

    for row_index in 0..vector_of_points.len()
    {
        for col_index in 0..vector_of_points.len()
        {
            if result[row_index][col_index].is_nan()
            {
                result[row_index][col_index] = double_derivative::get_partial_custom(function, &[row_index, col_index], vector_of_points, step_size, mode);

                result[col_index][row_index] = result[row_index][col_index]; //exploit the fact that a hessian is a symmetric matrix
            }
        }
    }

    return result;
}