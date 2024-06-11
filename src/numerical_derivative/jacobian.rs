use crate::numerical_derivative::single_derivative;
use crate::numerical_derivative::mode as mode;
use num_complex::ComplexFloat;

pub fn get<T: ComplexFloat>(function_matrix: &Vec<Box<dyn Fn(&Vec<T>) -> T>>, vector_of_points: &Vec<T>) -> Vec<Vec<T>>
{    
    return get_custom(function_matrix, vector_of_points, 0.00001, mode::DiffMode::CentralFixedStep);
}


//for more advanced users who want to control the parameters used for differentiation
pub fn get_custom<T: ComplexFloat>(function_matrix: &Vec<Box<dyn Fn(&Vec<T>) -> T>>, vector_of_points: &Vec<T>, step_size: f64, mode: mode::DiffMode) -> Vec<Vec<T>>
{
    assert!(function_matrix.len() > 0, "function matrix cannot be empty");
    assert!(vector_of_points.len() > 0, "points cannot be empty");

    let mut result = vec![vec![T::zero(); vector_of_points.len()]; function_matrix.len()];

    for row_index in 0..function_matrix.len()
    {
        for col_index in 0..vector_of_points.len()
        {
            result[row_index][col_index] = single_derivative::get_partial_custom(&function_matrix[row_index], col_index, vector_of_points, step_size, mode);
        }
    }
    
    return result;
}