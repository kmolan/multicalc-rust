use crate::numerical_derivative::single_derivative;
use crate::numerical_derivative::mode as mode;
use num_complex::ComplexFloat;

/// Returns the jacobian matrix for a given vector of functions
/// Can handle multivariable functions of any order or complexity
/// 
/// assume our function vector is (x*y*z ,  x^2 + y^2). First define both the functions
/// ```
/// use multicalc::numerical_derivative::jacobian;
///    let my_func1 = | args: &Vec<f64> | -> f64 
///    { 
///        return args[0]*args[1]*args[2]; //x*y*z
///    };
/// 
///    let my_func2 = | args: &Vec<f64> | -> f64 
///    { 
///        return args[0].powf(2.0) + args[1].powf(2.0); //x^2 + y^2
///    };
/// 
/// //define the function vector
/// let function_matrix: Vec<Box<dyn Fn(&Vec<f64>) -> f64>> = vec![Box::new(my_func1), Box::new(my_func2)];
/// let points = vec![1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix
/// 
/// let result = jacobian::get(&function_matrix, &points);
/// ```
/// 
/// the above example can also be extended to complex numbers:
///```
/// use multicalc::numerical_derivative::jacobian;
///    let my_func1 = | args: &Vec<num_complex::Complex64> | -> num_complex::Complex64 
///    { 
///        return args[0]*args[1]*args[2]; //x*y*z
///    };
/// 
///    let my_func2 = | args: &Vec<num_complex::Complex64> | -> num_complex::Complex64 
///    { 
///        return args[0].powf(2.0) + args[1].powf(2.0); //x^2 + y^2
///    };
/// 
/// //define the function vector
/// let function_matrix: Vec<Box<dyn Fn(&Vec<num_complex::Complex64>) -> num_complex::Complex64>> = vec![Box::new(my_func1), Box::new(my_func2)];
/// 
/// //the point around which we want the jacobian matrix
/// let points = vec![num_complex::c64(1.0, 3.0), num_complex::c64(2.0, 3.5), num_complex::c64(3.0, 0.0)];
/// 
/// let result = jacobian::get(&function_matrix, &points);
///``` 
/// 
pub fn get<T: ComplexFloat>(function_matrix: &Vec<Box<dyn Fn(&Vec<T>) -> T>>, vector_of_points: &Vec<T>) -> Vec<Vec<T>>
{    
    return get_custom(function_matrix, vector_of_points, 0.00001, mode::DiffMode::CentralFixedStep);
}


///same as [get()] but with the option to change the differentiation mode used, reserved for more advanced users
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