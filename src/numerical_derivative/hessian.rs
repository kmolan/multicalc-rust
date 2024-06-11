use crate::numerical_derivative::double_derivative as double_derivative;
use crate::numerical_derivative::mode as mode;
use num_complex::ComplexFloat;


/// Returns the hessian matrix for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// assume our function is y*sin(x) + 2*x*e^y. First define the function
/// ```
/// use multicalc::numerical_derivative::hessian;
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
///    };
/// 
/// let points = vec![1.0, 2.0]; //the point around which we want the hessian matrix
/// 
/// let result = hessian::get(&my_func, &points);
/// ```
/// 
/// the above example can also be extended to complex numbers:
///```
/// use multicalc::numerical_derivative::hessian;
///    let my_func = | args: &Vec<num_complex::Complex64> | -> num_complex::Complex64 
///    { 
///        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
///    };
/// 
/// //the point around which we want the hessian matrix
/// let points = vec![num_complex::c64(1.0, 2.5), num_complex::c64(2.0, 5.0)];
/// 
/// let result = hessian::get(&my_func, &points);
///``` 
/// 
pub fn get<T: ComplexFloat>(function: &dyn Fn(&Vec<T>) -> T, vector_of_points: &Vec<T>) -> Vec<Vec<T>>
{
    return get_custom(function, vector_of_points, 0.00001, mode::DiffMode::CentralFixedStep);
}

///same as [get()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_custom<T: ComplexFloat>(function: &dyn Fn(&Vec<T>) -> T, vector_of_points: &Vec<T>, step_size: f64, mode: mode::DiffMode) -> Vec<Vec<T>>
{
    assert!(vector_of_points.len() > 0, "points cannot be empty");

    let mut result = vec![vec![T::from(f64::NAN).unwrap(); vector_of_points.len()]; vector_of_points.len()];

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