use std::vec::Vec;

use crate::vec::numerical_derivative::double_derivative;
use crate::vec::numerical_derivative::mode as mode;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;


/// Returns the hessian matrix for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// The 2-D matrix returned has the structure [[d2f/d2var1, d2f/dvar1*dvar2, ... , d2f/dvar1*dvarN], 
///                                            [                   ...                            ], 
///                                            [d2f/dvar1*dvarN, d2f/dvar2*dvarN, ... , dfM/d2varN]]
/// where 'N' is the total number of variables
/// 
/// assume our function is y*sin(x) + 2*x*e^y. First define the function
/// ```
/// use multicalc::vec::numerical_derivative::hessian;
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
/// use multicalc::vec::numerical_derivative::hessian;
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
    return get_custom(function, vector_of_points, mode::DEFAULT_STEP_SIZE, mode::DiffMode::CentralFixedStep).unwrap();
}

///same as [get()] but with the option to change the differentiation mode used, reserved for more advanced users
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the derivative step size is zero
pub fn get_custom<T: ComplexFloat>(function: &dyn Fn(&Vec<T>) -> T, vector_of_points: &Vec<T>, step_size: f64, mode: mode::DiffMode) -> Result<Vec<Vec<T>>, ErrorCode>
{
    let num_vars = vector_of_points.len();
    
    let mut result = std::vec![std::vec![T::from(f64::NAN).unwrap(); num_vars]; num_vars];

    for row_index in 0..num_vars
    {
        for col_index in 0..num_vars
        {
            if result[row_index][col_index].is_nan()
            {
                result[row_index][col_index] = double_derivative::get_partial_custom(function, &[row_index, col_index], vector_of_points, step_size, mode)?;

                result[col_index][row_index] = result[row_index][col_index]; //exploit the fact that a hessian is a symmetric matrix
            }
        }
    }

    return Ok(result);
}