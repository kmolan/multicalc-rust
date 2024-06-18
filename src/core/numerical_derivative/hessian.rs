use crate::core::numerical_derivative::double_derivative as double_derivative;
use crate::core::numerical_derivative::mode as mode;
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
/// use multicalc::core::numerical_derivative::hessian;
///    let my_func = | args: &[f64; 2] | -> f64 
///    { 
///        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
///    };
/// 
/// let points = [1.0, 2.0]; //the point around which we want the hessian matrix
/// 
/// let result = hessian::get(&my_func, &points);
/// ```
/// 
/// the above example can also be extended to complex numbers:
///```
/// use multicalc::core::numerical_derivative::hessian;
///    let my_func = | args: &[num_complex::Complex64; 2] | -> num_complex::Complex64 
///    { 
///        return args[1]*args[0].sin() + 2.0*args[0]*args[1].exp();
///    };
/// 
/// //the point around which we want the hessian matrix
/// let points = [num_complex::c64(1.0, 2.5), num_complex::c64(2.0, 5.0)];
/// 
/// let result = hessian::get(&my_func, &points);
///``` 
/// 
pub fn get<T: ComplexFloat, const NUM_VARS: usize>(function: &dyn Fn(&[T; NUM_VARS]) -> T, vector_of_points: &[T; NUM_VARS]) -> [[T; NUM_VARS]; NUM_VARS]
{
    return get_custom(function, vector_of_points, mode::DEFAULT_STEP_SIZE, mode::DiffMode::CentralFixedStep).unwrap();
}

///same as [get()] but with the option to change the differentiation mode used, reserved for more advanced users
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the derivative step size is zero
pub fn get_custom<T: ComplexFloat, const NUM_VARS: usize>(function: &dyn Fn(&[T; NUM_VARS]) -> T, vector_of_points: &[T; NUM_VARS], step_size: f64, mode: mode::DiffMode) -> Result<[[T; NUM_VARS]; NUM_VARS], ErrorCode>
{
    let mut result = [[T::from(f64::NAN).unwrap(); NUM_VARS]; NUM_VARS];

    for row_index in 0..NUM_VARS
    {
        for col_index in 0..NUM_VARS
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