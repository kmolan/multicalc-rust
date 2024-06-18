use crate::numerical_derivative::single_derivative;
use crate::numerical_derivative::mode as mode;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;

/// Returns the jacobian matrix for a given vector of functions
/// Can handle multivariable functions of any order or complexity
/// 
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// VectorOfFunctionsCannotBeEmpty -> if function_matrix argument is an empty array
/// 
/// assume our function vector is (x*y*z ,  x^2 + y^2). First define both the functions
/// ```
/// use multicalc::numerical_derivative::jacobian;
///    let my_func1 = | args: &[f64; 3] | -> f64 
///    { 
///        return args[0]*args[1]*args[2]; //x*y*z
///    };
/// 
///    let my_func2 = | args: &[f64; 3] | -> f64 
///    { 
///        return args[0].powf(2.0) + args[1].powf(2.0); //x^2 + y^2
///    };
/// 
/// //define the function vector
/// let function_matrix: [&dyn Fn(&[f64; 3]) -> f64; 2] = [&my_func1, &my_func2];
/// let points = [1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix
/// 
/// let result = jacobian::get(&function_matrix, &points).unwrap();
/// ```
/// 
/// the above example can also be extended to complex numbers:
///```
/// use multicalc::numerical_derivative::jacobian;
///    let my_func1 = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
///    { 
///        return args[0]*args[1]*args[2]; //x*y*z
///    };
/// 
///    let my_func2 = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
///    { 
///        return args[0].powf(2.0) + args[1].powf(2.0); //x^2 + y^2
///    };
/// 
/// //define the function vector
/// let function_matrix: [&dyn Fn(&[num_complex::Complex64; 3]) -> num_complex::Complex64; 2] = [&my_func1, &my_func2];
/// 
/// //the point around which we want the jacobian matrix
/// let points = [num_complex::c64(1.0, 3.0), num_complex::c64(2.0, 3.5), num_complex::c64(3.0, 0.0)];
/// 
/// let result = jacobian::get(&function_matrix, &points).unwrap();
///``` 
/// 
pub fn get<T: ComplexFloat, const NUM_FUNCS: usize, const NUM_VARS: usize>(function_matrix: &[&dyn Fn(&[T; NUM_VARS]) -> T; NUM_FUNCS], vector_of_points: &[T; NUM_VARS]) -> Result<[[T; NUM_VARS]; NUM_FUNCS], ErrorCode>
{    
    return get_custom(function_matrix, vector_of_points, mode::DEFAULT_STEP_SIZE, mode::DiffMode::CentralFixedStep);
}


///same as [get()] but with the option to change the differentiation mode used, reserved for more advanced users
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// VectorOfFunctionsCannotBeEmpty -> if function_matrix argument is an empty array
/// NumberOfStepsCannotBeZero -> if the derivative step size is zero
pub fn get_custom<T: ComplexFloat, const NUM_FUNCS: usize, const NUM_VARS: usize>(function_matrix: &[&dyn Fn(&[T; NUM_VARS]) -> T; NUM_FUNCS], vector_of_points: &[T; NUM_VARS], step_size: f64, mode: mode::DiffMode) -> Result<[[T; NUM_VARS]; NUM_FUNCS], ErrorCode>
{
    if function_matrix.is_empty()
    {
        return Err(ErrorCode::VectorOfFunctionsCannotBeEmpty);
    }

    let mut result = [[T::zero(); NUM_VARS]; NUM_FUNCS];

    for row_index in 0..NUM_FUNCS
    {
        for col_index in 0..NUM_VARS
        {
            result[row_index][col_index] = single_derivative::get_partial_custom(&function_matrix[row_index], col_index, vector_of_points, step_size, mode)?;
        }
    }
    
    return Ok(result);
}