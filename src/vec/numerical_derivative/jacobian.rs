use std::vec::Vec;
use crate::vec::numerical_derivative::single_derivative;
use crate::vec::numerical_derivative::mode as mode;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;

/// Returns the jacobian matrix for a given vector of functions
/// Can handle multivariable functions of any order or complexity
/// 
/// The 2-D matrix returned has the structure [[df1/dvar1, df1/dvar2, ... , df1/dvarN],
///                                            [         ...              ], 
///                                            [dfM/dvar1, dfM/dvar2, ... , dfM/dvarN]]
/// 
/// where 'N' is the total number of variables, and 'M' is the total number of functions
/// 
/// consult the helper utility [`multicalc::utils::helper::transpose`] to transpose the matrix shape if required
/// 
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// VectorOfFunctionsCannotBeEmpty -> if function_matrix argument is an empty array
/// 
/// assume our function vector is (x*y*z ,  x^2 + y^2). First define both the functions
/// ```
/// use multicalc::vec::numerical_derivative::jacobian;
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
/// let function_matrix: [&dyn Fn(&Vec<f64>) -> f64; 2] = [&my_func1, &my_func2];
/// let points = vec![1.0, 2.0, 3.0]; //the point around which we want the jacobian matrix
/// 
/// let result = jacobian::get(&function_matrix, &points).unwrap();
/// ```
/// 
/// the above example can also be extended to complex numbers:
///```
/// use multicalc::vec::numerical_derivative::jacobian;
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
/// let function_matrix: [&dyn Fn(&Vec<num_complex::Complex64>) -> num_complex::Complex64; 2] = [&my_func1, &my_func2];
/// 
/// //the point around which we want the jacobian matrix
/// let points = vec![num_complex::c64(1.0, 3.0), num_complex::c64(2.0, 3.5), num_complex::c64(3.0, 0.0)];
/// 
/// let result = jacobian::get(&function_matrix, &points).unwrap();
///``` 
/// 
pub fn get<T: ComplexFloat, const NUM_FUNCS: usize>(function_matrix: &[&dyn Fn(&Vec<T>) -> T; NUM_FUNCS], vector_of_points: &Vec<T>) -> Result<Vec<Vec<T>>, ErrorCode>
{    
    return get_custom(function_matrix, vector_of_points, mode::DEFAULT_STEP_SIZE, mode::DiffMode::CentralFixedStep);
}


///same as [get()] but with the option to change the differentiation mode used, reserved for more advanced users
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// VectorOfFunctionsCannotBeEmpty -> if function_matrix argument is an empty array
/// NumberOfStepsCannotBeZero -> if the derivative step size is zero
pub fn get_custom<T: ComplexFloat, const NUM_FUNCS: usize>(function_matrix: &[&dyn Fn(&Vec<T>) -> T; NUM_FUNCS], vector_of_points: &Vec<T>, step_size: f64, mode: mode::DiffMode) -> Result<Vec<Vec<T>>, ErrorCode>
{
    if function_matrix.is_empty()
    {
        return Err(ErrorCode::VectorOfFunctionsCannotBeEmpty);
    }

    let mut result: Vec<Vec<T>> = std::vec![];

    for row_index in 0..NUM_FUNCS
    {
        let mut cur_row: Vec<T> = std::vec![];
        for col_index in 0..vector_of_points.len()
        {
            cur_row.push(single_derivative::get_partial_custom(&function_matrix[row_index], col_index, vector_of_points, step_size, mode)?);
        }

        result.push(cur_row);
    }
    
    return Ok(result);
}