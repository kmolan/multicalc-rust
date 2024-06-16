use crate::numerical_derivative::single_derivative;
use crate::numerical_derivative::mode as mode;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;

/// Returns the double total derivative value for a given function
/// Only ideal for single variable functions
/// 
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the derivative step size is zero
/// 
/// assume we want to differentiate x*Sin(x) . the function would be:
/// ```
///    let my_func = | args: &[f64; 1] | -> f64 
///    { 
///        return args[0]*args[0].sin();
///    };
/// 
///// where args[0] = x
///
///// We also need to define the point at which we want to differentiate. Assuming our point x = 5.0
///// if we then want to differentiate this function over x with a step size of 0.001, we would use:
///
/// use multicalc::numerical_derivative::double_derivative;
/// 
/// let val = double_derivative::get_total(&my_func,      //<- our closure                                           
///                                         5.0,          //<- point around which we want to differentiate
///                                         0.001);       //<- required step size
/// 
/// let expected_val = 2.0*f64::cos(5.0) - 5.0*f64::sin(5.0);
/// assert!(f64::abs(val.unwrap() - expected_val) < 0.00001);
/// ```
/// 
/// the above example can also be extended to complex numbers:
///```
///    let my_func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
///    { 
///        return args[0]*args[0].sin();
///    };
///
///// where args[0] = x
/// 
/// //point of interest is x = (5.0 + 2.5i)
/// let point = num_complex::c64(5.0, 2.5);
/// 
/// use multicalc::numerical_derivative::double_derivative;
///
/// let val = double_derivative::get_total(&my_func,   //<- our closure                                          
///                                        point,      //<- point around which we want to differentiate
///                                        0.001);     //<- required step size
/// 
/// let expected_val = 2.0*point.cos() - point*point.sin();
/// assert!(num_complex::ComplexFloat::abs(val.unwrap().re - expected_val.re) < 0.0001);
/// assert!(num_complex::ComplexFloat::abs(val.unwrap().im - expected_val.im) < 0.0001);
///``` 
/// 
pub fn get_total<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, point: T, step: f64) -> Result<T, ErrorCode>
{
    return get_total_custom(func, point, step, mode::DiffMode::CentralFixedStep);
}


///same as [get_total()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_total_custom<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, point: T, step: f64, mode: mode::DiffMode) -> Result<T, ErrorCode>
{
    let vec_point = [point; NUM_VARS];

    match mode
    {
        mode::DiffMode::ForwardFixedStep => return get_forward_difference(func, &[0, 0], &vec_point, step),
        mode::DiffMode::BackwardFixedStep => return get_backward_difference(func, &[0, 0], &vec_point, step),
        mode::DiffMode::CentralFixedStep => return get_central_difference(func, &[0, 0], &vec_point, step) 
    }
}


/// Returns the partial double derivative value for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the derivative step size is zero
/// IndexToDerivativeOutOfRange -> if the value of idx_to_derivate is greater than the number of variables
/// 
/// assume we want to differentiate y*sin(x) + x*cos(y) + x*y*e^z . the function would be:
/// ```
///    let my_func = | args: &[f64; 3] | -> f64 
///    { 
///        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
///    };
/// 
///// where args[0] = x, args[1] = y and args[2] = z. Also, we know our function must accept 3 arguments.
///
///// We also need to define the point at which we want to differentiate. Assuming our point is (1.0, 2.0, 3.0)
///
/// let point = [1.0, 2.0, 3.0];
///
///// For double differentiation, we can choose which variables we want to differentiate over
///// For a total double differentiation over x, idx = [0, 0]  (this is same as calling get_total() )
///// For a partial mixed differentiation, say first over x and then y, idx = [0, 1] since 'x' is the 0th, and 'y' the 1st in our function
///
///// if we then want to partially differentiate this function first over x then y, for (x, y, z) = (1.0, 2.0, 3.0) with a step size of 0.001, we would use:
///
/// use multicalc::numerical_derivative::double_derivative;
/// 
/// let val = double_derivative::get_partial(&my_func,   //<- our closure                
///                                          &[0, 1],    //<- idx, index of variables we want to differentiate                            
///                                          &point,     //<- point around which we want to differentiate
///                                          0.001);     //<- required step size
/// 
/// let expected_value = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
/// assert!(f64::abs(val.unwrap() - expected_value) < 0.001);
/// ```
/// 
/// the above example can also be extended to complex numbers:
///```
///    let my_func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
///    { 
///        return args[1]*args[0].sin() + args[0]*args[1].cos() + args[0]*args[1]*args[2].exp();
///    };
///
///// where args[0] = x, args[1] = y and args[2] = z
/// 
/// //point of interest is (x, y, z) = (1.0 + 4.0i, 2.0 + 2.5i, 3.0 + 0.0i)
/// let point = [num_complex::c64(1.0, 4.0), num_complex::c64(2.0, 2.5), num_complex::c64(3.0, 0.0)];
/// 
/// use multicalc::numerical_derivative::double_derivative;
///
/// let val = double_derivative::get_partial(&my_func,   //<- our closure                
///                                          &[0, 1],    //<- idx, index of variables we want to differentiate                            
///                                          &point,     //<- point around which we want to differentiate
///                                          0.001);     //<- required step size
/// 
/// let expected_val = point[0].cos() - point[1].sin() + point[2].exp();
/// assert!(num_complex::ComplexFloat::abs(val.unwrap().re - expected_val.re) < 0.0001);
/// assert!(num_complex::ComplexFloat::abs(val.unwrap().im - expected_val.im) < 0.0001);
///``` 
/// 
pub fn get_partial<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS], step: f64) -> Result<T, ErrorCode>
{
    return get_partial_custom(func, idx_to_derivate, point, step, mode::DiffMode::CentralFixedStep);
}

///same as [get_partial()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_partial_custom<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS], step: f64, mode: mode::DiffMode) -> Result<T, ErrorCode>
{
    match mode
    {
        mode::DiffMode::ForwardFixedStep => return get_forward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::BackwardFixedStep => return get_backward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::CentralFixedStep => return get_central_difference(func, idx_to_derivate, point, step) 
    }
}



pub fn get_forward_difference<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS], step: f64) -> Result<T, ErrorCode>
{
    let f0 = single_derivative::get_partial_custom(func, idx_to_derivate[1], point, step, mode::DiffMode::ForwardFixedStep)?;

    let mut f1_point = *point;
    f1_point[idx_to_derivate[0]] = f1_point[idx_to_derivate[0]] + T::from(step).unwrap();
    let f1 = single_derivative::get_partial_custom(func, idx_to_derivate[1], &f1_point, step, mode::DiffMode::ForwardFixedStep)?;

    return Ok((f1 - f0)/T::from(step).unwrap());
}

pub fn get_backward_difference<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS], step: f64) -> Result<T, ErrorCode>
{
    let mut f0_point = *point;
    f0_point[idx_to_derivate[0]] = f0_point[idx_to_derivate[0]] - T::from(step).unwrap();
    let f0 = single_derivative::get_partial_custom(func, idx_to_derivate[1], &f0_point, step, mode::DiffMode::BackwardFixedStep)?;

    let f1 = single_derivative::get_partial_custom(func, idx_to_derivate[1], point, step, mode::DiffMode::BackwardFixedStep)?;

    return Ok((f1 - f0)/T::from(step).unwrap());
}

pub fn get_central_difference<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_derivate: &[usize; 2], point: &[T; NUM_VARS], step: f64) -> Result<T, ErrorCode>
{
    let mut f0_point = *point;
    f0_point[idx_to_derivate[0]] = f0_point[idx_to_derivate[0]] - T::from(step).unwrap();
    let f0 = single_derivative::get_partial_custom(func, idx_to_derivate[1], &f0_point, step, mode::DiffMode::CentralFixedStep)?;

    let mut f1_point = *point;
    f1_point[idx_to_derivate[0]] = f1_point[idx_to_derivate[0]] + T::from(step).unwrap();
    let f1 = single_derivative::get_partial_custom(func, idx_to_derivate[1], &f1_point, step, mode::DiffMode::CentralFixedStep)?;

    return Ok((f1 - f0)/(T::from(2.0*step).unwrap()));
}

