use crate::numerical_integration::mode;

use num_complex::ComplexFloat;


/// Returns the total single integration value for a given function
/// Only ideal for single variable functions
/// 
/// NOTE: Returns a Result<T, &'static str>
/// Possible &'static str are:
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
/// 
/// assume we want to integrate 2*x . the function would be:
/// ```
///    let my_func = | args: &[f64; 1] | -> f64 
///    { 
///        return 2.0*args[0];
///    };
///
/////where args[0] = x. We also need to define the integration limit around which we want to integrate. Assuming our limit is [0.0, 2.0]
///
/// let integration_limit = [0.0, 2.0];
/// 
/// use multicalc::numerical_integration::single_integration;
///
/// let val = single_integration::get_total(&my_func,              //<- our closure                 
///                                         &integration_limit);   //<- The integration limit needed   
/// 
/// assert!(f64::abs(val.unwrap() - 4.0) < 0.00001);
///```
/// 
/// the above method example can also be extended to complex numbers:
/// ```
///    let my_func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
///    { 
///        return 2.0*args[0];
///    };
///
/////where args[0] = x. Assuming our integration limit is (0.0 + 0.0i) to (2.0 + 2.0i)
///
/// let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 2.0)];
/// 
/// use multicalc::numerical_integration::single_integration;
///
/// let val = single_integration::get_total(&my_func,              //<- our closure                 
///                                         &integration_limit);   //<- The integration limit needed   
/// 
/// assert!(num_complex::ComplexFloat::abs(val.unwrap().re - 0.0) < 0.00001);
/// assert!(num_complex::ComplexFloat::abs(val.unwrap().im - 8.0) < 0.00001);
///```
/// 
pub fn get_total<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limit: &[T; 2]) -> Result<T, &'static str>
{
    return get_total_custom(mode::IntegrationMethod::Trapezoidal, func, integration_limit, mode::DEFAULT_TOTAL_ITERATIONS);
}


///same as [get_total()] but with the option to change the integration parameters used, reserved for more advanced users
/// NOTE: Returns a Result<T, &'static str>
/// Possible &'static str are:
/// NumberOfStepsCannotBeZero -> if the number of steps argument, "n" is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
/// GaussLegendreOrderOutOfRange-> if integration_method == mode::IntegrationMethod::GaussLegendre, and if n < 2 or n > 15
/// The argument 'n' denotes the number of steps to be used. However, for [`mode::IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
pub fn get_total_custom<T: ComplexFloat, const NUM_VARS: usize>(integration_method: mode::IntegrationMethod, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limit: &[T; 2], n: u64) -> Result<T, &'static str>
{
    let point = [integration_limit[1]; NUM_VARS];

    return Ok(T::zero());
}


/// Returns the partial single integration value for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// NOTE: Returns a Result<T, &'static str>
/// Possible &'static str are:
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
/// 
/// assume we want to partially integrate for y for the equation 2.0*x + y*z. The function would be:
/// ```
///    let my_func = | args: &[f64; 3] | -> f64 
///    { 
///        return 2.0*args[0] + args[1]*args[2];
///    };
///
/////where args[0] = x, args[1] = y and args[2] = z. We also need to define the integration limit around which we want to integrate. Assuming our limit is [0.0, 2.0]
///
/// let integration_limit = [0.0, 2.0];
/// 
///// For partial integration to work, we also need to define the static values for the remaining variables. 
///// Assuming x = 1.0, z = 3.0 and we want to integrate over y:
/// 
/// let point = [1.0, 2.0, 3.0];
///
///// Note above that the point vector has the same number of elements as the number of elements my_func expects. 
///// The element to integrate, y, has index = 1. We MUST therefore make the point vector's 1st element the same as the integration intervals's upper limit which is 2.0
/// 
///// if we then want to integrate this function over y with 100 steps, we would use:
/// 
/// use multicalc::numerical_integration::single_integration;
///
/// let val = single_integration::get_partial(&my_func,                        //<- our closure   
///                                           1,                               //<- index of variable we want to integrate, in this case "y", which is 1 
///                                           &integration_limit,              //<- The integration limit needed 
///                                           &point);                         //<- The final point with all x,y,z values
/// 
/// assert!(f64::abs(val.unwrap() - 10.0) < 0.00001);
///```
/// 
/// the above method example can also be extended to complex numbers:
/// ```
///    let my_func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
///    { 
///        return 2.0*args[0] + args[1]*args[2];
///    };
///
/////where args[0] = x, args[1] = y and args[2] = z. Assuming our integration limit is (0.0 + 0.0i) to (2.0 + 0.0i)
///
/// let integration_limit = [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 0.0)];
/// 
///// For partial integration to work, we also need to define the static values for the remaining variables. 
///// Assuming x = 1.0 + 1.0i, z = 3.0 + 0.5i and we want to integrate over y:
/// 
/// let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];
///
///// Note above that the point vector has the same number of elements as the number of elements my_func expects. 
///// The element to integrate, y, has index = 1. We MUST therefore make the point vector's 1st element the same as the integration intervals's upper limit which is 2.0 + 0.0i
/// 
///// if we then want to integrate this function over y, we would use:
/// 
/// use multicalc::numerical_integration::single_integration;
///
/// let val = single_integration::get_partial(&my_func,                        //<- our closure   
///                                           1,                               //<- index of variable we want to integrate, in this case "y", which is 1 
///                                           &integration_limit,              //<- The integration limit needed 
///                                           &point);                         //<- The final point with all x,y,z values
/// 
/// assert!(num_complex::ComplexFloat::abs(val.unwrap().re - 10.0) < 0.00001);
/// assert!(num_complex::ComplexFloat::abs(val.unwrap().im - 5.0) < 0.00001);
///```
/// 
pub fn get_partial<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
{
    return get_partial_custom(mode::IntegrationMethod::Trapezoidal, func, idx_to_integrate, integration_limit, point, mode::DEFAULT_TOTAL_ITERATIONS);
}


///same as [get_partial()] but with the option to change the integration parameters used, reserved for more advanced user
/// The argument 'n' denotes the number of steps to be used. However, for [`mode::IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// NOTE: Returns a Result<T, &'static str>
/// Possible &'static str are:
/// NumberOfStepsCannotBeZero -> if the number of steps argument, "n" is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
/// GaussLegendreOrderOutOfRange-> if integration_method == mode::IntegrationMethod::GaussLegendre, and if n < 2 or n > 15
pub fn get_partial_custom<T: ComplexFloat, const NUM_VARS: usize>(integration_method: mode::IntegrationMethod, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: usize, integration_limit: &[T; 2], point: &[T; NUM_VARS], n: u64) -> Result<T, &'static str>
{
    return Ok(T::zero());
}

