use crate::numerical_derivative::double_derivative;
use crate::numerical_derivative::mode as mode;
use num_complex::ComplexFloat;

/// Returns the triple total derivative value for a given function
/// Only ideal for single variable functions
/// 
/// assume we want to differentiate x^4 . the function would be:
/// ```
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return args[0].powf(4.0);
///    };
/// 
//// where args[0] = x
///
//// We also need to define the point at which we want to differentiate. Assuming our point x = 1.0
//// if we then want to differentiate this function over x with a step size of 0.001, we would use:
///
/// use multicalc::numerical_derivative::triple_derivative;
/// 
/// let val = triple_derivative::get_total(&my_func,      //<- our closure                                           
///                                         1.0,          //<- point around which we want to differentiate
///                                         0.001);       //<- required step size
/// 
/// assert!(f64::abs(val - 24.0) < 0.00001);
/// ```
/// 
/// the above example can also be extended to complex numbers
///```
///    let my_func = | args: &Vec<num_complex::Complex64> | -> num_complex::Complex64 
///    { 
///        return args[0].powf(4.0);
///    };
///
//// where args[0] = x
/// 
/// //point of interest is x = 1.0 + 4.0i
/// let point = num_complex::c64(1.0, 4.0);
/// 
/// use multicalc::numerical_derivative::triple_derivative;
///
/// let val = triple_derivative::get_total(&my_func,  //<- our closure                                          
///                                        point,     //<- point around which we want to differentiate
///                                        0.001);    //<- required step size
/// 
/// let expected_val = 24.0*point;
/// assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.0001);
/// assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.0001);
///``` 
/// 
pub fn get_total<T: ComplexFloat>(func: &dyn Fn(&Vec<T>) -> T, point: T, step: f64) -> T
{
    return get_total_custom(func, point, step, mode::DiffMode::CentralFixedStep);
}


///same as [get_total()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_total_custom<T: ComplexFloat>(func: &dyn Fn(&Vec<T>) -> T, point: T, step: f64, mode: mode::DiffMode) -> T
{
    let vec_point = vec![point];

    match mode
    {
        mode::DiffMode::ForwardFixedStep => return get_forward_difference(func, &[0, 0, 0], &vec_point, step),
        mode::DiffMode::BackwardFixedStep => return get_backward_difference(func, &[0, 0, 0], &vec_point, step),
        mode::DiffMode::CentralFixedStep => return get_central_difference(func, &[0, 0, 0], &vec_point, step) 
    }
}

/// Returns the partial triple derivative value for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// assume we want to differentiate x^3 * y^3 * z^3 . the function would be:
/// ```
///    let my_func = | args: &Vec<f64> | -> f64 
///    { 
///        return args[0].powf(3.0)*args[1].powf(3.0)*args[2].powf(3.0);
///    };
/// 
//// where args[0] = x and args[1] = y. Also, we know our function must accept 2 arguments.
///
//// We also need to define the point at which we want to differentiate. Assuming our point is (1.0, 2.0, 3.0)
///
/// let point = vec![1.0, 2.0, 3.0];
///
//// For triple differentiation, we can choose which variables we want to differentiate over
//// For a total triple differentiation over x, idx = [0, 0, 0] (this is same as calling get_total() )
//// For a partial mixed differentiation, say first over x and then two times over y, idx = [0, 1, 1] since 'x' is the 0th, and 'y' the 1st in our function
///
//// For the partial mixed differentiation with a step size of 0.001, we would use:
///
/// use multicalc::numerical_derivative::triple_derivative;
/// 
/// let val = triple_derivative::get_partial(&my_func,    //<- our closure                
///                                          &[0, 1, 1],  //<- idx, index of variables we want to differentiate                            
///                                          &point,      //<- point around which we want to differentiate
///                                          0.001);      //<- required step size
/// 
/// assert!(f64::abs(val - 972.0) < 0.001);
/// ```
/// the above example can also be extended to complex numbers
///```
///    let my_func = | args: &Vec<num_complex::Complex64> | -> num_complex::Complex64 
///    { 
///        return args[0].powf(3.0)*args[1].powf(3.0)*args[2].powf(3.0);
///    };
///
//// where args[0] = x
/// 
/// //point of interest is (x, y, z) = (1.0 + 4.0i, 2.0 + 2.5i, 3.0 + 0.0i)
/// let point = vec![num_complex::c64(1.0, 4.0), num_complex::c64(2.0, 2.5), num_complex::c64(3.0, 0.0)];
/// 
/// use multicalc::numerical_derivative::triple_derivative;
///
/// let val = triple_derivative::get_partial(&my_func,    //<- our closure                
///                                          &[0, 1, 1],  //<- idx, index of variables we want to differentiate                            
///                                          &point,      //<- point around which we want to differentiate
///                                          0.001);      //<- required step size
/// 
/// let expected_val = 18.0*point[0].powf(2.0)*point[1]*point[2].powf(3.0);
/// assert!(num_complex::ComplexFloat::abs(val.re - expected_val.re) < 0.01);
/// assert!(num_complex::ComplexFloat::abs(val.im - expected_val.im) < 0.01);
///``` 
/// 
pub fn get_partial<T: ComplexFloat>(func: &dyn Fn(&Vec<T>) -> T, idx_to_derivate: &[usize; 3], point: &Vec<T>, step: f64) -> T
{
    return get_partial_custom(func, idx_to_derivate, point, step, mode::DiffMode::CentralFixedStep);
}


///same as [get_partial()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_partial_custom<T: ComplexFloat>(func: &dyn Fn(&Vec<T>) -> T, idx_to_derivate: &[usize; 3], point: &Vec<T>, step: f64, mode: mode::DiffMode) -> T
{
    match mode
    {
        mode::DiffMode::ForwardFixedStep => return get_forward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::BackwardFixedStep => return get_backward_difference(func, idx_to_derivate, point, step),
        mode::DiffMode::CentralFixedStep => return get_central_difference(func, idx_to_derivate, point, step) 
    }
}

fn get_forward_difference<T: ComplexFloat>(func: &dyn Fn(&Vec<T>) -> T, idx_to_derivate: &[usize; 3], point: &Vec<T>, step: f64) -> T
{
    let f0 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], point, step, mode::DiffMode::ForwardFixedStep);

    let mut f1_point = point.clone();
    f1_point[idx_to_derivate[0]] = f1_point[idx_to_derivate[0]] + T::from(step).unwrap();
    let f1 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], &f1_point, step, mode::DiffMode::ForwardFixedStep);

    return (f1 - f0)/T::from(step).unwrap();    
}

fn get_backward_difference<T: ComplexFloat>(func: &dyn Fn(&Vec<T>) -> T, idx_to_derivate: &[usize; 3], point: &Vec<T>, step: f64) -> T
{
    let mut f0_point = point.clone();
    f0_point[idx_to_derivate[0]] = f0_point[idx_to_derivate[0]] - T::from(step).unwrap();
    let f0 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], &f0_point, step, mode::DiffMode::BackwardFixedStep);

    let f1 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], point, step, mode::DiffMode::BackwardFixedStep);

    return (f1 - f0)/T::from(step).unwrap();
}

fn get_central_difference<T: ComplexFloat>(func: &dyn Fn(&Vec<T>) -> T, idx_to_derivate: &[usize; 3], point: &Vec<T>, step: f64) -> T
{
    let mut f0_point = point.clone();
    f0_point[idx_to_derivate[0]] = f0_point[idx_to_derivate[0]] - T::from(step).unwrap();
    let f0 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], &f0_point, step, mode::DiffMode::CentralFixedStep);

    let mut f1_point = point.clone();
    f1_point[idx_to_derivate[0]] = f1_point[idx_to_derivate[0]] + T::from(step).unwrap();
    let f1 = double_derivative::get_partial_custom(func, &[idx_to_derivate[1], idx_to_derivate[2]], &f1_point, step, mode::DiffMode::CentralFixedStep);

    return (f1 - f0)/(T::from(2.0*step).unwrap());
}
