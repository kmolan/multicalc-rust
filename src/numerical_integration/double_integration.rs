use crate::numerical_integration::mode::IntegrationMethod;
use crate::numerical_integration::single_integration;
use num_complex::ComplexFloat;

#[cfg(feature = "std")]
use crate::utils::gl_table as gl_table;

/// Returns the total double integration value for a given function
/// Only ideal for single variable functions
/// 
/// assume we want to integrate 6*x. The function would be:
/// ```
///    let my_func = | args: &[f64; 1] | -> f64 
///    { 
///        return 6.0*args[0];
///    };
///
////// where args[0] = x. We also need to define the intervals around which we want to integrate.
////// Assuming we first we want to integrate over [0.0, 1.0] and then over [0.0, 3.0]
///
/// let integration_limits = [[0.0, 1.0], [0.0, 3.0]];
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::double_integration;
///
/// let val = double_integration::get_total(IntegrationMethod::Trapezoidal,   //<- The method for integration we want to use
///                                          &my_func,                        //<- our closure                 
///                                          &integration_limits,             //<- The integration interval needed                          
///                                          10);                             //<- number of steps
/// 
/// assert!(f64::abs(val - 27.0) < 0.00001);
///```
/// 
/// the above method example can also be extended to complex numbers:
/// ```
///    let my_func = | args: &[num_complex::Complex64; 1] | -> num_complex::Complex64 
///    { 
///        return 6.0*args[0];
///    };
///
////// where args[0] = x. Assuming we first we want to integrate over (0.0 + 0.0i) to (2.0 + 1.0i) twice
/// let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(2.0, 1.0)]];
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::double_integration;
///
/// let val = double_integration::get_total(IntegrationMethod::Trapezoidal,   //<- The method for integration we want to use
///                                          &my_func,                        //<- our closure                 
///                                          &integration_limits,             //<- The integration interval needed                          
///                                          10);                             //<- number of steps
/// 
/// assert!(num_complex::ComplexFloat::abs(val.re - 6.0) < 0.00001);
/// assert!(num_complex::ComplexFloat::abs(val.im - 33.0) < 0.00001);
///```
/// Note: The argument 'n' denotes the number of steps to be used. However, for [`IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// 
pub fn get_total<T: ComplexFloat, const NUM_VARS: usize>(integration_method: IntegrationMethod, func: &dyn Fn(&[T; NUM_VARS]) -> T, integration_limits: &[[T; 2]; 2], n: u64) -> T 
{
    let point = [integration_limits[0][1]; NUM_VARS];

    match integration_method
    {
        IntegrationMethod::Booles        => return get_booles(func, [0, 0], integration_limits, &point, n),
        IntegrationMethod::GaussLegendre => 
        {
            #[cfg(feature = "std")]
            return get_gauss_legendre(func, [0, 0], integration_limits, &point, n as usize);

            #[cfg(not(feature = "std"))]
            panic!("enable std context to use!");
        }
        IntegrationMethod::Simpsons      => return get_simpsons(func, [0, 0], integration_limits, &point, n),
        IntegrationMethod::Trapezoidal   => return get_trapezoidal(func, [0, 0], integration_limits, &point, n)
    }
}


/// Returns the partial double integration value for a given function
/// Can handle multivariable functions of any order or complexity
/// 
/// assume we want to partially integrate first for x then y for the equation 2.0*x + y*z. The function would be:
/// ```
///    let my_func = | args: &[f64; 3] | -> f64 
///    { 
///        return 2.0*args[0] + args[1]*args[2];
///    };
///
////// where args[0] = x, args[1] = y and args[2] = z. We also need to define the intervals around which we want to integrate.
////// Assuming we first we want to integrate over [0.0, 1.0] and then over [0.0, 1.0]
///
/// let integration_limits = [[0.0, 1.0], [0.0, 1.0]];
/// 
////// For partial integration to work, we also need to define the static values for the remaining variables. 
////// Assuming z = 3.0:
/// 
/// let point = [1.0, 2.0, 3.0];
/// 
////// Note above that the point vector has the same number of elements as the number of elements my_func expects. 
////// Note above that in the point vector, the indexes of the variables to integrate, 0 and 1, 
////// MUST have same value as those variables' integration interval's upper limit, which is 1.0 and 2.0 respectively 
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::double_integration;
///
/// let val = double_integration::get_partial(IntegrationMethod::Trapezoidal,  //<- The method for integration we want to use
///                                           &my_func,                        //<- our closure   
///                                           [0, 1],                          //<- index of variables we want to differentiate, in this case x and y              
///                                           &integration_limits,             //<- The integration interval needed
///                                           &point,                          //<- The final point with all x,y,z values                          
///                                           10);                             //<- number of steps
/// 
/// assert!(f64::abs(val - 2.50) < 0.00001);
///```
/// 
/// the above method example can also be extended to complex numbers:
/// ```
///    let my_func = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
///    { 
///        return 2.0*args[0] + args[1]*args[2];
///    };
///
///////where args[0] = x, args[1] = y and args[2] = z. Assuming we first we want to integrate over (0.0 + 0.0i) to (2.0 + 1.0i) twice
///let integration_limits = [[num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)], [num_complex::c64(0.0, 0.0), num_complex::c64(1.0, 1.0)]];
/// 
///////For partial integration to work, we also need to define the static values for the remaining variables. 
///////Assuming z = 3.0 + 0.5i: 
/// let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];
/// 
///////Note above that the point vector has the same number of elements as the number of elements my_func expects. 
///////Note above that in the point vector, the indexes of the variables to integrate, 0 and 1, 
///////MUST have same value as those variables' integration interval's upper limit, which is 1.0 + 1.0i and 2.0 + 0.0i respectively 
/// 
/// use multicalc::numerical_integration::mode::IntegrationMethod;
/// use multicalc::numerical_integration::double_integration;
///
/// let val = double_integration::get_partial(IntegrationMethod::Trapezoidal,  //<- The method for integration we want to use
///                                           &my_func,                        //<- our closure   
///                                           [0, 1],                          //<- index of variables we want to differentiate, in this case x and y              
///                                           &integration_limits,             //<- The integration interval needed
///                                           &point,                          //<- The final point with all x,y,z values                          
///                                           10);                             //<- number of steps
/// 
/// assert!(num_complex::ComplexFloat::abs(val.re + 5.5) < 0.00001);
/// assert!(num_complex::ComplexFloat::abs(val.im - 4.5) < 0.00001);
///```
/// Note: The argument 'n' denotes the number of steps to be used. However, for [`IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// 
pub fn get_partial<T: ComplexFloat, const NUM_VARS: usize>(integration_method: IntegrationMethod, func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS], n: u64) -> T 
{
    match integration_method
    {
        IntegrationMethod::Booles        => return get_booles(func, idx_to_integrate, integration_limits, point, n),
        IntegrationMethod::GaussLegendre => 
        {
            #[cfg(feature = "std")]
            return get_gauss_legendre(func, idx_to_integrate, integration_limits, point, n as usize);

            #[cfg(not(feature = "std"))]
            panic!("enable std context to use!");
        }
        IntegrationMethod::Simpsons      => return get_simpsons(func, idx_to_integrate, integration_limits, point, n),
        IntegrationMethod::Trapezoidal   => return get_trapezoidal(func, idx_to_integrate, integration_limits, point, n)
    }
}

fn get_booles<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS], steps: u64) -> T
{
    let mut current_vec = *point;
    current_vec[idx_to_integrate[0]] = integration_limits[0][0];

    let mut ans = T::from(7.0).unwrap()*single_integration::get_partial(IntegrationMethod::Booles, func, idx_to_integrate[1], &integration_limits[1], &current_vec, steps);

    let delta = (integration_limits[0][1] - integration_limits[0][0])/(T::from(steps).unwrap());
    let mut multiplier = T::from(32.0).unwrap();

    for iter in 0..steps-1
    {
        current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
        ans = ans + multiplier*single_integration::get_partial(IntegrationMethod::Booles, func, idx_to_integrate[1], &integration_limits[1], &current_vec, steps);
        
        if (iter + 2) % 2 != 0
        {
            multiplier = T::from(32.0).unwrap();
        }
        else if (iter + 2) % 4 == 0
        {
            multiplier = T::from(14.0).unwrap();
        }
        else
        {
            multiplier = T::from(12.0).unwrap();
        }
    }

    current_vec[idx_to_integrate[0]] = integration_limits[0][1];

    ans = ans + T::from(7.0).unwrap()*single_integration::get_partial(IntegrationMethod::Booles, func, idx_to_integrate[1], &integration_limits[1], &current_vec, steps);

    return T::from(2.0).unwrap()*delta*ans/T::from(45.0).unwrap();
}


#[cfg(feature = "std")]
fn get_gauss_legendre<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS], order: usize) -> T
{
    let mut ans = T::zero();
    let abcsissa_coeff = (integration_limits[0][1] - integration_limits[0][0])/T::from(2.0).unwrap();
    let intercept = (integration_limits[0][1] + integration_limits[0][0])/T::from(2.0).unwrap();

    let (weight, abcsissa) = gl_table::get_gl_weights_and_abscissae(order);

    let mut args = *point;

    for iter in 0..order
    {
        args[idx_to_integrate[0]] = abcsissa_coeff*T::from(abcsissa[iter]).unwrap() + intercept;

        ans = ans + T::from(weight[iter]).unwrap()*single_integration::get_partial(IntegrationMethod::GaussLegendre, func, idx_to_integrate[1], &integration_limits[1], point, order as u64)
    }

    return abcsissa_coeff*ans;
}

fn get_simpsons<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS], steps: u64) -> T
{
    let mut current_vec = *point;
    current_vec[idx_to_integrate[0]] = integration_limits[0][0];

    let mut ans = single_integration::get_partial(IntegrationMethod::Simpsons, func, idx_to_integrate[1], &integration_limits[1], &current_vec, steps);
    let delta = (integration_limits[0][1] - integration_limits[0][0])/(T::from(steps).unwrap());

    let mut multiplier = T::from(3.0).unwrap();

    for iter in 0..steps-1
    {
        current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
        ans = ans + multiplier*single_integration::get_partial(IntegrationMethod::Simpsons, func, idx_to_integrate[1], &integration_limits[1], &current_vec, steps);        

        if (iter + 2) % 3 == 0
        {
            multiplier = T::from(2.0).unwrap();
        }
        else
        {
            multiplier = T::from(3.0).unwrap();
        }
    }

    current_vec[idx_to_integrate[0]] = integration_limits[0][1];

    ans = ans + single_integration::get_partial(IntegrationMethod::Simpsons, func, idx_to_integrate[1], &integration_limits[1], &current_vec, steps);

    return T::from(3.0).unwrap()*delta*ans/T::from(8.0).unwrap();
}

fn get_trapezoidal<T: ComplexFloat, const NUM_VARS: usize>(func: &dyn Fn(&[T; NUM_VARS]) -> T, idx_to_integrate: [usize; 2], integration_limits: &[[T; 2]; 2], point: &[T; NUM_VARS], steps: u64) -> T
{
    let mut current_vec = *point;
    current_vec[idx_to_integrate[0]] = integration_limits[0][0];

    let mut ans = single_integration::get_partial(IntegrationMethod::Trapezoidal, func, idx_to_integrate[1], &integration_limits[1], &current_vec, steps);

    let delta = (integration_limits[0][1] - integration_limits[0][0])/(T::from(steps).unwrap());

    for _ in 0..steps-1
    {
        current_vec[idx_to_integrate[0]] = current_vec[idx_to_integrate[0]] + delta;
        ans = ans + T::from(2.0).unwrap()*single_integration::get_partial(IntegrationMethod::Trapezoidal, func, idx_to_integrate[1], &integration_limits[1], &current_vec, steps);  
    }

    current_vec[idx_to_integrate[0]] = integration_limits[0][1];

    ans = ans + single_integration::get_partial(IntegrationMethod::Trapezoidal, func, idx_to_integrate[1], &integration_limits[1], &current_vec, steps);

    return T::from(0.5).unwrap()*delta*ans;
}