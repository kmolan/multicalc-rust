use num_complex::ComplexFloat;
use crate::utils::error_codes::ErrorCode;
use crate::core::numerical_integration::mode::DEFAULT_TOTAL_ITERATIONS;

///solves for the line integral for parametrized curves in a 2D vector field
/// 
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
/// 
/// assume a vector field, V, and a curve, C
/// V is characterized in 2 dimensions, Vx and Vy
/// C is parameterized by a single variable, say, "t".
/// We also need a transformation to go t->x and t->y
/// The line integral limits are also based on this parameter t
/// 
/// [vector_field] is an array of 2 elements. The 0th element has vector field's contribution to the x-axis based on x and y values. The 1st element does the same for y-axis
/// [transformations] is an array of 2 elements. The 0th element contains the transformation from t->x, and 1st element for t->y
/// [integration_limit] is the limit parameter 't' goes to
/// [steps] is the total number of steps that the integration is discretized into. Higher number gives more accuracy, but at the cost of more computation time
/// 
/// Example:
/// Assume we have a vector field (y, -x)
/// The curve is a unit circle, parameterized by (Cos(t), Sin(t)), such that t goes from 0->2*pi
/// ```
/// use multicalc::core::vector_field::line_integral;
/// let vector_field_matrix: [&dyn Fn(&f64, &f64) -> f64; 2] = [&(|_:&f64, y:&f64|-> f64 { *y }), &(|x:&f64, _:&f64|-> f64 { -x })];
///
/// let transformation_matrix: [&dyn Fn(&f64) -> f64; 2] = [&(|t:&f64|->f64 { t.cos() }), &(|t:&f64|->f64 { t.sin() })];
///
/// let integration_limit = [0.0, 6.28];
///
/// //line integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of -2.0*pi
/// let val = line_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit).unwrap();
/// assert!(f64::abs(val + 6.28) < 0.01);
/// ```
pub fn get_2d<T: ComplexFloat>(vector_field: &[&dyn Fn(&T, &T) -> T; 2], transformations: &[&dyn Fn(&T) -> T; 2], integration_limit: &[T; 2]) -> Result<T, ErrorCode>
{
    return get_2d_custom(vector_field, transformations, integration_limit, DEFAULT_TOTAL_ITERATIONS);
}


///same as [get_2d()] but with the option to change the total iterations used, reserved for more advanced user
/// The argument 'n' denotes the number of steps to be used. However, for [`mode::IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_2d_custom<T: ComplexFloat>(vector_field: &[&dyn Fn(&T, &T) -> T; 2], transformations: &[&dyn Fn(&T) -> T; 2], integration_limit: &[T; 2], total_iterations: u64) -> Result<T, ErrorCode>
{
    return Ok(get_partial_2d(vector_field, transformations, integration_limit, total_iterations, 0)?
            + get_partial_2d(vector_field, transformations, integration_limit, total_iterations, 1)?);
}


/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_partial_2d<T: ComplexFloat>(vector_field: &[&dyn Fn(&T, &T) -> T; 2], transformations: &[&dyn Fn(&T) -> T; 2], integration_limit: &[T; 2], max_iterations: u64, idx: usize) -> Result<T, ErrorCode>
{
    if max_iterations == 0
    {
        return Err(ErrorCode::NumberOfStepsCannotBeZero);
    }
    if integration_limit[0].abs() >= integration_limit[1].abs()
    {
        return Err(ErrorCode::IntegrationLimitsIllDefined);
    }

    let mut ans = T::zero();

    let mut cur_point = integration_limit[0];

    let delta = (integration_limit[1] - integration_limit[0])/(T::from(max_iterations).unwrap());

    //use the trapezoidal rule for line integrals
    //https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter25/section04.html
    for _ in 0..max_iterations
    {
        let coords = get_transformed_coordinates_2d(transformations, &cur_point, &delta);

        ans = ans + (coords[idx + 2] - coords[idx])*(vector_field[idx](&coords[2], &coords[3]) + vector_field[idx](&coords[0], &coords[1]))/(T::from(2.0).unwrap());

        cur_point = cur_point + delta;
    }

    return Ok(ans);
}


///same as [`get_2d`] but for parametrized curves in a 3D vector field
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_3d<T: ComplexFloat>(vector_field: &[&dyn Fn(&T, &T, &T) -> T; 3], transformations: &[&dyn Fn(&T) -> T; 3], integration_limit: &[T; 2]) -> Result<T, ErrorCode>
{
    return get_3d_custom(vector_field, transformations, integration_limit, DEFAULT_TOTAL_ITERATIONS);
}


///same as [get_3d()] but with the option to change the total iterations used, reserved for more advanced user
/// The argument 'n' denotes the number of steps to be used. However, for [`mode::IntegrationMethod::GaussLegendre`], it denotes the highest order of our equation
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_3d_custom<T: ComplexFloat>(vector_field: &[&dyn Fn(&T, &T, &T) -> T; 3], transformations: &[&dyn Fn(&T) -> T; 3], integration_limit: &[T; 2], total_iterations: u64) -> Result<T, ErrorCode>
{
    return Ok(get_partial_3d(vector_field, transformations, integration_limit, total_iterations, 0)?
            + get_partial_3d(vector_field, transformations, integration_limit, total_iterations, 1)?
            + get_partial_3d(vector_field, transformations, integration_limit, total_iterations, 2)?);
}

/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the number of steps is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_partial_3d<T: ComplexFloat>(vector_field: &[&dyn Fn(&T, &T, &T) -> T; 3], transformations: &[&dyn Fn(&T) -> T; 3], integration_limit: &[T; 2], steps: u64, idx: usize) -> Result<T, ErrorCode>
{
    if steps == 0
    {
        return Err(ErrorCode::NumberOfStepsCannotBeZero);
    }
    if integration_limit[0].abs() >= integration_limit[1].abs()
    {
        return Err(ErrorCode::IntegrationLimitsIllDefined);
    }

    let mut ans = T::zero();

    let mut cur_point = integration_limit[0];

    let delta = (integration_limit[1] - integration_limit[0])/(T::from(steps).unwrap());

    //use the trapezoidal rule for line integrals
    //https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter25/section04.html
    for _ in 0..steps
    {
        let coords = get_transformed_coordinates_3d(transformations, &cur_point, &delta);

        ans = ans + (coords[idx + 3] - coords[idx])*(vector_field[idx](&coords[3], &coords[4], &coords[5]) + vector_field[idx](&coords[0], &coords[1], &coords[2]))/(T::from(2.0).unwrap());

        cur_point = cur_point + delta;
    }

    return Ok(ans);
}




fn get_transformed_coordinates_2d<T: ComplexFloat>(transformations: &[&dyn Fn(&T) -> T; 2], cur_point: &T, delta: &T) -> [T; 4]
{
    let mut ans = [T::zero(); 4];

    ans[0] = transformations[0](cur_point); //x at t
    ans[1] = transformations[1](cur_point); //y at t

    ans[2] = transformations[0](&(*cur_point + *delta)); //x at t + delta
    ans[3] = transformations[1](&(*cur_point + *delta)); //y at t + delta

    return ans;
}


fn get_transformed_coordinates_3d<T: ComplexFloat>(transformations: &[&dyn Fn(&T) -> T; 3], cur_point: &T, delta: &T) -> [T; 6]
{
    let mut ans = [T::zero(); 6];

    ans[0] = transformations[0](cur_point); //x at t
    ans[1] = transformations[1](cur_point); //y at t
    ans[2] = transformations[1](cur_point); //z at t

    ans[3] = transformations[0](&(*cur_point + *delta)); //x at t + delta
    ans[4] = transformations[1](&(*cur_point + *delta)); //y at t + delta
    ans[5] = transformations[1](&(*cur_point + *delta)); //z at t + delta

    return ans;
}