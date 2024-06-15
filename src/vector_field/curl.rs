use crate::numerical_derivative::mode as mode;
use crate::numerical_derivative::single_derivative;
use num_complex::ComplexFloat;


///solves for the curl of a 3D vector field around a given point
/// 
/// assume a vector field, V
/// V is characterized in 3 dimensions: Vx, Vy and Vz
/// The curl is then defined as Cx, Cy and Cz, where:
/// Cx = dVz/dy - dVy/dVz
/// Cy = dVx/dz - dVz/dVx
/// Cz = dVy/dx - dVx/dVy
/// 
/// Example:
/// Assume we have a vector field (y, -x, 2*z)
/// ```
/// use multicalc::vector_field::curl;
/// 
/// let vf_x = | args: &[f64; 3] | -> f64 
/// { 
///     return args[1];
/// };
///
/// //y-component
/// let vf_y = | args: &[f64; 3] | -> f64 
/// { 
///     return -args[0];
/// };
///
/// //z-component
/// let vf_z = | args: &[f64; 3] | -> f64 
/// { 
///     return 2.0*args[2];
/// };
/// 
/// let vector_field_matrix: [&dyn Fn(&[f64; 3]) -> f64; 3] = [&vf_x, &vf_y, &vf_z];
///
/// let point = [0.0, 1.0, 3.0]; //the point of interest
///
/// //curl is known to be (0.0, 0.0, -2.0)
/// let val = curl::get_3d(&vector_field_matrix, &point);
/// assert!(f64::abs(val[0] - 0.00) < 0.00001);
/// assert!(f64::abs(val[1] - 0.00) < 0.00001);
/// assert!(f64::abs(val[2] + 2.00) < 0.00001);
/// ```
pub fn get_3d<T: ComplexFloat, const NUM_VARS: usize>(vector_field: &[&dyn Fn(&[T; NUM_VARS]) -> T; 3], point: &[T; NUM_VARS]) -> [T; 3]
{
    return get_3d_custom(vector_field, point, 0.00001, mode::DiffMode::CentralFixedStep);
}

///same as [get_3d()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_3d_custom<T: ComplexFloat, const NUM_VARS: usize>(vector_field: &[&dyn Fn(&[T; NUM_VARS]) -> T; 3], point: &[T; NUM_VARS], step_size: f64, mode: mode::DiffMode) -> [T; 3]
{
    let mut ans = [T::zero(); 3];

    ans[0] = single_derivative::get_partial_custom(vector_field[2], 1, point, step_size, mode) - single_derivative::get_partial_custom(vector_field[1], 2, point, step_size, mode);
    ans[1] = single_derivative::get_partial_custom(vector_field[0], 2, point, step_size, mode) - single_derivative::get_partial_custom(vector_field[2], 0, point, step_size, mode);
    ans[2] = single_derivative::get_partial_custom(vector_field[1], 0, point, step_size, mode) - single_derivative::get_partial_custom(vector_field[0], 1, point, step_size, mode);

    return ans;
}


///solves for the curl of a 2D vector field around a given point
/// 
/// assume a vector field, V
/// V is characterized in 3 dimensions: Vx and Vy
/// The curl is then defined as dVy/dx - dVx/dVy
/// Example:
/// Assume we have a vector field (2*x*y, 3*cos(y))
/// ```
/// use multicalc::vector_field::curl;
/// 
/// //x-component
/// let vf_x = | args: &[f64; 2] | -> f64 
/// { 
///     return 2.0*args[0]*args[1];
/// };
///
/// //y-component
/// let vf_y = | args: &[f64; 2] | -> f64 
/// { 
///     return 3.0*args[1].cos()
/// };
/// 
/// let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&vf_x, &vf_y];
///
/// let point = [1.0, 3.14]; //the point of interest
///
/// //curl is known to be -2.0
/// let val = curl::get_2d(&vector_field_matrix, &point);
/// assert!(f64::abs(val + 2.0) < 0.00001);
/// ```
pub fn get_2d<T: ComplexFloat, const NUM_VARS: usize>(vector_field: &[&dyn Fn(&[T; NUM_VARS]) -> T; 2], point: &[T; NUM_VARS]) -> T
{
    return get_2d_custom(vector_field, point, 0.00001, mode::DiffMode::CentralFixedStep);
}

///same as [get_2d()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_2d_custom<T: ComplexFloat, const NUM_VARS: usize>(vector_field: &[&dyn Fn(&[T; NUM_VARS]) -> T; 2], point: &[T; NUM_VARS], step_size: f64, mode: mode::DiffMode) -> T
{
    return single_derivative::get_partial_custom(vector_field[1], 0, point, step_size, mode)
         - single_derivative::get_partial_custom(vector_field[0], 1, point, step_size, mode);
}