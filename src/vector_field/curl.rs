
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
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
/// NOTE: Returns a Result<T, &'static str>
/// Possible &'static str are:
/// NumberOfStepsCannotBeZero -> if the derivative step size is zero
/// 
/// Example:
/// Assume we have a vector field (y, -x, 2*z)
/// ```
/// use multicalc::vector_field::curl;
/// use multicalc::numerical_derivative::finite_difference::*;
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
/// let derivator = MultiVariableSolver::default();
///
/// //curl is known to be (0.0, 0.0, -2.0)
/// let val = curl::get_3d(derivator, &vector_field_matrix, &point).unwrap();
/// assert!(f64::abs(val[0] - 0.00) < 0.00001);
/// assert!(f64::abs(val[1] - 0.00) < 0.00001);
/// assert!(f64::abs(val[2] + 2.00) < 0.00001);
/// ```
pub fn get_3d<T, D, const NUM_VARS: usize>(derivator: D, vector_field: &[&dyn Fn(&[T; NUM_VARS]) -> T; 3], point: &[T; NUM_VARS]) -> Result<[T; 3], &'static str>
where T: ComplexFloat, D: DerivatorMultiVariable
{
    let mut ans = [T::zero(); 3];

    ans[0] = derivator.get(1, vector_field[2], &[1], point)? - derivator.get(1, vector_field[1], &[2], point)?;
    ans[1] = derivator.get(1, vector_field[0], &[2], point)? - derivator.get(1, vector_field[2], &[0], point)?;
    ans[2] = derivator.get(1, vector_field[1], &[0], point)? - derivator.get(1, vector_field[0], &[1], point)?;

    return Ok(ans);
}


///solves for the curl of a 2D vector field around a given point
/// 
/// assume a vector field, V
/// V is characterized in 3 dimensions: Vx and Vy
/// The curl is then defined as dVy/dx - dVx/dVy
/// 
/// NOTE: Returns a Result<T, &'static str>
/// Possible &'static str are:
/// NumberOfStepsCannotBeZero -> if the derivative step size is zero
/// 
/// Example:
/// Assume we have a vector field (2*x*y, 3*cos(y))
/// ```
/// use multicalc::vector_field::curl;
/// use multicalc::numerical_derivative::finite_difference::*;
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
/// let derivator = MultiVariableSolver::default();
///
/// //curl is known to be -2.0
/// let val = curl::get_2d(derivator, &vector_field_matrix, &point).unwrap();
/// assert!(f64::abs(val + 2.0) < 0.00001);
/// ```
pub fn get_2d<T, D, const NUM_VARS: usize>(derivator: D, vector_field: &[&dyn Fn(&[T; NUM_VARS]) -> T; 2], point: &[T; NUM_VARS]) -> Result<T, &'static str>
where T: ComplexFloat, D: DerivatorMultiVariable
{
    return Ok(derivator.get(1, vector_field[1], &[0], point)? - derivator.get(1, vector_field[0], &[1], point)?);
}