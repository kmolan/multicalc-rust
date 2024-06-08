use crate::numerical_derivative::mode as mode;
use crate::numerical_derivative::single_derivative;

///solves for the divegerence of a 3D vector field around a given point
/// 
/// assume a vector field, V
/// V is characterized in 3 dimensions: Vx, Vy and Vz
/// The divergence is then defined as dVx/dx + dVy/dy + dVz/dz
/// 
/// Example:
/// Assume we have a vector field (y, -x, 2*z)
/// ```
/// use multicalc::vector_field::divergence;
/// let vector_field_matrix: [Box<dyn Fn(&Vec<f64>) -> f64>; 3] = [Box::new(|args: &Vec<f64>|-> f64 { args[1] }), Box::new(|args: &Vec<f64>|-> f64 { -args[0]}), Box::new(|args: &Vec<f64>|-> f64 { 2.0*args[2]})];
///
/// let point = vec![0.0, 1.0, 3.0]; //the point of interest
///
/// //diverge known to be 2.0 
/// let val = divergence::get_3d(&vector_field_matrix, &point);
/// assert!(f64::abs(val - 2.00) < 0.00001);
/// ```
pub fn get_3d(vector_field: &[Box<dyn Fn(&Vec<f64>) -> f64>; 3], point: &Vec<f64>) -> f64
{
    return get_3d_custom(vector_field, point, 0.00001, mode::DiffMode::CentralFixedStep);
}

///same as [get_3d()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_3d_custom(vector_field: &[Box<dyn Fn(&Vec<f64>) -> f64>; 3], point: &Vec<f64>, step_size: f64, mode: mode::DiffMode) -> f64
{
    return single_derivative::get_partial_custom(vector_field[0].as_ref(), 0, point, step_size, mode)
        +  single_derivative::get_partial_custom(vector_field[1].as_ref(), 1, point, step_size, mode)
        +  single_derivative::get_partial_custom(vector_field[2].as_ref(), 2, point, step_size, mode);
}


///solves for the divegerence of a 2D vector field around a given point
/// 
/// assume a vector field, V
/// V is characterized in 3 dimensions: Vx and Vy
/// The divergence is then defined as dVx/dx + dVy/dy
/// 
/// Example:
/// Assume we have a vector field (y, -x)
/// ```
/// use multicalc::vector_field::divergence;
/// let vector_field_matrix: [Box<dyn Fn(&Vec<f64>) -> f64>; 2] = [Box::new(|args: &Vec<f64>|-> f64 { args[1] }), Box::new(|args: &Vec<f64>|-> f64 { -args[0]})];
///
/// let point = vec![0.0, 1.0]; //the point of interest
///
/// //diverge known to be 0.0 
/// let val = divergence::get_2d(&vector_field_matrix, &point);
/// assert!(f64::abs(val - 0.00) < 0.00001);
/// ```
pub fn get_2d(vector_field: &[Box<dyn Fn(&Vec<f64>) -> f64>; 2], point: &Vec<f64>) -> f64
{
    return get_2d_custom(vector_field, point, 0.00001, mode::DiffMode::CentralFixedStep);
}

///same as [get_2d()] but with the option to change the differentiation mode used, reserved for more advanced users
pub fn get_2d_custom(vector_field: &[Box<dyn Fn(&Vec<f64>) -> f64>; 2], point: &Vec<f64>, step_size: f64, mode: mode::DiffMode) -> f64
{
    return single_derivative::get_partial_custom(vector_field[0].as_ref(), 0, point, step_size, mode)
        +  single_derivative::get_partial_custom(vector_field[1].as_ref(), 1, point, step_size, mode);
}