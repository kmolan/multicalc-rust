use crate::numerical_integration::mode::DEFAULT_TOTAL_ITERATIONS;
use crate::utils::error_codes::ErrorCode;
use crate::vector_field::line_integral;
use num_complex::ComplexFloat;


///solves for the flux integral for parametrized curves in a vector field
/// 
/// NOTE: Returns a Result<T, ErrorCode>
/// Possible ErrorCode are:
/// NumberOfStepsCannotBeZero -> if the number of steps argument is zero
/// IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
/// 
/// assume a vector field, V, and a curve, C
/// V is characterized in 2 dimensions
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
/// use multicalc::vector_field::flux_integral;
/// let vector_field_matrix: [&dyn Fn(&f64, &f64) -> f64; 2] = [&(|_:&f64, y:&f64|-> f64 { *y }), &(|x:&f64, _:&f64|-> f64 { -x })];
///
/// let transformation_matrix: [&dyn Fn(&f64) -> f64; 2] = [&(|t:&f64|->f64 { t.cos() }), &(|t:&f64|->f64 { t.sin() })];
///
/// let integration_limit = [0.0, 6.28];
///
/// //flux integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of 0.0
/// let val = flux_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit).unwrap();
/// assert!(f64::abs(val + 0.0) < 0.01);
/// ```
pub fn get_2d<T: ComplexFloat>(vector_field: &[&dyn Fn(&T, &T) -> T; 2], transformations: &[&dyn Fn(&T) -> T; 2], integration_limit: &[T; 2]) -> Result<T, ErrorCode>
{
    return Ok(line_integral::get_partial_2d(vector_field, transformations, integration_limit, DEFAULT_TOTAL_ITERATIONS, 0)?
            - line_integral::get_partial_2d(vector_field, transformations, integration_limit, DEFAULT_TOTAL_ITERATIONS, 1)?);
}

pub fn get_2d_custom<T: ComplexFloat>(vector_field: &[&dyn Fn(&T, &T) -> T; 2], transformations: &[&dyn Fn(&T) -> T; 2], integration_limit: &[T; 2], total_iterations: u64) -> Result<T, ErrorCode>
{
    return Ok(line_integral::get_partial_2d(vector_field, transformations, integration_limit, total_iterations, 0)?
            - line_integral::get_partial_2d(vector_field, transformations, integration_limit, total_iterations, 1)?);
}