use crate::numerical_integration::iterative_integration::DEFAULT_TOTAL_ITERATIONS;
use crate::utils::error_codes::CalcError;

use crate::vector_field::line_integral;

///solves for the flux integral for parametrized curves in a vector field
///
/// NOTE: Returns a Result<_, CalcError>
/// Possible CalcError are:
/// CalcError::IterationsZero -> if the number of steps argument is zero
/// CalcError::IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
///
/// assume a vector field, V, and a curve, C
/// V is characterized in 2 dimensions
/// C is parameterized by a single variable, say, "t".
/// We also need a transformation to go t->x and t->y
/// The line integral limits are also based on this parameter t
///
/// [vector_field] is an array of 2 elements. Each element takes the curve position [x, y] and returns that axis' contribution
/// [transformations] is an array of 2 elements. The 0th element contains the transformation from t->x, and 1st element for t->y
/// [integration_limit] is the limit parameter 't' goes to
///
/// Example:
/// Assume we have a vector field (y, -x)
/// The curve is a unit circle, parameterized by (Cos(t), Sin(t)), such that t goes from 0->2*pi
/// ```
/// use multicalc::vector_field::flux_integral;
/// let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&(|args:&[f64; 2]|-> f64 { args[1] }), &(|args:&[f64; 2]|-> f64 { -args[0] })];
///
/// let transformation_matrix: [&dyn Fn(f64) -> f64; 2] = [&(|t:f64|->f64 { t.cos() }), &(|t:f64|->f64 { t.sin() })];
///
/// let integration_limit = [0.0, 6.28];
///
/// //flux integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of 0.0
/// let val = flux_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit).unwrap();
/// assert!(f64::abs(val + 0.0) < 0.01);
/// ```
pub fn get_2d(
    vector_field: &[&dyn Fn(&[f64; 2]) -> f64; 2],
    transformations: &[&dyn Fn(f64) -> f64; 2],
    integration_limit: &[f64; 2],
) -> Result<f64, CalcError> {
    get_2d_custom(
        vector_field,
        transformations,
        integration_limit,
        DEFAULT_TOTAL_ITERATIONS,
    )
}

///same as [get_2d()] but with the option to change the total iterations used, reserved for more advanced user
/// NOTE: Returns a Result<_, CalcError>
/// Possible CalcError are:
/// CalcError::IterationsZero -> if the number of steps is zero
/// CalcError::IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_2d_custom(
    vector_field: &[&dyn Fn(&[f64; 2]) -> f64; 2],
    transformations: &[&dyn Fn(f64) -> f64; 2],
    integration_limit: &[f64; 2],
    total_iterations: u64,
) -> Result<f64, CalcError> {
    Ok(
        line_integral::get_partial_2d(vector_field, transformations, integration_limit, total_iterations, 0)?
            - line_integral::get_partial_2d(vector_field, transformations, integration_limit, total_iterations, 1)?,
    )
}
