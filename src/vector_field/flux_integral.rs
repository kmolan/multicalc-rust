use crate::numerical_integration::iterative_integration::DEFAULT_TOTAL_ITERATIONS;
use crate::utils::error_codes::CalcError;

use crate::vector_field::line_integral;

/// Computes the flux integral of a 2D vector field across a parametrized curve.
///
/// The curve is described by a parameter `t`: the transforms map `t` to each coordinate, and
/// the field is sampled at the resulting curve position. Uses the default iteration count;
/// see [`get_2d_custom`] to set it.
///
/// # Arguments
/// * `vector_field` - the two field components, each taking the curve position `[x, y]`.
/// * `transformations` - the two transforms mapping `t` to `x` and to `y`.
/// * `integration_limit` - the `[lower, upper]` range of the parameter `t`.
///
/// # Errors
/// [`CalcError::IntegrationLimitsIllDefined`] if the lower limit is not strictly less than the
/// upper limit.
///
/// # Examples
/// ```
/// use multicalc::vector_field::flux_integral;
///
/// // the field (y, -x) across the unit circle (cos t, sin t), for t in [0, 2*pi]
/// let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] =
///     [&(|args: &[f64; 2]| args[1]), &(|args: &[f64; 2]| -args[0])];
/// let transformation_matrix: [&dyn Fn(f64) -> f64; 2] =
///     [&(|t: f64| t.cos()), &(|t: f64| t.sin())];
///
/// let val = flux_integral::get_2d(&vector_field_matrix, &transformation_matrix, &[0.0, 6.28]).unwrap();
/// // the flux integral is 0
/// assert!(f64::abs(val) < 0.01);
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

/// Same as [`get_2d`] but with an explicit iteration count for finer control.
///
/// # Errors
/// [`CalcError::IterationsZero`] if `total_iterations` is zero, or
/// [`CalcError::IntegrationLimitsIllDefined`] if the lower limit is not strictly less than the
/// upper limit.
pub fn get_2d_custom(
    vector_field: &[&dyn Fn(&[f64; 2]) -> f64; 2],
    transformations: &[&dyn Fn(f64) -> f64; 2],
    integration_limit: &[f64; 2],
    total_iterations: u64,
) -> Result<f64, CalcError> {
    Ok(line_integral::get_partial_2d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        0,
    )? - line_integral::get_partial_2d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        1,
    )?)
}
