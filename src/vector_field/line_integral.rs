use crate::numerical_integration::iterative_integration::DEFAULT_TOTAL_ITERATIONS;
use crate::utils::error_codes::CalcError;

/// Builds the curve position [transformations[0](t), ..., transformations[N-1](t)].
fn curve_point<const N: usize>(transformations: &[&dyn Fn(f64) -> f64; N], t: f64) -> [f64; N] {
    let mut point = [0.0; N];
    for i in 0..N {
        point[i] = transformations[i](t);
    }
    point
}

/// Trapezoidal integration of the `idx`-th field component along the parametrized curve.
/// Generic over the dimension `N`, so the 2D and 3D paths share one body.
fn get_partial<const N: usize>(
    vector_field: &[&dyn Fn(&[f64; N]) -> f64; N],
    transformations: &[&dyn Fn(f64) -> f64; N],
    integration_limit: &[f64; 2],
    total_iterations: u64,
    idx: usize,
) -> Result<f64, CalcError> {
    if total_iterations == 0 {
        return Err(CalcError::IterationsZero);
    }
    if !(integration_limit[0] < integration_limit[1]) {
        return Err(CalcError::IntegrationLimitsIllDefined);
    }

    let delta = (integration_limit[1] - integration_limit[0]) / total_iterations as f64;
    let mut t = integration_limit[0];
    let mut ans = 0.0;

    //use the trapezoidal rule for line integrals, caching the shared endpoint so each
    //curve point and field value is evaluated once per node rather than twice
    //https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter25/section04.html
    let mut left = curve_point(transformations, t);
    let mut left_value = vector_field[idx](&left);

    for _ in 0..total_iterations {
        let right = curve_point(transformations, t + delta);
        let right_value = vector_field[idx](&right);

        ans += (right[idx] - left[idx]) * (left_value + right_value) / 2.0;

        t += delta;
        left = right;
        left_value = right_value;
    }

    Ok(ans)
}

///solves for the line integral for parametrized curves in a 2D vector field
///
/// NOTE: Returns a Result<_, CalcError>
/// Possible CalcError are:
/// CalcError::IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
///
/// assume a vector field, V, and a curve, C
/// V is characterized in 2 dimensions, Vx and Vy
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
/// use multicalc::vector_field::line_integral;
/// let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&(|args:&[f64; 2]|-> f64 { args[1] }), &(|args:&[f64; 2]|-> f64 { -args[0] })];
///
/// let transformation_matrix: [&dyn Fn(f64) -> f64; 2] = [&(|t:f64|->f64 { t.cos() }), &(|t:f64|->f64 { t.sin() })];
///
/// let integration_limit = [0.0, 6.28];
///
/// //line integral of a unit circle curve on our vector field from 0 to 2*pi, expect an answer of -2.0*pi
/// let val = line_integral::get_2d(&vector_field_matrix, &transformation_matrix, &integration_limit).unwrap();
/// assert!(f64::abs(val + 6.28) < 0.01);
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
        get_partial_2d(vector_field, transformations, integration_limit, total_iterations, 0)?
            + get_partial_2d(vector_field, transformations, integration_limit, total_iterations, 1)?,
    )
}

/// NOTE: Returns a Result<_, CalcError>
/// Possible CalcError are:
/// CalcError::IterationsZero -> if the number of steps is zero
/// CalcError::IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_partial_2d(
    vector_field: &[&dyn Fn(&[f64; 2]) -> f64; 2],
    transformations: &[&dyn Fn(f64) -> f64; 2],
    integration_limit: &[f64; 2],
    total_iterations: u64,
    idx: usize,
) -> Result<f64, CalcError> {
    get_partial(vector_field, transformations, integration_limit, total_iterations, idx)
}

///same as [`get_2d`] but for parametrized curves in a 3D vector field
/// NOTE: Returns a Result<_, CalcError>
/// Possible CalcError are:
/// CalcError::IterationsZero -> if the number of steps is zero
/// CalcError::IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_3d(
    vector_field: &[&dyn Fn(&[f64; 3]) -> f64; 3],
    transformations: &[&dyn Fn(f64) -> f64; 3],
    integration_limit: &[f64; 2],
) -> Result<f64, CalcError> {
    get_3d_custom(
        vector_field,
        transformations,
        integration_limit,
        DEFAULT_TOTAL_ITERATIONS,
    )
}

///same as [get_3d()] but with the option to change the total iterations used, reserved for more advanced user
/// NOTE: Returns a Result<_, CalcError>
/// Possible CalcError are:
/// CalcError::IterationsZero -> if the number of steps is zero
/// CalcError::IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_3d_custom(
    vector_field: &[&dyn Fn(&[f64; 3]) -> f64; 3],
    transformations: &[&dyn Fn(f64) -> f64; 3],
    integration_limit: &[f64; 2],
    total_iterations: u64,
) -> Result<f64, CalcError> {
    Ok(
        get_partial_3d(vector_field, transformations, integration_limit, total_iterations, 0)?
            + get_partial_3d(vector_field, transformations, integration_limit, total_iterations, 1)?
            + get_partial_3d(vector_field, transformations, integration_limit, total_iterations, 2)?,
    )
}

/// NOTE: Returns a Result<_, CalcError>
/// Possible CalcError are:
/// CalcError::IterationsZero -> if the number of steps is zero
/// CalcError::IntegrationLimitsIllDefined -> if the integration lower limit is not strictly lesser than the integration upper limit
pub fn get_partial_3d(
    vector_field: &[&dyn Fn(&[f64; 3]) -> f64; 3],
    transformations: &[&dyn Fn(f64) -> f64; 3],
    integration_limit: &[f64; 2],
    total_iterations: u64,
    idx: usize,
) -> Result<f64, CalcError> {
    get_partial(vector_field, transformations, integration_limit, total_iterations, idx)
}
