use crate::numeric::Numeric;
use crate::numerical_integration::iterative_integration::DEFAULT_TOTAL_ITERATIONS;
use crate::utils::error_codes::CalcError;

/// Builds the curve position [transformations[0](t), ..., transformations[N-1](t)].
fn curve_point<T: Numeric, const N: usize>(transformations: &[&dyn Fn(T) -> T; N], t: T) -> [T; N] {
    let mut point = [T::ZERO; N];
    for i in 0..N {
        point[i] = transformations[i](t);
    }
    point
}

/// Trapezoidal integration of the `idx`-th field component along the parametrized curve.
/// Generic over the dimension `N`, so the 2D and 3D paths share one body.
fn get_partial<T: Numeric, const N: usize>(
    vector_field: &[&dyn Fn(&[T; N]) -> T; N],
    transformations: &[&dyn Fn(T) -> T; N],
    integration_limit: &[T; 2],
    total_iterations: u64,
    idx: usize,
) -> Result<T, CalcError> {
    if total_iterations == 0 {
        return Err(CalcError::IterationsZero);
    }
    // rejects NaN, equal, and reversed limits (partial_cmp is None for NaN)
    if !matches!(
        integration_limit[0].partial_cmp(&integration_limit[1]),
        Some(core::cmp::Ordering::Less)
    ) {
        return Err(CalcError::IntegrationLimitsIllDefined);
    }

    let delta = (integration_limit[1] - integration_limit[0]) / T::from_u64(total_iterations);
    let mut t = integration_limit[0];
    let mut ans = T::ZERO;

    //use the trapezoidal rule for line integrals, caching the shared endpoint so each
    //curve point and field value is evaluated once per node rather than twice
    //https://ocw.mit.edu/ans7870/18/18.013a/textbook/HTML/chapter25/section04.html
    let mut left = curve_point(transformations, t);
    let mut left_value = vector_field[idx](&left);

    for _ in 0..total_iterations {
        let right = curve_point(transformations, t + delta);
        let right_value = vector_field[idx](&right);

        ans += (right[idx] - left[idx]) * (left_value + right_value) / T::TWO;

        t += delta;
        left = right;
        left_value = right_value;
    }

    Ok(ans)
}

/// Computes the line integral of a 2D vector field along a parametrized curve.
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
/// use multicalc::vector_field::line_integral;
///
/// // the field (y, -x) along the unit circle (cos t, sin t), for t in [0, 2*pi]
/// let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] =
///     [&(|args: &[f64; 2]| args[1]), &(|args: &[f64; 2]| -args[0])];
/// let transformation_matrix: [&dyn Fn(f64) -> f64; 2] =
///     [&(|t: f64| t.cos()), &(|t: f64| t.sin())];
///
/// let val = line_integral::get_2d(&vector_field_matrix, &transformation_matrix, &[0.0, 6.28]).unwrap();
/// // the line integral is -2*pi
/// assert!(f64::abs(val + 6.28) < 0.01);
/// ```
pub fn get_2d<T: Numeric>(
    vector_field: &[&dyn Fn(&[T; 2]) -> T; 2],
    transformations: &[&dyn Fn(T) -> T; 2],
    integration_limit: &[T; 2],
) -> Result<T, CalcError> {
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
pub fn get_2d_custom<T: Numeric>(
    vector_field: &[&dyn Fn(&[T; 2]) -> T; 2],
    transformations: &[&dyn Fn(T) -> T; 2],
    integration_limit: &[T; 2],
    total_iterations: u64,
) -> Result<T, CalcError> {
    Ok(get_partial_2d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        0,
    )? + get_partial_2d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        1,
    )?)
}

/// Line integral of a single field component (`idx`) along the 2D curve. Used by both
/// [`get_2d_custom`] and the flux integral.
///
/// # Errors
/// [`CalcError::IterationsZero`] if `total_iterations` is zero, or
/// [`CalcError::IntegrationLimitsIllDefined`] if the lower limit is not strictly less than the
/// upper limit.
pub fn get_partial_2d<T: Numeric>(
    vector_field: &[&dyn Fn(&[T; 2]) -> T; 2],
    transformations: &[&dyn Fn(T) -> T; 2],
    integration_limit: &[T; 2],
    total_iterations: u64,
    idx: usize,
) -> Result<T, CalcError> {
    get_partial(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        idx,
    )
}

/// Same as [`get_2d`] but for a parametrized curve in a 3D vector field. Uses the default
/// iteration count; see [`get_3d_custom`] to set it.
///
/// # Errors
/// [`CalcError::IntegrationLimitsIllDefined`] if the lower limit is not strictly less than the
/// upper limit.
pub fn get_3d<T: Numeric>(
    vector_field: &[&dyn Fn(&[T; 3]) -> T; 3],
    transformations: &[&dyn Fn(T) -> T; 3],
    integration_limit: &[T; 2],
) -> Result<T, CalcError> {
    get_3d_custom(
        vector_field,
        transformations,
        integration_limit,
        DEFAULT_TOTAL_ITERATIONS,
    )
}

/// Same as [`get_3d`] but with an explicit iteration count for finer control.
///
/// # Errors
/// [`CalcError::IterationsZero`] if `total_iterations` is zero, or
/// [`CalcError::IntegrationLimitsIllDefined`] if the lower limit is not strictly less than the
/// upper limit.
pub fn get_3d_custom<T: Numeric>(
    vector_field: &[&dyn Fn(&[T; 3]) -> T; 3],
    transformations: &[&dyn Fn(T) -> T; 3],
    integration_limit: &[T; 2],
    total_iterations: u64,
) -> Result<T, CalcError> {
    Ok(get_partial_3d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        0,
    )? + get_partial_3d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        1,
    )? + get_partial_3d(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        2,
    )?)
}

/// Line integral of a single field component (`idx`) along the 3D curve. Used by both
/// [`get_3d_custom`] and the flux integral.
///
/// # Errors
/// [`CalcError::IterationsZero`] if `total_iterations` is zero, or
/// [`CalcError::IntegrationLimitsIllDefined`] if the lower limit is not strictly less than the
/// upper limit.
pub fn get_partial_3d<T: Numeric>(
    vector_field: &[&dyn Fn(&[T; 3]) -> T; 3],
    transformations: &[&dyn Fn(T) -> T; 3],
    integration_limit: &[T; 2],
    total_iterations: u64,
    idx: usize,
) -> Result<T, CalcError> {
    get_partial(
        vector_field,
        transformations,
        integration_limit,
        total_iterations,
        idx,
    )
}
