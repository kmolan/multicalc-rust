use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::utils::error_codes::CalcError;

/// Computes the curl of a 3D vector field at a point.
///
/// For a field `V = (Vx, Vy, Vz)`, the curl is the vector
/// `(dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy)`.
///
/// # Arguments
/// * `derivator` - the derivator used for the partial derivatives.
/// * `vector_field` - the three field components, each taking the point `[x, y, z]`.
/// * `point` - the point at which the curl is evaluated.
///
/// # Errors
/// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
///
/// # Examples
/// ```
/// use multicalc::vector_field::curl;
/// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
///
/// // the field (y, -x, 2z)
/// let vf_x = |args: &[f64; 3]| args[1];
/// let vf_y = |args: &[f64; 3]| -args[0];
/// let vf_z = |args: &[f64; 3]| 2.0 * args[2];
/// let vector_field_matrix: [&dyn Fn(&[f64; 3]) -> f64; 3] = [&vf_x, &vf_y, &vf_z];
///
/// let derivator = FiniteDifferenceMulti::default();
/// let val = curl::get_3d(derivator, &vector_field_matrix, &[0.0, 1.0, 3.0]).unwrap();
/// // curl is known to be (0, 0, -2)
/// assert!(f64::abs(val[2] + 2.0) < 1e-5);
/// ```
pub fn get_3d<D: DerivatorMultiVariable, const NUM_VARS: usize>(
    derivator: D,
    vector_field: &[&dyn Fn(&[f64; NUM_VARS]) -> f64; 3],
    point: &[f64; NUM_VARS],
) -> Result<[f64; 3], CalcError> {
    let mut ans = [0.0; 3];

    ans[0] = derivator.get_single_partial(&vector_field[2], 1, point)?
        - derivator.get_single_partial(&vector_field[1], 2, point)?;
    ans[1] = derivator.get_single_partial(&vector_field[0], 2, point)?
        - derivator.get_single_partial(&vector_field[2], 0, point)?;
    ans[2] = derivator.get_single_partial(&vector_field[1], 0, point)?
        - derivator.get_single_partial(&vector_field[0], 1, point)?;

    Ok(ans)
}

/// Computes the (scalar) curl of a 2D vector field at a point.
///
/// For a field `V = (Vx, Vy)`, the curl is `dVy/dx - dVx/dy`.
///
/// # Arguments
/// * `derivator` - the derivator used for the partial derivatives.
/// * `vector_field` - the two field components, each taking the point `[x, y]`.
/// * `point` - the point at which the curl is evaluated.
///
/// # Errors
/// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
///
/// # Examples
/// ```
/// use multicalc::vector_field::curl;
/// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
///
/// // the field (2xy, 3cos(y))
/// let vf_x = |args: &[f64; 2]| 2.0 * args[0] * args[1];
/// let vf_y = |args: &[f64; 2]| 3.0 * args[1].cos();
/// let vector_field_matrix: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&vf_x, &vf_y];
///
/// let derivator = FiniteDifferenceMulti::default();
/// let val = curl::get_2d(derivator, &vector_field_matrix, &[1.0, 3.14]).unwrap();
/// // curl is known to be -2
/// assert!(f64::abs(val + 2.0) < 1e-5);
/// ```
pub fn get_2d<D: DerivatorMultiVariable, const NUM_VARS: usize>(
    derivator: D,
    vector_field: &[&dyn Fn(&[f64; NUM_VARS]) -> f64; 2],
    point: &[f64; NUM_VARS],
) -> Result<f64, CalcError> {
    Ok(derivator.get_single_partial(&vector_field[1], 0, point)?
        - derivator.get_single_partial(&vector_field[0], 1, point)?)
}
