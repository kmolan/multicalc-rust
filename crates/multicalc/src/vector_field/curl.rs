use crate::error::DiffError;
use crate::numerical_derivative::DerivatorMultiVariable;
use crate::scalar::Component;
use crate::scalar::{Numeric, VectorFn};

/// Computes the curl of a 3D vector field at a point.
///
/// For a field `V = (Vx, Vy, Vz)`, the curl is the vector
/// `(dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy)`.
///
/// # Arguments
/// * `derivator` - the derivator used for the partial derivatives.
/// * `vector_field` - the three field components as one vector-valued function of `[x, y, z]`.
/// * `point` - the point at which the curl is evaluated.
///
/// # Errors
/// [`DiffError::StepSizeZero`] if the derivator's step size is zero.
///
/// # Examples
/// ```
/// use multicalc::numerical_derivative::AutoDiffMulti;
/// use multicalc::scalar::constant;
/// use multicalc::scalar_fn_vec;
/// use multicalc::vector_field::curl_3d;
///
/// // the field (y, -x, 2z)
/// let vf = scalar_fn_vec!(|point: &[f64; 3]| [point[1], -point[0], constant(2.0) * point[2]]);
///
/// let derivator = AutoDiffMulti::default();
/// let point = [0.0, 1.0, 3.0];
/// let val = curl_3d(derivator, &vf, &point).unwrap();
/// // curl is known to be (0, 0, -2)
/// assert!(f64::abs(val[2] + 2.0) < 1e-12);
/// ```
pub fn curl_3d<D: DerivatorMultiVariable, F: VectorFn<NUM_VARS, 3>, const NUM_VARS: usize>(
    derivator: D,
    vector_field: &F,
    point: &[D::Scalar; NUM_VARS],
) -> Result<[D::Scalar; 3], DiffError> {
    let field_x = Component::new(vector_field, 0);
    let field_y = Component::new(vector_field, 1);
    let field_z = Component::new(vector_field, 2);

    let mut ans = [<D::Scalar as Numeric>::ZERO; 3];
    ans[0] = derivator.first_partial_derivative(&field_z, 1, point)?
        - derivator.first_partial_derivative(&field_y, 2, point)?;
    ans[1] = derivator.first_partial_derivative(&field_x, 2, point)?
        - derivator.first_partial_derivative(&field_z, 0, point)?;
    ans[2] = derivator.first_partial_derivative(&field_y, 0, point)?
        - derivator.first_partial_derivative(&field_x, 1, point)?;

    Ok(ans)
}

/// Computes the (scalar) curl of a 2D vector field at a point.
///
/// For a field `V = (Vx, Vy)`, the curl is `dVy/dx - dVx/dy`.
///
/// # Arguments
/// * `derivator` - the derivator used for the partial derivatives.
/// * `vector_field` - the two field components as one vector-valued function of `[x, y]`.
/// * `point` - the point at which the curl is evaluated.
///
/// # Errors
/// [`DiffError::StepSizeZero`] if the derivator's step size is zero.
///
/// # Examples
/// ```
/// use multicalc::numerical_derivative::AutoDiffMulti;
/// use multicalc::scalar::constant;
/// use multicalc::scalar_fn_vec;
/// use multicalc::vector_field::curl_2d;
///
/// // the field (2xy, 3cos(y))
/// let vf = scalar_fn_vec!(|point: &[f64; 2]| [constant(2.0) * point[0] * point[1], constant(3.0) * point[1].cos()]);
///
/// let derivator = AutoDiffMulti::default();
/// let point = [1.0, 3.14];
/// let val = curl_2d(derivator, &vf, &point).unwrap();
/// // curl is known to be -2
/// assert!(f64::abs(val + 2.0) < 1e-12);
/// ```
pub fn curl_2d<D: DerivatorMultiVariable, F: VectorFn<NUM_VARS, 2>, const NUM_VARS: usize>(
    derivator: D,
    vector_field: &F,
    point: &[D::Scalar; NUM_VARS],
) -> Result<D::Scalar, DiffError> {
    let field_x = Component::new(vector_field, 0);
    let field_y = Component::new(vector_field, 1);

    Ok(derivator.first_partial_derivative(&field_y, 0, point)?
        - derivator.first_partial_derivative(&field_x, 1, point)?)
}
