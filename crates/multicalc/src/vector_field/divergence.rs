use crate::error::DiffError;
use crate::numerical_derivative::DerivatorMultiVariable;
use crate::scalar::Component;
use crate::scalar::VectorFn;

/// Computes the divergence of a 3D vector field at a point.
///
/// For a field `V = (Vx, Vy, Vz)`, the divergence is `dVx/dx + dVy/dy + dVz/dz`.
///
/// # Arguments
/// * `derivator` - the derivator used for the partial derivatives.
/// * `vector_field` - the three field components as one vector-valued function of `[x, y, z]`.
/// * `point` - the point at which the divergence is evaluated.
///
/// # Errors
/// [`DiffError::StepSizeZero`] if the derivator's step size is zero.
///
/// # Examples
/// ```
/// use multicalc::numerical_derivative::AutoDiffMulti;
/// use multicalc::scalar::c;
/// use multicalc::scalar_fn_vec;
/// use multicalc::vector_field::divergence_3d;
///
/// // the field (y, -x, 2z)
/// let vf = scalar_fn_vec!(|v: &[f64; 3]| [v[1], -v[0], c(2.0) * v[2]]);
///
/// let derivator = AutoDiffMulti::default();
/// let point = [0.0, 1.0, 3.0];
/// let val = divergence_3d(derivator, &vf, &point).unwrap();
/// // divergence is known to be 2
/// assert!(f64::abs(val - 2.0) < 1e-12);
/// ```
pub fn divergence_3d<D: DerivatorMultiVariable, F: VectorFn<NUM_VARS, 3>, const NUM_VARS: usize>(
    derivator: D,
    vector_field: &F,
    point: &[D::Scalar; NUM_VARS],
) -> Result<D::Scalar, DiffError> {
    let vx = Component::new(vector_field, 0);
    let vy = Component::new(vector_field, 1);
    let vz = Component::new(vector_field, 2);

    Ok(derivator.first_partial_derivative(&vx, 0, point)?
        + derivator.first_partial_derivative(&vy, 1, point)?
        + derivator.first_partial_derivative(&vz, 2, point)?)
}

/// Computes the divergence of a 2D vector field at a point.
///
/// For a field `V = (Vx, Vy)`, the divergence is `dVx/dx + dVy/dy`.
///
/// # Arguments
/// * `derivator` - the derivator used for the partial derivatives.
/// * `vector_field` - the two field components as one vector-valued function of `[x, y]`.
/// * `point` - the point at which the divergence is evaluated.
///
/// # Errors
/// [`DiffError::StepSizeZero`] if the derivator's step size is zero.
///
/// # Examples
/// ```
/// use multicalc::numerical_derivative::AutoDiffMulti;
/// use multicalc::scalar_fn_vec;
/// use multicalc::vector_field::divergence_2d;
///
/// // the field (y, -x)
/// let vf = scalar_fn_vec!(|v: &[f64; 2]| [v[1], -v[0]]);
///
/// let derivator = AutoDiffMulti::default();
/// let point = [0.0, 1.0];
/// let val = divergence_2d(derivator, &vf, &point).unwrap();
/// // divergence is known to be 0
/// assert!(f64::abs(val) < 1e-12);
/// ```
pub fn divergence_2d<D: DerivatorMultiVariable, F: VectorFn<NUM_VARS, 2>, const NUM_VARS: usize>(
    derivator: D,
    vector_field: &F,
    point: &[D::Scalar; NUM_VARS],
) -> Result<D::Scalar, DiffError> {
    let vx = Component::new(vector_field, 0);
    let vy = Component::new(vector_field, 1);

    Ok(derivator.first_partial_derivative(&vx, 0, point)?
        + derivator.first_partial_derivative(&vy, 1, point)?)
}
