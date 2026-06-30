use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::scalar::function::Component;
use crate::scalar::{Numeric, VectorFn};
use crate::utils::error_codes::CalcError;

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
/// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
///
/// # Examples
/// ```
/// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
/// use multicalc::scalar::c;
/// use multicalc::scalar_fn_vec;
/// use multicalc::vector_field::curl;
///
/// // the field (y, -x, 2z)
/// let vf = scalar_fn_vec!(|v: &[f64; 3]| [v[1], -v[0], c(2.0) * v[2]]);
///
/// let derivator = FiniteDifferenceMulti::default();
/// let val = curl::get_3d(derivator, &vf, &[0.0, 1.0, 3.0]).unwrap();
/// // curl is known to be (0, 0, -2)
/// assert!(f64::abs(val[2] + 2.0) < 1e-5);
/// ```
pub fn get_3d<D: DerivatorMultiVariable, F: VectorFn<NUM_VARS, 3>, const NUM_VARS: usize>(
    derivator: D,
    vector_field: &F,
    point: &[D::Scalar; NUM_VARS],
) -> Result<[D::Scalar; 3], CalcError> {
    let vx = Component::new(vector_field, 0);
    let vy = Component::new(vector_field, 1);
    let vz = Component::new(vector_field, 2);

    let mut ans = [<D::Scalar as Numeric>::ZERO; 3];
    ans[0] = derivator.get_single_partial(&vz, 1, point)?
        - derivator.get_single_partial(&vy, 2, point)?;
    ans[1] = derivator.get_single_partial(&vx, 2, point)?
        - derivator.get_single_partial(&vz, 0, point)?;
    ans[2] = derivator.get_single_partial(&vy, 0, point)?
        - derivator.get_single_partial(&vx, 1, point)?;

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
/// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
///
/// # Examples
/// ```
/// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
/// use multicalc::scalar::c;
/// use multicalc::scalar_fn_vec;
/// use multicalc::vector_field::curl;
///
/// // the field (2xy, 3cos(y))
/// let vf = scalar_fn_vec!(|v: &[f64; 2]| [c(2.0) * v[0] * v[1], c(3.0) * v[1].cos()]);
///
/// let derivator = FiniteDifferenceMulti::default();
/// let val = curl::get_2d(derivator, &vf, &[1.0, 3.14]).unwrap();
/// // curl is known to be -2
/// assert!(f64::abs(val + 2.0) < 1e-5);
/// ```
pub fn get_2d<D: DerivatorMultiVariable, F: VectorFn<NUM_VARS, 2>, const NUM_VARS: usize>(
    derivator: D,
    vector_field: &F,
    point: &[D::Scalar; NUM_VARS],
) -> Result<D::Scalar, CalcError> {
    let vx = Component::new(vector_field, 0);
    let vy = Component::new(vector_field, 1);

    Ok(derivator.get_single_partial(&vy, 0, point)?
        - derivator.get_single_partial(&vx, 1, point)?)
}
