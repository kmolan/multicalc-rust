use crate::error::DiffError;
use crate::linear_algebra::Matrix;
use crate::numerical_derivative::autodiff::AutoDiffMulti;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::scalar::{Numeric, ScalarFnN};

/// Computes the Hessian matrix of a scalar multi-variable function. The differentiation backend
/// defaults to autodiff ([`AutoDiffMulti`]); pass a finite-difference derivator explicitly to use
/// that instead.
pub struct Hessian<D: DerivatorMultiVariable = AutoDiffMulti> {
    derivator: D,
}

impl<D: DerivatorMultiVariable + Default> Default for Hessian<D> {
    fn default() -> Self {
        Hessian {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> Hessian<D> {
    /// Builds a Hessian from an explicit derivator. Use this to supply a custom derivator,
    /// either one from this crate or your own implementation of [`DerivatorMultiVariable`].
    pub fn from_derivator(derivator: D) -> Self {
        Hessian { derivator }
    }

    /// Returns the Hessian matrix of `function` evaluated at `vector_of_points`.
    ///
    /// The result is the symmetric matrix of second partial derivatives, so entry `[i][j]`
    /// is `d²(function)/d(variable i) d(variable j)`. Only the upper triangle and diagonal
    /// are evaluated; the rest is mirrored, relying on the symmetry of the Hessian.
    ///
    /// # Arguments
    /// * `function` - the scalar function to differentiate.
    /// * `vector_of_points` - the point at which the derivatives are taken.
    ///
    /// # Errors
    /// [`DiffError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::hessian::Hessian;
    /// use multicalc::scalar::c;
    /// use multicalc::scalar_fn;
    ///
    /// // f(x, y) = y*sin(x) + 2*x*e^y
    /// let my_func =
    ///     scalar_fn!(|args: &[f64; 2]| args[1] * args[0].sin() + c(2.0) * args[0] * args[1].exp());
    ///
    ///
    /// let hessian: Hessian = Hessian::default();
    /// let result = hessian.get(&my_func, &[1.0, 2.0]).unwrap();
    /// assert!(f64::abs(result[0][0] - (-2.0 * f64::sin(1.0))) < 1e-12);
    /// ```
    pub fn get<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize>(
        &self,
        function: &F,
        vector_of_points: &[D::Scalar; NUM_VARS],
    ) -> Result<[[D::Scalar; NUM_VARS]; NUM_VARS], DiffError> {
        let mut result: Matrix<NUM_VARS, NUM_VARS, D::Scalar> =
            Matrix::from_fn(|_, _| <D::Scalar as Numeric>::NAN);

        // explicit indices drive the symmetric mirror write `result[(col, row)]`
        for row_index in 0..NUM_VARS {
            for col_index in 0..NUM_VARS {
                if result[(row_index, col_index)].is_nan() {
                    result[(row_index, col_index)] = self.derivator.get_double_partial(
                        function,
                        &[row_index, col_index],
                        vector_of_points,
                    )?;
                    // a Hessian is symmetric, so mirror instead of recomputing
                    result[(col_index, row_index)] = result[(row_index, col_index)];
                }
            }
        }

        Ok(result.into_array())
    }
}
