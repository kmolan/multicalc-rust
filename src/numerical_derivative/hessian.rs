use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::utils::error_codes::CalcError;

/// Computes the Hessian matrix of a scalar multi-variable function, using any derivator
/// that implements [`DerivatorMultiVariable`].
pub struct Hessian<D: DerivatorMultiVariable> {
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
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    /// use multicalc::numerical_derivative::hessian::Hessian;
    ///
    /// // f(x, y) = y*sin(x) + 2*x*e^y
    /// let my_func = |args: &[f64; 2]| args[1] * args[0].sin() + 2.0 * args[0] * args[1].exp();
    ///
    /// let hessian = Hessian::<FiniteDifferenceMulti>::default();
    /// let result = hessian.get(&my_func, &[1.0, 2.0]).unwrap();
    /// assert!(f64::abs(result[0][0] - (-2.0 * f64::sin(1.0))) < 1e-5);
    /// ```
    pub fn get<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize>(
        &self,
        function: &F,
        vector_of_points: &[f64; NUM_VARS],
    ) -> Result<[[f64; NUM_VARS]; NUM_VARS], CalcError> {
        let mut result = [[f64::NAN; NUM_VARS]; NUM_VARS];

        for row_index in 0..NUM_VARS {
            for col_index in 0..NUM_VARS {
                if result[row_index][col_index].is_nan() {
                    result[row_index][col_index] = self.derivator.get_double_partial(
                        function,
                        &[row_index, col_index],
                        vector_of_points,
                    )?;

                    result[col_index][row_index] = result[row_index][col_index];
                    //exploit the fact that a hessian is a symmetric matrix
                }
            }
        }

        Ok(result)
    }
}
