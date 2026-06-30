use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::scalar::function::Component;
use crate::scalar::{Numeric, VectorFn};
use crate::utils::error_codes::CalcError;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Computes the Jacobian matrix of a vector-valued function, using any derivator that
/// implements [`DerivatorMultiVariable`].
pub struct Jacobian<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable + Default> Default for Jacobian<D> {
    fn default() -> Self {
        Jacobian {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> Jacobian<D> {
    /// Builds a Jacobian from an explicit derivator. Use this to supply a custom derivator,
    /// either one from this crate or your own implementation of [`DerivatorMultiVariable`].
    pub fn from_derivator(derivator: D) -> Self {
        Jacobian { derivator }
    }

    /// Returns the Jacobian matrix of `function` evaluated at `vector_of_points`.
    ///
    /// The result has one row per output and one column per variable, so entry `[m][n]`
    /// is `d(output m)/d(variable n)`.
    ///
    /// # Arguments
    /// * `function` - the vector-valued function whose partial derivatives form the rows.
    /// * `vector_of_points` - the point at which the derivatives are taken.
    ///
    /// # Errors
    /// [`CalcError::EmptyFunctionSet`] if `function` has no outputs, or
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    /// use multicalc::numerical_derivative::jacobian::Jacobian;
    /// use multicalc::scalar_fn_vec;
    ///
    /// // the vector function (x*y*z, x^2 + y^2)
    /// let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    ///
    /// let jacobian = Jacobian::<FiniteDifferenceMulti>::default();
    /// let result = jacobian.get(&f, &[1.0, 2.0, 3.0]).unwrap();
    /// // result is [[6, 3, 2], [2, 4, 0]]
    /// assert!(f64::abs(result[0][0] - 6.0) < 1e-6);
    /// ```
    pub fn get<F: VectorFn<NUM_VARS, NUM_FUNCS>, const NUM_FUNCS: usize, const NUM_VARS: usize>(
        &self,
        function: &F,
        vector_of_points: &[D::Scalar; NUM_VARS],
    ) -> Result<[[D::Scalar; NUM_VARS]; NUM_FUNCS], CalcError> {
        if NUM_FUNCS == 0 {
            return Err(CalcError::EmptyFunctionSet);
        }

        let mut result = [[<D::Scalar as Numeric>::ZERO; NUM_VARS]; NUM_FUNCS];

        for (m, row) in result.iter_mut().enumerate() {
            let component = Component::new(function, m);
            for (col_index, slot) in row.iter_mut().enumerate() {
                *slot =
                    self.derivator
                        .get_single_partial(&component, col_index, vector_of_points)?;
            }
        }

        Ok(result)
    }

    /// Same as [`Jacobian::get`] but returns a heap-allocated `Vec<Vec<_>>`, which avoids a
    /// stack overflow on large inputs. Requires the `alloc` feature (off by default).
    ///
    /// The result has one row per output and one column per variable, so entry `[m][n]`
    /// is `d(output m)/d(variable n)`.
    ///
    /// # Arguments
    /// * `function` - the vector-valued function whose partial derivatives form the rows.
    /// * `vector_of_points` - the point at which the derivatives are taken.
    ///
    /// # Errors
    /// [`CalcError::EmptyFunctionSet`] if `function` has no outputs, or
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    #[cfg(feature = "alloc")]
    pub fn get_on_heap<
        F: VectorFn<NUM_VARS, NUM_FUNCS>,
        const NUM_FUNCS: usize,
        const NUM_VARS: usize,
    >(
        &self,
        function: &F,
        vector_of_points: &[D::Scalar; NUM_VARS],
    ) -> Result<Vec<Vec<D::Scalar>>, CalcError> {
        if NUM_FUNCS == 0 {
            return Err(CalcError::EmptyFunctionSet);
        }

        let mut result: Vec<Vec<D::Scalar>> = Vec::new();

        for m in 0..NUM_FUNCS {
            let component = Component::new(function, m);
            let mut cur_row: Vec<D::Scalar> = Vec::new();
            for col_index in 0..NUM_VARS {
                cur_row.push(self.derivator.get_single_partial(
                    &component,
                    col_index,
                    vector_of_points,
                )?);
            }
            result.push(cur_row);
        }

        Ok(result)
    }
}
