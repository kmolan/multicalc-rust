use crate::error::DiffError;
use crate::linear_algebra::Matrix;
use crate::numerical_derivative::autodiff::AutoDiffMulti;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::scalar::VectorFn;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Computes the Jacobian matrix of a vector-valued function. The differentiation backend
/// defaults to autodiff ([`AutoDiffMulti`]); pass a finite-difference derivator explicitly to
/// use that instead.
pub struct Jacobian<D: DerivatorMultiVariable = AutoDiffMulti> {
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
    /// [`DiffError::EmptyFunctionSet`] if `function` has no outputs, or
    /// [`DiffError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::jacobian::Jacobian;
    /// use multicalc::scalar_fn_vec;
    ///
    /// // the vector function (x*y*z, x^2 + y^2)
    /// let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    ///
    ///
    /// let jacobian: Jacobian = Jacobian::default();
    /// let result = jacobian.get(&f, &[1.0, 2.0, 3.0]).unwrap();
    /// // result is [[6, 3, 2], [2, 4, 0]]
    /// assert!(f64::abs(result[0][0] - 6.0) < 1e-12);
    /// ```
    pub fn get<F: VectorFn<NUM_VARS, NUM_FUNCS>, const NUM_FUNCS: usize, const NUM_VARS: usize>(
        &self,
        function: &F,
        vector_of_points: &[D::Scalar; NUM_VARS],
    ) -> Result<[[D::Scalar; NUM_VARS]; NUM_FUNCS], DiffError> {
        if NUM_FUNCS == 0 {
            return Err(DiffError::EmptyFunctionSet);
        }

        let mut result: Matrix<NUM_FUNCS, NUM_VARS, D::Scalar> = Matrix::zeros();

        for n in 0..NUM_VARS {
            let column = self
                .derivator
                .jacobian_column(function, n, vector_of_points)?;
            for (m, &value) in column.iter().enumerate() {
                result[(m, n)] = value;
            }
        }

        Ok(result.into_array())
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
    /// [`DiffError::EmptyFunctionSet`] if `function` has no outputs, or
    /// [`DiffError::StepSizeZero`] if the derivator's step size is zero.
    #[cfg(feature = "alloc")]
    pub fn get_on_heap<
        F: VectorFn<NUM_VARS, NUM_FUNCS>,
        const NUM_FUNCS: usize,
        const NUM_VARS: usize,
    >(
        &self,
        function: &F,
        vector_of_points: &[D::Scalar; NUM_VARS],
    ) -> Result<Vec<Vec<D::Scalar>>, DiffError> {
        if NUM_FUNCS == 0 {
            return Err(DiffError::EmptyFunctionSet);
        }

        let mut result: Vec<Vec<D::Scalar>> = Vec::with_capacity(NUM_FUNCS);
        for _ in 0..NUM_FUNCS {
            result.push(Vec::with_capacity(NUM_VARS));
        }

        for n in 0..NUM_VARS {
            let column = self
                .derivator
                .jacobian_column(function, n, vector_of_points)?;
            for (m, &value) in column.iter().enumerate() {
                result[m].push(value);
            }
        }

        Ok(result)
    }
}
