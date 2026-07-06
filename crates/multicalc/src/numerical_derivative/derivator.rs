use crate::scalar::{Numeric, ScalarFn, ScalarFnN};
use crate::utils::error_codes::CalcError;

/// Base trait for single-variable differentiation.
pub trait DerivatorSingleVariable {
    /// The scalar the derivative is computed in.
    type Scalar: Numeric;

    /// Computes the `order`-th derivative of `func` at `point`.
    ///
    /// # Errors
    /// [`CalcError::DerivativeOrderZero`] if `order` is zero, or
    /// [`CalcError::StepSizeZero`] if the configured step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceSingle;
    /// use multicalc::scalar_fn;
    ///
    /// let func = scalar_fn!(|x| x * x * x);
    /// let derivator = FiniteDifferenceSingle::default();
    ///
    /// let val = derivator.get(1, &func, 2.0).unwrap();
    /// assert!(f64::abs(val - 12.0) < 1e-7);
    /// let val = derivator.get(2, &func, 2.0).unwrap();
    /// assert!(f64::abs(val - 12.0) < 1e-5);
    /// ```
    fn get<F: ScalarFn>(
        &self,
        order: usize,
        func: &F,
        point: Self::Scalar,
    ) -> Result<Self::Scalar, CalcError>;

    /// Convenience wrapper for the first derivative.
    fn get_single<F: ScalarFn>(
        &self,
        func: &F,
        point: Self::Scalar,
    ) -> Result<Self::Scalar, CalcError> {
        self.get(1, func, point)
    }

    /// Convenience wrapper for the second derivative.
    fn get_double<F: ScalarFn>(
        &self,
        func: &F,
        point: Self::Scalar,
    ) -> Result<Self::Scalar, CalcError> {
        self.get(2, func, point)
    }
}

/// Base trait for multi-variable differentiation.
pub trait DerivatorMultiVariable {
    /// The scalar the derivative is computed in.
    type Scalar: Numeric;

    /// Computes the partial derivative of `func` at `point`, differentiating once
    /// with respect to each variable index listed in `idx_to_differentiate`. The
    /// derivative order equals the length of that array.
    ///
    /// # Errors
    /// [`CalcError::DerivativeOrderZero`] if `idx_to_differentiate` is empty,
    /// [`CalcError::StepSizeZero`] if the step size is zero, or
    /// [`CalcError::IndexOutOfRange`] if any index is `>= NUM_VARS`.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::derivator::DerivatorMultiVariable;
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    /// use multicalc::scalar_fn;
    ///
    /// // f(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z
    /// let func = scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin() + v[0] * v[1].cos() + v[0] * v[1] * v[2].exp());
    /// let derivator = FiniteDifferenceMulti::default();
    ///
    /// // mixed partial d(df/dx)/dy
    /// let val = derivator.get(&func, &[0, 1], &[1.0, 2.0, 3.0]).unwrap();
    /// let expected = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
    /// assert!(f64::abs(val - expected) < 0.001);
    /// ```
    fn get<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, CalcError>;

    /// Convenience wrapper for a single partial derivative.
    fn get_single_partial<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize>(
        &self,
        func: &F,
        idx_to_differentiate: usize,
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, CalcError> {
        self.get(func, &[idx_to_differentiate], point)
    }

    /// Convenience wrapper for a second partial derivative.
    fn get_double_partial<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize>(
        &self,
        func: &F,
        idx_to_differentiate: &[usize; 2],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, CalcError> {
        self.get(func, idx_to_differentiate, point)
    }
}
