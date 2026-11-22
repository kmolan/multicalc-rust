use crate::error::DiffError;
use crate::scalar::{Numeric, ScalarFn, ScalarFnN, VectorFn};

/// Base trait for single-variable differentiation.
pub trait DerivatorSingleVariable {
    /// The scalar the derivative is computed in.
    type Scalar: Numeric;

    /// Computes the `order`-th derivative of `func` at `point`.
    ///
    /// # Errors
    /// [`DiffError::OrderZero`] if `order` is zero, or
    /// [`DiffError::StepSizeZero`] if the configured step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::DerivatorSingleVariable;
    /// use multicalc::numerical_derivative::FiniteDifferenceSingle;
    /// use multicalc::scalar_fn;
    ///
    /// let function = scalar_fn!(|x| x * x * x);
    /// let derivator = FiniteDifferenceSingle::default();
    /// let point = 2.0;
    ///
    /// let first_order = 1;
    /// let slope = derivator.differentiate(first_order, &function, point).unwrap();
    /// assert!(f64::abs(slope - 12.0) < 1e-7);
    ///
    /// let second_order = 2;
    /// let bend = derivator.differentiate(second_order, &function, point).unwrap();
    /// assert!(f64::abs(bend - 12.0) < 1e-5);
    /// ```
    fn differentiate<F: ScalarFn>(
        &self,
        order: usize,
        func: &F,
        point: Self::Scalar,
    ) -> Result<Self::Scalar, DiffError>;

    /// Convenience wrapper for the first derivative.
    fn first_derivative<F: ScalarFn>(
        &self,
        func: &F,
        point: Self::Scalar,
    ) -> Result<Self::Scalar, DiffError> {
        self.differentiate(1, func, point)
    }

    /// Convenience wrapper for the second derivative.
    fn second_derivative<F: ScalarFn>(
        &self,
        func: &F,
        point: Self::Scalar,
    ) -> Result<Self::Scalar, DiffError> {
        self.differentiate(2, func, point)
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
    /// [`DiffError::OrderZero`] if `idx_to_differentiate` is empty,
    /// [`DiffError::StepSizeZero`] if the step size is zero, or
    /// [`DiffError::IndexOutOfRange`] if any index is `>= NUM_VARS`.
    ///
    /// # Examples
    /// ```
    /// use multicalc::numerical_derivative::DerivatorMultiVariable;
    /// use multicalc::numerical_derivative::FiniteDifferenceMulti;
    /// use multicalc::scalar_fn;
    ///
    /// // f(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z
    /// let function = scalar_fn!(|point: &[f64; 3]| point[1] * point[0].sin() + point[0] * point[1].cos() + point[0] * point[1] * point[2].exp());
    /// let derivator = FiniteDifferenceMulti::default();
    /// let variable_indices = [0, 1];   // once by x, then by y
    /// let point = [1.0, 2.0, 3.0];
    ///
    /// let mixed = derivator.differentiate(&function, &variable_indices, &point).unwrap();
    /// let expected = f64::cos(1.0) - f64::sin(2.0) + f64::exp(3.0);
    /// assert!(f64::abs(mixed - expected) < 0.001);
    /// ```
    fn differentiate<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize, const NUM_ORDER: usize>(
        &self,
        func: &F,
        idx_to_differentiate: &[usize; NUM_ORDER],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, DiffError>;

    /// Convenience wrapper for a single partial derivative.
    fn first_partial_derivative<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize>(
        &self,
        func: &F,
        idx_to_differentiate: usize,
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, DiffError> {
        self.differentiate(func, &[idx_to_differentiate], point)
    }

    /// Convenience wrapper for a second partial derivative.
    fn second_partial_derivative<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize>(
        &self,
        func: &F,
        idx_to_differentiate: &[usize; 2],
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<Self::Scalar, DiffError> {
        self.differentiate(func, idx_to_differentiate, point)
    }

    /// Returns one column of a vector function's Jacobian: the partial derivative of every
    /// output with respect to input `col`, at `point`.
    ///
    /// Implementations compute the whole column at once — forward-mode from a
    /// single seeded pass, finite difference from two full-vector evaluations.
    ///
    /// # Errors
    /// [`DiffError::IndexOutOfRange`] if `col >= NUM_VARS`, plus whatever error the underlying
    /// evaluation returns.
    fn jacobian_column<
        F: VectorFn<NUM_VARS, NUM_FUNCS>,
        const NUM_VARS: usize,
        const NUM_FUNCS: usize,
    >(
        &self,
        func: &F,
        col: usize,
        point: &[Self::Scalar; NUM_VARS],
    ) -> Result<[Self::Scalar; NUM_FUNCS], DiffError>;
}
