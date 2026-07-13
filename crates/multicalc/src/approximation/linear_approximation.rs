use crate::linear_algebra::Vector;
use crate::numerical_derivative::autodiff::AutoDiffMulti;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::scalar::{Numeric, ScalarFnN};
use crate::utils::error_codes::CalcError;
use crate::utils::summation::SummationMethod;

/// A first-order (linear) Taylor approximation of a function about a base point:
/// `f(x) ≈ value + Σ gradient[i] * (x[i] - point[i])`.
#[derive(Debug, Clone, Copy)]
pub struct LinearApproximation<const NUM_VARS: usize, T = f64> {
    point: [T; NUM_VARS],
    value: T,
    gradient: [T; NUM_VARS],
    summation: SummationMethod,
}

/// Goodness-of-fit metrics for a [`LinearApproximation`] over a set of sample points.
#[derive(Debug, Clone, Copy)]
pub struct LinearApproximationPredictionMetrics<T = f64> {
    /// Mean absolute error.
    pub mean_absolute_error: T,
    /// Mean squared error.
    pub mean_squared_error: T,
    /// Root mean squared error.
    pub root_mean_squared_error: T,
    /// Coefficient of determination; `NaN` when the truth is constant over the points.
    pub r_squared: T,
    /// R² adjusted for the number of predictors; `NaN` when there are too few points.
    pub adjusted_r_squared: T,
}

impl<const NUM_VARS: usize, T: Numeric> LinearApproximation<NUM_VARS, T> {
    /// Evaluates the approximation at `x`.
    #[inline]
    pub fn predict(&self, x: &[T; NUM_VARS]) -> T {
        let dx = Vector::from(*x) - Vector::from(self.point);
        self.value + Vector::from(self.gradient).dot(dx)
    }

    /// The base point the approximation is centered on.
    pub fn point(&self) -> &[T; NUM_VARS] {
        &self.point
    }

    /// The gradient at the base point. These are also the coefficients of the expanded
    /// linear form `intercept + Σ coefficients[i] * x[i]`.
    pub fn coefficients(&self) -> &[T; NUM_VARS] {
        &self.gradient
    }

    /// The intercept of the expanded form `intercept + Σ coefficients[i] * x[i]`.
    pub fn intercept(&self) -> T {
        let mut intercept = self.value;
        for i in 0..NUM_VARS {
            intercept -= self.gradient[i] * self.point[i];
        }
        intercept
    }

    /// Computes goodness-of-fit metrics against `original_function` over `points`.
    ///
    /// Uses the summation method chosen when the approximator was built
    /// (pairwise by default; Kahan if [`LinearApproximator::with_kahan_summation`]
    /// was used).
    ///
    /// `r_squared` is `NaN` when the truth is constant over `points`;
    /// `adjusted_r_squared` is `NaN` when there are too few points.
    pub fn get_prediction_metrics<O: ScalarFnN<NUM_VARS>, const NUM_POINTS: usize>(
        &self,
        points: &[[T; NUM_VARS]; NUM_POINTS],
        original_function: &O,
    ) -> LinearApproximationPredictionMetrics<T> {
        let (mae, mse, rmse, r_squared, adjusted_r_squared) = crate::approximation::compute_metrics(
            |x| self.predict(x),
            points,
            &|x: &[T; NUM_VARS]| original_function.eval(x),
            NUM_VARS, // p = N linear coefficients
            self.summation,
        );

        LinearApproximationPredictionMetrics {
            mean_absolute_error: mae,
            mean_squared_error: mse,
            root_mean_squared_error: rmse,
            r_squared,
            adjusted_r_squared,
        }
    }
}

/// Builds a [`LinearApproximation`] of a function. The differentiation backend defaults to autodiff
/// ([`AutoDiffMulti`]); pass a finite-difference derivator explicitly to use that instead.
pub struct LinearApproximator<D: DerivatorMultiVariable = AutoDiffMulti> {
    derivator: D,
    summation: SummationMethod,
}

impl<D: DerivatorMultiVariable + Default> Default for LinearApproximator<D> {
    fn default() -> Self {
        LinearApproximator {
            derivator: D::default(),
            summation: SummationMethod::Pairwise,
        }
    }
}

impl<D: DerivatorMultiVariable> LinearApproximator<D> {
    /// Builds an approximator from an explicit derivator.
    pub fn from_derivator(derivator: D) -> Self {
        LinearApproximator {
            derivator,
            summation: SummationMethod::Pairwise,
        }
    }

    /// Opt in to Kahan compensated summation for prediction metrics.
    ///
    /// Pairwise summation remains the default. Call this before [`Self::get`] so the
    /// resulting [`LinearApproximation`] accumulates metrics with Kahan.
    pub fn with_kahan_summation(mut self) -> Self {
        self.summation = SummationMethod::Kahan;
        self
    }

    /// Builds a linear (first-order Taylor) approximation of `function` about `point`.
    ///
    /// # Errors
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::approximation::linear_approximation::LinearApproximator;
    /// use multicalc::scalar::ScalarFnN;
    /// use multicalc::scalar_fn;
    ///
    /// // x + y^2 + z^3
    /// let function_to_approximate = scalar_fn!(|v: &[f64; 3]| v[0] + v[1].powi(2) + v[2].powi(3));
    ///
    /// let point = [1.0, 2.0, 3.0]; // the point we want to linearize around
    /// let approximator: LinearApproximator = LinearApproximator::default();
    /// let result = approximator.get(&function_to_approximate, &point).unwrap();
    ///
    /// // the approximation is exact at the base point
    /// assert!(f64::abs(function_to_approximate.eval(&point) - result.predict(&point)) < 1e-12);
    /// ```
    pub fn get<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize>(
        &self,
        function: &F,
        point: &[D::Scalar; NUM_VARS],
    ) -> Result<LinearApproximation<NUM_VARS, D::Scalar>, CalcError> {
        let value = function.eval(point);

        let mut gradient = [<D::Scalar as Numeric>::ZERO; NUM_VARS];
        for (i, slot) in gradient.iter_mut().enumerate() {
            *slot = self.derivator.get_single_partial(function, i, point)?;
        }

        Ok(LinearApproximation {
            point: *point,
            value,
            gradient,
            summation: self.summation,
        })
    }
}
