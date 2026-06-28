use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::utils::error_codes::CalcError;

/// A first-order (linear) Taylor approximation of a function about a base point:
/// `f(x) ≈ value + Σ gradient[i] * (x[i] - point[i])`.
#[derive(Debug, Clone, Copy)]
pub struct LinearApproximation<const NUM_VARS: usize> {
    point: [f64; NUM_VARS],
    value: f64,
    gradient: [f64; NUM_VARS],
}

/// Goodness-of-fit metrics for a [`LinearApproximation`] over a set of sample points.
#[derive(Debug, Clone, Copy)]
pub struct LinearApproximationPredictionMetrics {
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
}

impl<const NUM_VARS: usize> LinearApproximation<NUM_VARS> {
    /// Evaluates the approximation at `x`.
    pub fn predict(&self, x: &[f64; NUM_VARS]) -> f64 {
        let mut result = self.value;
        for i in 0..NUM_VARS {
            result += self.gradient[i] * (x[i] - self.point[i]);
        }
        result
    }

    /// The base point the approximation is centered on.
    pub fn point(&self) -> &[f64; NUM_VARS] {
        &self.point
    }

    /// The gradient at the base point. These are also the coefficients of the expanded
    /// linear form `intercept + Σ coefficients[i] * x[i]`.
    pub fn coefficients(&self) -> &[f64; NUM_VARS] {
        &self.gradient
    }

    /// The intercept of the expanded form `intercept + Σ coefficients[i] * x[i]`.
    pub fn intercept(&self) -> f64 {
        let mut intercept = self.value;
        for i in 0..NUM_VARS {
            intercept -= self.gradient[i] * self.point[i];
        }
        intercept
    }

    /// Computes goodness-of-fit metrics against `original_function` over `points`.
    ///
    /// `r_squared` is `NaN` when the truth is constant over `points`;
    /// `adjusted_r_squared` is `NaN` when there are too few points.
    pub fn get_prediction_metrics<O: Fn(&[f64; NUM_VARS]) -> f64, const NUM_POINTS: usize>(
        &self,
        points: &[[f64; NUM_VARS]; NUM_POINTS],
        original_function: &O,
    ) -> LinearApproximationPredictionMetrics {
        let (mae, mse, rmse, r_squared, adjusted_r_squared) = crate::approximation::compute_metrics(
            |x| self.predict(x),
            points,
            original_function,
            NUM_VARS, // p = N linear coefficients
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

pub struct LinearApproximator<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable + Default> Default for LinearApproximator<D> {
    fn default() -> Self {
        LinearApproximator {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> LinearApproximator<D> {
    pub fn from_derivator(derivator: D) -> Self {
        LinearApproximator { derivator }
    }

    /// Builds a linear (first-order Taylor) approximation of `function` about `point`.
    ///
    /// # Errors
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::approximation::linear_approximation::LinearApproximator;
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    ///
    /// // x + y^2 + z^3
    /// let function_to_approximate = | args: &[f64; 3] | -> f64
    /// {
    ///     return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
    /// };
    ///
    /// let point = [1.0, 2.0, 3.0]; //the point we want to linearize around
    /// let approximator = LinearApproximator::<FiniteDifferenceMulti>::default();
    /// let result = approximator.get(&function_to_approximate, &point).unwrap();
    ///
    /// //the approximation is exact at the base point
    /// assert!(f64::abs(function_to_approximate(&point) - result.predict(&point)) < 1e-9);
    /// ```
    pub fn get<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize>(
        &self,
        function: &F,
        point: &[f64; NUM_VARS],
    ) -> Result<LinearApproximation<NUM_VARS>, CalcError> {
        let value = function(point);

        let mut gradient = [0.0; NUM_VARS];
        for i in 0..NUM_VARS {
            gradient[i] = self.derivator.get_single_partial(function, i, point)?;
        }

        Ok(LinearApproximation {
            point: *point,
            value,
            gradient,
        })
    }
}
