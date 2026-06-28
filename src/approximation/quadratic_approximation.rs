use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::utils::error_codes::CalcError;

/// A second-order (quadratic) Taylor approximation of a function about a base point:
/// `f(x) ≈ value + Σ gradient[i]·dx[i] + ½ Σ_i Σ_j hessian[i][j]·dx[i]·dx[j]`,
/// where `dx[i] = x[i] - point[i]`.
#[derive(Debug, Clone, Copy)]
pub struct QuadraticApproximation<const NUM_VARS: usize> {
    point: [f64; NUM_VARS],
    value: f64,
    gradient: [f64; NUM_VARS],
    hessian: [[f64; NUM_VARS]; NUM_VARS],
}

/// Goodness-of-fit metrics for a [`QuadraticApproximation`] over a set of sample points.
#[derive(Debug, Clone, Copy)]
pub struct QuadraticApproximationPredictionMetrics {
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
}

impl<const NUM_VARS: usize> QuadraticApproximation<NUM_VARS> {
    /// Evaluates the approximation at `x`. The `½` keeps the quadratic term correct for
    /// both diagonal and off-diagonal Hessian entries.
    pub fn predict(&self, x: &[f64; NUM_VARS]) -> f64 {
        let mut result = self.value;
        for i in 0..NUM_VARS {
            let di = x[i] - self.point[i];
            result += self.gradient[i] * di;
            for j in 0..NUM_VARS {
                result += 0.5 * self.hessian[i][j] * di * (x[j] - self.point[j]);
            }
        }
        result
    }

    /// The base point the approximation is centered on.
    pub fn point(&self) -> &[f64; NUM_VARS] {
        &self.point
    }

    /// The gradient at the base point.
    pub fn gradient(&self) -> &[f64; NUM_VARS] {
        &self.gradient
    }

    /// The Hessian matrix at the base point.
    pub fn hessian(&self) -> &[[f64; NUM_VARS]; NUM_VARS] {
        &self.hessian
    }

    /// Computes goodness-of-fit metrics against `original_function` over `points`.
    ///
    /// `r_squared` is `NaN` when the truth is constant over `points`;
    /// `adjusted_r_squared` is `NaN` when there are too few points.
    pub fn get_prediction_metrics<O: Fn(&[f64; NUM_VARS]) -> f64, const NUM_POINTS: usize>(
        &self,
        points: &[[f64; NUM_VARS]; NUM_POINTS],
        original_function: &O,
    ) -> QuadraticApproximationPredictionMetrics {
        // p = N gradient terms + N(N+1)/2 distinct (symmetric) Hessian terms
        let num_predictors = NUM_VARS + NUM_VARS * (NUM_VARS + 1) / 2;

        let (mae, mse, rmse, r_squared, adjusted_r_squared) = crate::approximation::compute_metrics(
            |x| self.predict(x),
            points,
            original_function,
            num_predictors,
        );

        QuadraticApproximationPredictionMetrics {
            mean_absolute_error: mae,
            mean_squared_error: mse,
            root_mean_squared_error: rmse,
            r_squared,
            adjusted_r_squared,
        }
    }
}

pub struct QuadraticApproximator<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable + Default> Default for QuadraticApproximator<D> {
    fn default() -> Self {
        QuadraticApproximator {
            derivator: D::default(),
        }
    }
}

impl<D: DerivatorMultiVariable> QuadraticApproximator<D> {
    pub fn from_derivator(derivator: D) -> Self {
        QuadraticApproximator { derivator }
    }

    /// Builds a quadratic (second-order Taylor) approximation of `function` about `point`.
    ///
    /// # Errors
    /// [`CalcError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::approximation::quadratic_approximation::QuadraticApproximator;
    /// use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
    ///
    /// // e^(x/2) + sin(y) + 2z
    /// let function_to_approximate = | args: &[f64; 3] | -> f64
    /// {
    ///     return f64::exp(args[0]/2.0) + f64::sin(args[1]) + 2.0*args[2];
    /// };
    ///
    /// let point = [0.0, 1.57, 10.0]; //the point we want to approximate around
    /// let approximator = QuadraticApproximator::<FiniteDifferenceMulti>::default();
    /// let result = approximator.get(&function_to_approximate, &point).unwrap();
    ///
    /// //the approximation is exact at the base point
    /// assert!(f64::abs(function_to_approximate(&point) - result.predict(&point)) < 1e-9);
    /// ```
    pub fn get<F: Fn(&[f64; NUM_VARS]) -> f64, const NUM_VARS: usize>(
        &self,
        function: &F,
        point: &[f64; NUM_VARS],
    ) -> Result<QuadraticApproximation<NUM_VARS>, CalcError> {
        let value = function(point);

        let mut gradient = [0.0; NUM_VARS];
        for i in 0..NUM_VARS {
            gradient[i] = self.derivator.get_single_partial(function, i, point)?;
        }

        // symmetric fill: compute the upper triangle + diagonal once, mirror the rest
        let mut hessian = [[f64::NAN; NUM_VARS]; NUM_VARS];
        for row in 0..NUM_VARS {
            for col in 0..NUM_VARS {
                if hessian[row][col].is_nan() {
                    hessian[row][col] =
                        self.derivator.get_double_partial(function, &[row, col], point)?;
                    hessian[col][row] = hessian[row][col];
                }
            }
        }

        Ok(QuadraticApproximation {
            point: *point,
            value,
            gradient,
            hessian,
        })
    }
}
