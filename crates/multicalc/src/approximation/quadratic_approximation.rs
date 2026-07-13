use crate::linear_algebra::{Matrix, Vector};
use crate::numerical_derivative::autodiff::AutoDiffMulti;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::scalar::{Numeric, ScalarFnN};
use crate::error::DiffError;

/// A second-order (quadratic) Taylor approximation of a function about a base point:
/// `f(x) ≈ value + Σ gradient[i]·dx[i] + ½ Σ_i Σ_j hessian[i][j]·dx[i]·dx[j]`,
/// where `dx[i] = x[i] - point[i]`.
#[derive(Debug, Clone, Copy)]
pub struct QuadraticApproximation<const NUM_VARS: usize, T = f64> {
    point: [T; NUM_VARS],
    value: T,
    gradient: [T; NUM_VARS],
    hessian: [[T; NUM_VARS]; NUM_VARS],
}

/// Goodness-of-fit metrics for a [`QuadraticApproximation`] over a set of sample points.
#[derive(Debug, Clone, Copy)]
pub struct QuadraticApproximationPredictionMetrics<T = f64> {
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

impl<const NUM_VARS: usize, T: Numeric> QuadraticApproximation<NUM_VARS, T> {
    /// Evaluates the approximation at `x`. The `½` keeps the quadratic term correct for
    /// both diagonal and off-diagonal Hessian entries.
    #[inline]
    pub fn predict(&self, x: &[T; NUM_VARS]) -> T {
        let dx = Vector::from(*x) - Vector::from(self.point);
        let hessian = Matrix::from(self.hessian);
        self.value + Vector::from(self.gradient).dot(dx) + T::HALF * (hessian * dx).dot(dx)
    }

    /// The base point the approximation is centered on.
    pub fn point(&self) -> &[T; NUM_VARS] {
        &self.point
    }

    /// The gradient at the base point.
    pub fn gradient(&self) -> &[T; NUM_VARS] {
        &self.gradient
    }

    /// The Hessian matrix at the base point.
    pub fn hessian(&self) -> &[[T; NUM_VARS]; NUM_VARS] {
        &self.hessian
    }

    /// Computes goodness-of-fit metrics against `original_function` over `points`.
    ///
    /// `r_squared` is `NaN` when the truth is constant over `points`;
    /// `adjusted_r_squared` is `NaN` when there are too few points.
    pub fn get_prediction_metrics<O: ScalarFnN<NUM_VARS>, const NUM_POINTS: usize>(
        &self,
        points: &[[T; NUM_VARS]; NUM_POINTS],
        original_function: &O,
    ) -> QuadraticApproximationPredictionMetrics<T> {
        // p = N gradient terms + N(N+1)/2 distinct (symmetric) Hessian terms
        let num_predictors = NUM_VARS + NUM_VARS * (NUM_VARS + 1) / 2;

        let (mae, mse, rmse, r_squared, adjusted_r_squared) = crate::approximation::compute_metrics(
            |x| self.predict(x),
            points,
            &|x: &[T; NUM_VARS]| original_function.eval(x),
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

/// Builds a [`QuadraticApproximation`] of a function. The differentiation backend defaults to
/// autodiff ([`AutoDiffMulti`]); pass a finite-difference derivator explicitly to use that instead.
pub struct QuadraticApproximator<D: DerivatorMultiVariable = AutoDiffMulti> {
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
    /// Builds an approximator from an explicit derivator.
    pub fn from_derivator(derivator: D) -> Self {
        QuadraticApproximator { derivator }
    }

    /// Builds a quadratic (second-order Taylor) approximation of `function` about `point`.
    ///
    /// # Errors
    /// [`DiffError::StepSizeZero`] if the derivator's step size is zero.
    ///
    /// # Examples
    /// ```
    /// use multicalc::approximation::quadratic_approximation::QuadraticApproximator;
    /// use multicalc::scalar::{c, ScalarFnN};
    /// use multicalc::scalar_fn;
    ///
    /// // e^(x/2) + sin(y) + 2z
    /// let function_to_approximate =
    ///     scalar_fn!(|v: &[f64; 3]| (c(0.5) * v[0]).exp() + v[1].sin() + c(2.0) * v[2]);
    ///
    /// let point = [0.0, 1.57, 10.0]; // the point we want to approximate around
    /// let approximator: QuadraticApproximator = QuadraticApproximator::default();
    /// let result = approximator.get(&function_to_approximate, &point).unwrap();
    ///
    /// // the approximation is exact at the base point
    /// assert!(f64::abs(function_to_approximate.eval(&point) - result.predict(&point)) < 1e-12);
    /// ```
    pub fn get<F: ScalarFnN<NUM_VARS>, const NUM_VARS: usize>(
        &self,
        function: &F,
        point: &[D::Scalar; NUM_VARS],
    ) -> Result<QuadraticApproximation<NUM_VARS, D::Scalar>, DiffError> {
        let value = function.eval(point);

        let mut gradient = [<D::Scalar as Numeric>::ZERO; NUM_VARS];
        for (i, slot) in gradient.iter_mut().enumerate() {
            *slot = self.derivator.get_single_partial(function, i, point)?;
        }

        // symmetric fill: compute the upper triangle + diagonal once, mirror the rest.
        // the explicit indices are needed for the mirror write `hessian[col][row]`.
        let mut hessian = [[<D::Scalar as Numeric>::NAN; NUM_VARS]; NUM_VARS];
        #[allow(clippy::needless_range_loop)]
        for row in 0..NUM_VARS {
            for col in 0..NUM_VARS {
                if hessian[row][col].is_nan() {
                    hessian[row][col] =
                        self.derivator
                            .get_double_partial(function, &[row, col], point)?;
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
