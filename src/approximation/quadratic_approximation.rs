use crate::numerical_derivative::finite_difference::MultiVariableSolver;
use const_poly::Polynomial;
use crate::numerical_derivative::hessian::Hessian;
use crate::utils::helper;

#[derive(Debug)]
pub struct QuadraticApproximationResult<const NUM_VARS: usize> {
    pub intercept: f64,
    pub linear_coefficients: [f64; NUM_VARS],
    pub quadratic_coefficients: [[f64; NUM_VARS]; NUM_VARS],
}

#[derive(Debug)]
pub struct QuadraticApproximationPredictionMetrics {
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
}

///Helper functions if you don't care about the details and just want the predictor directly
impl<const NUM_VARS: usize> QuadraticApproximationResult<NUM_VARS> {
    pub fn get_prediction_value(&self, args: &[f64; NUM_VARS]) -> f64 {
        let mut result = self.intercept;

        for (i, arg) in args.iter().enumerate().take(NUM_VARS) {
            result = result + self.linear_coefficients[i] * *arg;
        }
        for i in 0..NUM_VARS {
            for j in 1..NUM_VARS {
                result = result + self.quadratic_coefficients[i][j] * args[i] * args[j];
            }
        }

        return result;
    }

    //get prediction metrics by feeding a list of points and the original function
    pub fn get_prediction_metrics<const NUM_POINTS: usize>(
        &self,
        points: &[[f64; NUM_VARS]; NUM_POINTS],
        original_function: &Polynomial<NUM_VARS>,
    ) -> QuadraticApproximationPredictionMetrics {
        //let num_points = points.len() as f64;
        let mut mae = 0.0;
        let mut mse = 0.0;

        for point in points.iter().take(NUM_POINTS) {
            let predicted_y = self.get_prediction_value(point);

            mae = mae + (predicted_y - original_function.evaluate(point));
            mse = mse + helper::powi(predicted_y - original_function.evaluate(point), 2);
        }

        mae = mae / (NUM_POINTS as f64);
        mse = mse / (NUM_POINTS as f64);

        let rmse = helper::sqrt(mse).abs();

        let mut r2_numerator = 0.0;
        let mut r2_denominator = 0.0;

        for point in points.iter().take(NUM_POINTS) {
            let predicted_y = self.get_prediction_value(point);

            r2_numerator = r2_numerator
                + helper::powi(predicted_y - original_function.evaluate(point), 2);
            r2_denominator =
                r2_numerator + helper::powi(mae - original_function.evaluate(point), 2);
        }

        let r2 = 1.0 - (r2_numerator / r2_denominator);

        let r2_adj = 1.0
            - (1.0 - r2) * ((NUM_POINTS as f64))
                / ((NUM_POINTS as f64) - 2.0);

        return QuadraticApproximationPredictionMetrics {
            mean_absolute_error: mae.abs(),
            mean_squared_error: mse.abs(),
            root_mean_squared_error: rmse,
            r_squared: r2.abs(),
            adjusted_r_squared: r2_adj.abs(),
        };
    }
}

pub struct QuadraticApproximator {
    derivator: MultiVariableSolver,
}

impl Default for QuadraticApproximator {
    fn default() -> Self {
        Self {
            derivator: MultiVariableSolver::default(),
        }
    }
}

impl QuadraticApproximator {

    /// For an n-dimensional approximation, the equation is approximated as I + L + Q, where:
    /// I = intercept
    /// L = linear_coefficients[0]*var_1 + linear_coefficients[1]*var_2 + ... + linear_coefficients[n-1]*var_n
    /// Q = quadratic_coefficients[0][0]*var_1*var_1 + quadratic_coefficients[0][1]*var_1*var_2 + ... + quadratic_coefficients[n-1][n-1]*var_n*var_n
    ///
    /// NOTE: Returns a Result<T, &'static str>
    /// Possible &'static str are:
    /// NumberOfStepsCannotBeZero -> if the derivative step size is zero
    ///
    ///example function is e^(x/2) + sin(y) + 2.0*z, which we want to approximate. First define the function:
    ///```
    ///use multicalc::approximation::quadratic_approximation::*;
    ///use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;
    ///
    ///let function_to_approximate = | args: &[f64; 3] | -> f64
    ///{
    ///    return f64::exp(args[0]/2.0) + f64::sin(args[1]) + 2.0*args[2];
    ///};
    ///
    ///let point = [0.0, 1.57, 10.0]; //the point we want to approximate around
    ///
    ///let approximator = QuadraticApproximator::<MultiVariableSolver>::default();
    ///let result = approximator.get(&function_to_approximate, &point).unwrap();
    ///
    ///assert!(f64::abs(function_to_approximate(&point) - result.get_prediction_value(&point)) < 1e-9);
    /// ```
    /// you can also inspect the results of the approximation. For an n-dimensional approximation, the equation is linearized as
    ///
    /// [`QuadraticApproximationResult::intercept`] gives you the required intercept
    /// [`QuadraticApproximationResult::linear_coefficients`] gives you the required linear coefficients in order
    /// [`QuadraticApproximationResult::quadratic_coefficients`] gives you the required quadratic coefficients as a matrix
    ///
    /// if you don't care about the results and want the predictor directly, use [`QuadraticApproximationResult::get_prediction_value()`]
    /// you can also inspect the prediction metrics by providing list of points, use [`QuadraticApproximationResult::get_prediction_metrics()`]
    ///
    /// to see how the [QuadraticApproximationResult::quadratic_coefficients] matrix should be used, refer to [`QuadraticApproximationResult::get_prediction_metrics()`]
    /// or refer to its tests.
    ///
    pub fn get<const NUM_VARS: usize>(
        &self,
        function: &Polynomial<NUM_VARS>,
        point: &[f64; NUM_VARS],
    ) -> Result<QuadraticApproximationResult<NUM_VARS>, &'static str> {
        let mut intercept_ = function.evaluate(point);

        let mut linear_coeffs_ = [0.0; NUM_VARS];

        let hessian_matrix = Hessian::default().get(function, point)?;

        for iter in 0..NUM_VARS {
            linear_coeffs_[iter] = self.derivator.get(1, function, &[iter], point)?;
            intercept_ =
                intercept_ - self.derivator.get(1, function, &[iter], point)? * point[iter];
        }

        let mut quad_coeff = [[0.0; NUM_VARS]; NUM_VARS];

        for row in 0..NUM_VARS {
            for col in row..NUM_VARS {
                quad_coeff[row][col] = hessian_matrix[row][col];

                intercept_ = intercept_ + hessian_matrix[row][col] * point[row] * point[row];

                if row == col {
                    linear_coeffs_[row] = linear_coeffs_[row]
                        - 2.0 * hessian_matrix[row][col] * point[row]
                } else {
                    linear_coeffs_[row] =
                        linear_coeffs_[row] - hessian_matrix[row][col] * point[col];
                    linear_coeffs_[col] =
                        linear_coeffs_[col] - hessian_matrix[row][col] * point[row];
                }
            }
        }

        return Ok(QuadraticApproximationResult {
            intercept: intercept_,
            linear_coefficients: linear_coeffs_,
            quadratic_coefficients: quad_coeff,
        });
    }
}
