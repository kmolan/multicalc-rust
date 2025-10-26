use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use const_poly::function_approximations;


#[derive(Debug)]
pub struct LinearApproximationResult<const NUM_VARS: usize> {
    pub intercept: f64,
    pub coefficients: [f64; NUM_VARS],
}

#[derive(Debug)]
pub struct LinearApproximationPredictionMetrics {
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
}

impl<const NUM_VARS: usize> LinearApproximationResult<NUM_VARS> {
    ///Helper function if you don't care about the details and just want the predictor directly
    pub fn get_prediction_value(&self, args: &[f64; NUM_VARS]) -> f64 {
        let mut result = self.intercept;
        for (iter, arg) in args.iter().enumerate().take(NUM_VARS) {
            result = result + self.coefficients[iter] * *arg;
        }

        return result;
    }

    //get prediction metrics by feeding a list of points and the original function
    pub fn get_prediction_metrics<const NUM_POINTS: usize>(
        &self,
        points: &[[f64; NUM_VARS]; NUM_POINTS],
        original_function: &dyn Fn(&[f64; NUM_VARS]) -> f64,
    ) -> LinearApproximationPredictionMetrics {
        //let num_points = NUM_POINTS as f64;
        let mut mae = 0.0;
        let mut mse = 0.0;

        for point in points.iter().take(NUM_POINTS) {
            let predicted_y = self.get_prediction_value(point);

            mae = mae + (predicted_y - original_function(point));
            mse = mse + function_approximations::static_powi(predicted_y - original_function(point), 2);
        }

        mae = mae / (NUM_POINTS as f64);
        mse = mse / (NUM_POINTS as f64);

        let rmse = function_approximations::sqrt_approx(mse).abs();

        let mut r2_numerator = 0.0;
        let mut r2_denominator = 0.0;

        for point in points.iter().take(NUM_POINTS) {
            let predicted_y = self.get_prediction_value(point);

            r2_numerator = r2_numerator
                + function_approximations::static_powi(predicted_y - original_function(point), 2);
            r2_denominator =
                r2_numerator + function_approximations::static_powi(mae - original_function(point), 2);
        }

        let r2 = 1.0 - (r2_numerator / r2_denominator);

        let r2_adj = 1.0
            - (1.0 - r2) * ((NUM_POINTS as f64))
                / ((NUM_POINTS as f64) - 2.0);

        return LinearApproximationPredictionMetrics {
            mean_absolute_error: mae.abs(),
            mean_squared_error: mse.abs(),
            root_mean_squared_error: rmse,
            r_squared: r2.abs(),
            adjusted_r_squared: r2_adj.abs(),
        };
    }
}

pub struct LinearApproximator<D: DerivatorMultiVariable> {
    derivator: D,
}

impl<D: DerivatorMultiVariable> Default for LinearApproximator<D> {
    fn default() -> Self {
        return LinearApproximator {
            derivator: D::default(),
        };
    }
}

impl<D: DerivatorMultiVariable> LinearApproximator<D> {
    pub fn from_derivator(derivator: D) -> Self {
        return LinearApproximator { derivator };
    }

    /// For an n-dimensional approximation, the equation is linearized as:
    /// coefficient[0]*var_1 + coefficient[1]*var_2 + ... + coefficient[n-1]*var_n + intercept
    ///
    /// NOTE: Returns a Result<T, &'static str>
    /// Possible &'static str are:
    /// NumberOfStepsCannotBeZero -> if the derivative step size is zero
    ///
    ///example function is x + y^2 + z^3, which we want to linearize. First define the function:
    ///```
    ///use multicalc::approximation::linear_approximation::*;
    ///use multicalc::numerical_derivative::finite_difference::MultiVariableSolver;
    ///
    ///let function_to_approximate = | args: &[f64; 3] | -> f64
    ///{
    ///    return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
    ///};
    ///
    ///let point = [1.0, 2.0, 3.0]; //the point we want to linearize around
    ///let approximator = LinearApproximator::<MultiVariableSolver>::default();
    ///let result = approximator.get(&function_to_approximate, &point).unwrap();
    ///
    ///assert!(f64::abs(function_to_approximate(&point) - result.get_prediction_value(&point)) < 1e-9);
    /// ```
    /// you can also inspect the results of the approximation. For an n-dimensional approximation, the equation is linearized as
    ///
    /// [`LinearApproximationResult::intercept`] gives you the required intercept
    /// [`LinearApproximationResult::coefficients`] gives you the required coefficients in order
    ///
    /// if you don't care about the results and want the predictor directly, use [`LinearApproximationResult::get_prediction_value()`]
    /// you can also inspect the prediction metrics by providing list of points, use [`LinearApproximationResult::get_prediction_metrics()`]
    ///
    pub fn get<const NUM_VARS: usize>(
        &self,
        function: &dyn Fn(&[f64; NUM_VARS]) -> f64,
        point: &[f64; NUM_VARS],
    ) -> Result<LinearApproximationResult<NUM_VARS>, &'static str> {
        let mut slopes_ = [0.0; NUM_VARS];

        let mut intercept_ = function(point);

        for iter in 0..NUM_VARS {
            slopes_[iter] = self.derivator.get(1, function, &[iter], point)?;
            intercept_ = intercept_ - slopes_[iter] * point[iter];
        }

        return Ok(LinearApproximationResult {
            intercept: intercept_,
            coefficients: slopes_,
        });
    }
}
