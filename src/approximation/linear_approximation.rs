use crate::numerical_derivative::single_derivative as single_derivative;
use crate::numerical_derivative::mode as mode;

#[derive(Debug)]
pub struct LinearApproximationResult
{
    pub intercept: f64,
    pub coefficients: Vec<f64>
}

#[derive(Debug)]
pub struct LinearApproximationPredictionMetrics
{
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub r_squared: f64,
    pub adjusted_r_squared: f64
}

impl LinearApproximationResult
{
    ///Helper function if you don't care about the details and just want the predictor directly
    pub fn get_prediction_value(&self, args: &Vec<f64>) -> f64
    {
        let mut result = self.intercept;
        for iter in 0..args.len()
        {
            result += self.coefficients[iter]*args[iter];    
        }
        
        return result;
    }

    //get prediction metrics by feeding a list of points and the original function
    pub fn get_prediction_metrics(&self, points: &Vec<Vec<f64>>, original_function: &dyn Fn(&Vec<f64>) -> f64) -> LinearApproximationPredictionMetrics
    {
        let num_points = points.len() as f64;
        let mut mae = 0.0;
        let mut mse = 0.0;
        
        for iter in 0..num_points as usize
        {
            let predicted_y = self.get_prediction_value(&points[iter]);
            
            mae += f64::abs(predicted_y - original_function(&points[iter]));
            mse += f64::powf(predicted_y - original_function(&points[iter]), 2.0);
        }

        mae = mae/num_points;
        mse = mse/num_points;

        let rmse = mse.sqrt();

        let mut r2_numerator = 0.0;
        let mut r2_denominator = 0.0;

        for iter in 0..num_points as usize
        {
            let predicted_y = self.get_prediction_value(&points[iter]);

            r2_numerator += f64::powf(predicted_y - original_function(&points[iter]), 2.0);
            r2_denominator += f64::powf(mae - original_function(&points[iter]), 2.0);
        }

        let r2 = 1.0 - r2_numerator/r2_denominator;

        let r2_adj = 1.0 - (1.0 - r2)*(num_points)/(num_points-2.0);

        return LinearApproximationPredictionMetrics
        {
            mean_absolute_error: mae,
            mean_squared_error: mse,
            root_mean_squared_error: rmse,
            r_squared: r2,
            adjusted_r_squared: r2_adj
        };
    }
}


/// For an n-dimensional approximation, the equation is linearized as:
/// coefficient[0]*var_1 + coefficient[1]*var_2 + ... + coefficient[n-1]*var_n + intercept
///
///example function is x + y^2 + z^3, which we want to linearize. First define the function:
///```
///use multicalc::approximation::linear_approximation;
/// 
///let function_to_approximate = | args: &Vec<f64> | -> f64
///{ 
///    return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
///};
///
///let point = vec![1.0, 2.0, 3.0]; //the point we want to linearize around
///
///let result = linear_approximation::get(&function_to_approximate, &point);
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
pub fn get(function: &dyn Fn(&Vec<f64>) -> f64, point: &Vec<f64>) -> LinearApproximationResult
{
    return get_custom(function, point, 0.00001, &mode::DiffMode::CentralFixedStep);
}


///same as [`get`], but for advanced users who want to control the differentiation parameters
pub fn get_custom(function: &dyn Fn(&Vec<f64>) -> f64, point: &Vec<f64>, step_size: f64, mode: &mode::DiffMode) -> LinearApproximationResult
{
    let mut slopes_ = vec![0.0; point.len()];

    let mut intercept_ = function(point);

    for iter in 0..point.len()
    {
        slopes_[iter] = single_derivative::get_partial_custom(function, iter, point, step_size, mode);
        intercept_ -= slopes_[iter]*point[iter];
    }

    return LinearApproximationResult
    {
        intercept: intercept_,
        coefficients: slopes_
    };
}