use crate::numerical_derivative::mode as mode;
use crate::numerical_derivative::single_derivative as single_derivative;
use crate::numerical_derivative::hessian as hessian;

#[derive(Debug)]
pub struct QuadraticApproximationResult
{
    pub intercept: f64,
    pub linear_coefficients: Vec<f64>,
    pub quadratic_coefficients: Vec<Vec<f64>>
}

#[derive(Debug)]
pub struct QuadraticApproximationPredictionMetrics
{
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub r_squared: f64,
    pub adjusted_r_squared: f64
}

///Helper function if you don't care about the details and just want the predictor directly
impl QuadraticApproximationResult
{
    pub fn get_prediction_value(&self, args: &Vec<f64>) -> f64
    {
        let mut result = self.intercept;

        for i in 0..args.len()
        {
            result += self.linear_coefficients[i]*args[i];
        }
        for i in 0..args.len()
        {
            for j in 1..args.len()
            {
                result += self.quadratic_coefficients[i][j]*args[i]*args[j];    
            }
        }
        
        return result;
    }

    //get prediction metrics by feeding a list of points and the original function
    pub fn get_prediction_metrics(&self, points: &Vec<Vec<f64>>, original_function: &dyn Fn(&Vec<f64>) -> f64) -> QuadraticApproximationPredictionMetrics
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

        return QuadraticApproximationPredictionMetrics
        {
            mean_absolute_error: mae,
            mean_squared_error: mse,
            root_mean_squared_error: rmse,
            r_squared: r2,
            adjusted_r_squared: r2_adj
        };
    }
}


/// For an n-dimensional approximation, the equation is approximated as I + L + Q, where:
/// I = intercept 
/// L = linear_coefficients[0]*var_1 + linear_coefficients[1]*var_2 + ... + linear_coefficients[n-1]*var_n
/// Q = quadratic_coefficients[0][0]*var_1*var_1 + quadratic_coefficients[0][1]*var_1*var_2 + ... + quadratic_coefficients[n-1][n-1]*var_n*var_n
///
///example function is e^(x/2) + sin(y) + 2.0*z, which we want to approximate. First define the function:
///```
///use multicalc::approximation::quadratic_approximation;
/// 
///let function_to_approximate = | args: &Vec<f64> | -> f64
///{ 
///    return f64::exp(args[0]/2.0) + f64::sin(args[1]) + 2.0*args[2];
///};
///
///let point = vec![0.0, 1.57, 10.0]; //the point we want to approximate around
///
///let result = quadratic_approximation::get(&function_to_approximate, &point);
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
pub fn get(function: &dyn Fn(&Vec<f64>) -> f64, point: &Vec<f64>) -> QuadraticApproximationResult
{
    return get_custom(function, point, 0.0001, mode::DiffMode::CentralFixedStep);
}


///same as [`get`], but for advanced users who want to control the differentiation parameters
pub fn get_custom(function: &dyn Fn(&Vec<f64>) -> f64, point: &Vec<f64>, step_size: f64, mode: mode::DiffMode) -> QuadraticApproximationResult
{
    let mut intercept_ = function(point);

    let mut linear_coeffs_ = vec![0.0; point.len()];

    let hessian_matrix = hessian::get_custom(function, point, step_size, mode);

    for iter in 0..point.len()
    {
        linear_coeffs_[iter] = single_derivative::get_partial_custom(function, iter, point, step_size, mode);
        intercept_ -= single_derivative::get_partial_custom(function, iter, point, step_size, mode)*point[iter];
    }

    let mut quad_coeff = vec![vec![0.0; hessian_matrix.len()]; hessian_matrix.len()];

    for row in 0..point.len()
    {
        for col in row..point.len()
        {
            quad_coeff[row][col] = hessian_matrix[row][col];

            intercept_ += hessian_matrix[row][col]*point[row]*point[row];

            if row == col
            {
                linear_coeffs_[row] -= 2.0*hessian_matrix[row][col]*point[row]
            }
            else 
            {
                linear_coeffs_[row] -= hessian_matrix[row][col]*point[col];
                linear_coeffs_[col] -= hessian_matrix[row][col]*point[row];
            }
        }
    }

    return QuadraticApproximationResult
    {
        intercept: intercept_,
        linear_coefficients: linear_coeffs_,
        quadratic_coefficients: quad_coeff
    };
}
