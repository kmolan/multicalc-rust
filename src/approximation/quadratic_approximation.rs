use crate::numerical_derivative::mode as mode;
use crate::numerical_derivative::single_derivative as single_derivative;
use crate::numerical_derivative::hessian as hessian;
use num_complex::ComplexFloat;

#[derive(Debug)]
pub struct QuadraticApproximationResult<T: ComplexFloat>
{
    pub intercept: T,
    pub linear_coefficients: Vec<T>,
    pub quadratic_coefficients: Vec<Vec<T>>
}

#[derive(Debug)]
pub struct QuadraticApproximationPredictionMetrics<T: ComplexFloat>
{
    pub mean_absolute_error: T::Real,
    pub mean_squared_error: T::Real,
    pub root_mean_squared_error: T::Real,
    pub r_squared: T::Real,
    pub adjusted_r_squared: T::Real
}

///Helper function if you don't care about the details and just want the predictor directly
impl<T: ComplexFloat> QuadraticApproximationResult<T>
{
    pub fn get_prediction_value(&self, args: &Vec<T>) -> T
    {
        let mut result = self.intercept;

        for i in 0..args.len()
        {
            result = result + self.linear_coefficients[i]*args[i];
        }
        for i in 0..args.len()
        {
            for j in 1..args.len()
            {
                result = result + self.quadratic_coefficients[i][j]*args[i]*args[j];    
            }
        }
        
        return result;
    }

    //get prediction metrics by feeding a list of points and the original function
    pub fn get_prediction_metrics(&self, points: &Vec<Vec<T>>, original_function: &dyn Fn(&Vec<T>) -> T) -> QuadraticApproximationPredictionMetrics<T>
    {
        let num_points = points.len() as f64;
        let mut mae = T::zero();
        let mut mse = T::zero();
        
        for iter in 0..num_points as usize
        {
            let predicted_y = self.get_prediction_value(&points[iter]);
            
            mae = mae + (predicted_y - original_function(&points[iter]));
            mse = mse + num_complex::ComplexFloat::powi(predicted_y - original_function(&points[iter]), 2);
        }

        mae = mae/T::from(num_points).unwrap();
        mse = mse/T::from(num_points).unwrap();

        let rmse = mse.sqrt().abs();

        let mut r2_numerator = T::zero();
        let mut r2_denominator = T::zero();

        for iter in 0..num_points as usize
        {
            let predicted_y = self.get_prediction_value(&points[iter]);

            r2_numerator = r2_numerator + num_complex::ComplexFloat::powi(predicted_y - original_function(&points[iter]), 2);
            r2_denominator = r2_numerator + num_complex::ComplexFloat::powi(mae - original_function(&points[iter]), 2);
        }

        let r2 = T::one() - (r2_numerator/r2_denominator);

        let r2_adj = T::one() - (T::one() - r2)*(T::from(num_points).unwrap())/(T::from(num_points).unwrap() - T::from(2.0).unwrap());

        return QuadraticApproximationPredictionMetrics
        {
            mean_absolute_error: mae.abs(),
            mean_squared_error: mse.abs(),
            root_mean_squared_error: rmse,
            r_squared: r2.abs(),
            adjusted_r_squared: r2_adj.abs()
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
pub fn get<T: ComplexFloat>(function: &dyn Fn(&Vec<T>) -> T, point: &Vec<T>) -> QuadraticApproximationResult<T>
{
    return get_custom(function, point, 0.0001, mode::DiffMode::CentralFixedStep);
}


///same as [`get`], but for advanced users who want to control the differentiation parameters
pub fn get_custom<T: ComplexFloat>(function: &dyn Fn(&Vec<T>) -> T, point: &Vec<T>, step_size: f64, mode: mode::DiffMode) -> QuadraticApproximationResult<T>
{
    let mut intercept_ = function(point);

    let mut linear_coeffs_ = vec![T::zero(); point.len()];

    let hessian_matrix = hessian::get_custom(function, point, step_size, mode);

    for iter in 0..point.len()
    {
        linear_coeffs_[iter] = single_derivative::get_partial_custom(function, iter, point, step_size, mode);
        intercept_ = intercept_ - single_derivative::get_partial_custom(function, iter, point, step_size, mode)*point[iter];
    }

    let mut quad_coeff = vec![vec![T::zero(); hessian_matrix.len()]; hessian_matrix.len()];

    for row in 0..point.len()
    {
        for col in row..point.len()
        {
            quad_coeff[row][col] = hessian_matrix[row][col];

            intercept_ = intercept_ + hessian_matrix[row][col]*point[row]*point[row];

            if row == col
            {
                linear_coeffs_[row] = linear_coeffs_[row] - T::from(2.0).unwrap()*hessian_matrix[row][col]*point[row]
            }
            else 
            {
                linear_coeffs_[row] = linear_coeffs_[row] - hessian_matrix[row][col]*point[col];
                linear_coeffs_[col] = linear_coeffs_[col] - hessian_matrix[row][col]*point[row];
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
