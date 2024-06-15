use crate::numerical_derivative::single_derivative as single_derivative;
use crate::numerical_derivative::mode as mode;
use num_complex::ComplexFloat;

#[derive(Debug)]
pub struct LinearApproximationResult<T: ComplexFloat, const NUM_VARS: usize>
{
    pub intercept: T,
    pub coefficients: [T; NUM_VARS]
}

#[derive(Debug)]
pub struct LinearApproximationPredictionMetrics<T: ComplexFloat>
{
    pub mean_absolute_error: T::Real,
    pub mean_squared_error: T::Real,
    pub root_mean_squared_error: T::Real,
    pub r_squared: T::Real,
    pub adjusted_r_squared: T::Real
}

impl<T: ComplexFloat, const NUM_VARS: usize> LinearApproximationResult<T, NUM_VARS>
{
    ///Helper function if you don't care about the details and just want the predictor directly
    pub fn get_prediction_value(&self, args: &[T; NUM_VARS]) -> T
    {
        let mut result = self.intercept;
        for (iter, arg) in args.iter().enumerate().take(NUM_VARS)
        {
            result = result + self.coefficients[iter]**arg;    
        }
        
        return result;
    }

    //get prediction metrics by feeding a list of points and the original function

    pub fn get_prediction_metrics<const NUM_POINTS: usize>(&self, points: &[[T; NUM_VARS]; NUM_POINTS], original_function: &dyn Fn(&[T; NUM_VARS]) -> T) -> LinearApproximationPredictionMetrics<T>
    {
        //let num_points = NUM_POINTS as f64;
        let mut mae = T::zero();
        let mut mse = T::zero();
        
        for point in points.iter().take(NUM_POINTS)
        {
            let predicted_y = self.get_prediction_value(point);
            
            mae = mae + (predicted_y - original_function(point));
            mse = mse + num_complex::ComplexFloat::powi(predicted_y - original_function(point), 2);
        }

        mae = mae/T::from(NUM_POINTS).unwrap();
        mse = mse/T::from(NUM_POINTS).unwrap();

        let rmse = mse.sqrt().abs();

        let mut r2_numerator = T::zero();
        let mut r2_denominator = T::zero();

        for point in points.iter().take(NUM_POINTS)
        {
            let predicted_y = self.get_prediction_value(point);

            r2_numerator = r2_numerator + num_complex::ComplexFloat::powi(predicted_y - original_function(point), 2);
            r2_denominator = r2_numerator + num_complex::ComplexFloat::powi(mae - original_function(point), 2);
        }

        let r2 = T::one() - (r2_numerator/r2_denominator);

        let r2_adj = T::one() - (T::one() - r2)*(T::from(NUM_POINTS).unwrap())/(T::from(NUM_POINTS).unwrap() - T::from(2.0).unwrap());

        return LinearApproximationPredictionMetrics
        {
            mean_absolute_error: mae.abs(),
            mean_squared_error: mse.abs(),
            root_mean_squared_error: rmse,
            r_squared: r2.abs(),
            adjusted_r_squared: r2_adj.abs()
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
///let function_to_approximate = | args: &[f64; 3] | -> f64
///{ 
///    return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
///};
///
///let point = [1.0, 2.0, 3.0]; //the point we want to linearize around
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
pub fn get<T: ComplexFloat, const NUM_VARS: usize>(function: &dyn Fn(&[T; NUM_VARS]) -> T, point: &[T; NUM_VARS]) -> LinearApproximationResult<T, NUM_VARS>
{
    return get_custom(function, point, 0.00001, mode::DiffMode::CentralFixedStep);
}


///same as [`get`], but for advanced users who want to control the differentiation parameters
pub fn get_custom<T: ComplexFloat, const NUM_VARS: usize>(function: &dyn Fn(&[T; NUM_VARS]) -> T, point: &[T; NUM_VARS], step_size: f64, mode: mode::DiffMode) -> LinearApproximationResult<T, NUM_VARS>
{
    let mut slopes_ = [T::zero(); NUM_VARS];

    let mut intercept_ = function(point);

    for iter in 0..NUM_VARS
    {
        slopes_[iter] = single_derivative::get_partial_custom(function, iter, point, step_size, mode);
        intercept_ = intercept_ - slopes_[iter]*point[iter];
    }

    return LinearApproximationResult
    {
        intercept: intercept_,
        coefficients: slopes_
    };
}