use crate::numerical_derivative::mode::*;
use crate::numerical_derivative::derivator::Derivator;
use crate::numerical_derivative::fixed_step::FixedStep;
use crate::utils::error_codes::ErrorCode;
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

pub struct LinearApproximator
{
    derivator: FixedStep
}

impl Default for LinearApproximator
{
    fn default() -> Self 
    {
        return LinearApproximator { derivator: FixedStep::default() };    
    }
}

impl LinearApproximator
{
    pub fn set_step_size(&mut self, step_size: f64)
    {
        self.derivator.set_step_size(step_size);
    }

    pub fn get_step_size(&self) -> f64
    {
        return self.derivator.get_step_size();
    }

    pub fn set_derivative_method(&mut self, method: FixedStepMode)
    {
        self.derivator.set_method(method);
    }

    pub fn get_derivative_method(&self) -> FixedStepMode
    {
        return self.derivator.get_method();
    }

    pub fn from_parameters(step_size: f64, method: FixedStepMode) -> Self
    {
        return LinearApproximator { derivator: FixedStep::from_parameters(step_size, method) };
    }

    pub fn from_derivator(derivator: FixedStep) -> Self
    {
        return LinearApproximator {derivator: derivator}
    }

    /// For an n-dimensional approximation, the equation is linearized as:
    /// coefficient[0]*var_1 + coefficient[1]*var_2 + ... + coefficient[n-1]*var_n + intercept
    /// 
    /// NOTE: Returns a Result<T, ErrorCode>
    /// Possible ErrorCode are:
    /// NumberOfStepsCannotBeZero -> if the derivative step size is zero
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
    pub fn get<T: ComplexFloat, const NUM_VARS: usize>(&self, function: &dyn Fn(&[T; NUM_VARS]) -> T, point: &[T; NUM_VARS]) -> Result<LinearApproximationResult<T, NUM_VARS>, ErrorCode>
    {
        let mut slopes_ = [T::zero(); NUM_VARS];

        let mut intercept_ = function(point);

        for iter in 0..NUM_VARS
        {
            slopes_[iter] = self.derivator.get_single_partial(function, iter, point)?;
            intercept_ = intercept_ - slopes_[iter]*point[iter];
        }

        return Ok(LinearApproximationResult
        {
            intercept: intercept_,
            coefficients: slopes_
        });
    }
}