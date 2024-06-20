use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::numerical_derivative::hessian::Hessian;
use crate::utils::error_codes::ErrorCode;
use num_complex::ComplexFloat;

#[derive(Debug)]
pub struct QuadraticApproximationResult<T: ComplexFloat, const NUM_VARS: usize>
{
    pub intercept: T,
    pub linear_coefficients: [T; NUM_VARS],
    pub quadratic_coefficients: [[T; NUM_VARS]; NUM_VARS]
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

///Helper functions if you don't care about the details and just want the predictor directly
impl<T: ComplexFloat, const NUM_VARS: usize> QuadraticApproximationResult<T, NUM_VARS>
{
    pub fn get_prediction_value(&self, args: &[T; NUM_VARS]) -> T
    {
        let mut result = self.intercept;

        for (i, arg) in args.iter().enumerate().take(NUM_VARS)
        {
            result = result + self.linear_coefficients[i]**arg;
        }
        for i in 0..NUM_VARS
        {
            for j in 1..NUM_VARS
            {
                result = result + self.quadratic_coefficients[i][j]*args[i]*args[j];    
            }
        }
        
        return result;
    }

    //get prediction metrics by feeding a list of points and the original function
    pub fn get_prediction_metrics<const NUM_POINTS: usize>(&self, points: &[[T; NUM_VARS]; NUM_POINTS], original_function: &dyn Fn(&[T; NUM_VARS]) -> T) -> QuadraticApproximationPredictionMetrics<T>
    {
        //let num_points = points.len() as f64;
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

pub struct QuadraticApproximator<D: DerivatorMultiVariable>
{
    derivator: D,
}

impl<D: DerivatorMultiVariable> Default for QuadraticApproximator<D>
{
    fn default() -> Self 
    {
        return QuadraticApproximator { derivator: D::default()};    
    }
}

impl<D: DerivatorMultiVariable> QuadraticApproximator<D>
{
    pub fn from_derivator(derivator: D) -> Self
    {
        return QuadraticApproximator {derivator: derivator};
    }

    /// For an n-dimensional approximation, the equation is approximated as I + L + Q, where:
    /// I = intercept 
    /// L = linear_coefficients[0]*var_1 + linear_coefficients[1]*var_2 + ... + linear_coefficients[n-1]*var_n
    /// Q = quadratic_coefficients[0][0]*var_1*var_1 + quadratic_coefficients[0][1]*var_1*var_2 + ... + quadratic_coefficients[n-1][n-1]*var_n*var_n
    ///
    /// NOTE: Returns a Result<T, ErrorCode>
    /// Possible ErrorCode are:
    /// NumberOfStepsCannotBeZero -> if the derivative step size is zero
    /// 
    ///example function is e^(x/2) + sin(y) + 2.0*z, which we want to approximate. First define the function:
    ///```
    ///use multicalc::approximation::quadratic_approximation;
    /// 
    ///let function_to_approximate = | args: &[f64; 3] | -> f64
    ///{ 
    ///    return f64::exp(args[0]/2.0) + f64::sin(args[1]) + 2.0*args[2];
    ///};
    ///
    ///let point = [0.0, 1.57, 10.0]; //the point we want to approximate around
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
    pub fn get<T: ComplexFloat, const NUM_VARS: usize>(&self, function: &dyn Fn(&[T; NUM_VARS]) -> T, point: &[T; NUM_VARS]) -> Result<QuadraticApproximationResult<T, NUM_VARS>, ErrorCode>
    {
        let mut intercept_ = function(point);

        let mut linear_coeffs_ = [T::zero(); NUM_VARS];

        let hessian_matrix = Hessian::from_derivator(self.derivator).get(function, point)?;

        for iter in 0..NUM_VARS
        {
            linear_coeffs_[iter] = self.derivator.get(1, function, &[iter], point)?;
            intercept_ = intercept_ - self.derivator.get(1, function, &[iter], point)?*point[iter];
        }

        let mut quad_coeff = [[T::zero(); NUM_VARS]; NUM_VARS];

        for row in 0..NUM_VARS
        {
            for col in row..NUM_VARS
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

        return Ok(QuadraticApproximationResult
        {
            intercept: intercept_,
            linear_coefficients: linear_coeffs_,
            quadratic_coefficients: quad_coeff
        });
    }
}