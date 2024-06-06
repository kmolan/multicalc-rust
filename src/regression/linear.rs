use crate::math::mean as mean;

pub struct FitResult
{
    //slope and intercept are the regression results
    pub slope: f64,
    pub intercept: f64,

    //below are all the prediction accuracy metrics on the training data
    pub mean_absolute_error: f64,
    pub mean_squared_error: f64,
    pub root_mean_squared_error: f64,
    pub r_squared: f64,
    pub adjusted_r_squared: f64
}

impl FitResult
{
    pub fn predict(&self, point: f64) -> f64
    {
        return self.slope*point + self.intercept;
    }
}


/// Given a set of X and Y points fit them into the form Y = slope*X + intercept
///
/// You can also inspect the results of the approximation. For an n-dimensional approximation, the equation is linearized as
/// 
/// [`FitResult::intercept`] gives you the required intercept
/// [`FitResult::slope`] gives you the slope
/// 
/// If you don't care about the fit coefficients and want to use the predictor directly, use [`FitResult::predict`]
/// 
/// To inspect the regression metrics use:
/// [`FitResult::mean_absolute_error`]
/// [`FitResult::mean_squared_error`]
/// [`FitResult::root_mean_squared_error`]
/// [`FitResult::r_squared`]
/// [`FitResult::adjusted_r_squared`]
///
pub fn fit(x: &Vec<f64>, y: &Vec<f64>)-> FitResult
{
    assert!(x.len() > 0 && y.len() > 0, "points cannot be empty");

    assert!(x.len() == y.len(), "points have to be the same length");

    let num_points = x.len() as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let x_mean = mean::get(&x);
    let y_mean = mean::get(&y);

    for iter in 0..num_points as usize
    {
        numerator += (x[iter] - x_mean) * (y[iter] - y_mean);
        denominator += f64::powf(x[iter] - x_mean, 2.0);
    }

    let _slope = numerator / denominator;
    let _intercept = y_mean - _slope * x_mean;


    //now compute all the accuracy metrics

    let mut mae = 0.0;
    let mut mse = 0.0;
    
    for iter in 0..num_points as usize
    {
        let predicted_y = _slope*x[iter] + _intercept;
        
        mae += f64::abs(predicted_y - y[iter]);
        mse += f64::powf(predicted_y - y[iter], 2.0);
    }

    mae = mae/num_points;
    mse = mse/num_points;

    let rmse = mse.sqrt();

    let mut r2_numerator = 0.0;
    let mut r2_denominator = 0.0;

    for iter in 0..num_points as usize
    {
        let predicted_y = _slope*x[iter] + _intercept;

        r2_numerator += f64::powf(predicted_y - y[iter], 2.0);
        r2_denominator += f64::powf(mae - y[iter], 2.0);
    }

    let r2 = 1.0 - r2_numerator/r2_denominator;

    let r2_adj = 1.0 - (1.0 - r2)*(num_points)/(num_points-2.0);

    let result = FitResult
    {
        slope: _slope, 
        intercept: _intercept,
        mean_absolute_error: mae,
        mean_squared_error: mse,
        root_mean_squared_error: rmse,
        r_squared: r2,
        adjusted_r_squared: r2_adj
    };

    return result;
}