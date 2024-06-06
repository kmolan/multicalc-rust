use nalgebra as na;

pub struct FitResult
{
    //coefficients for the fit, the degree of coefficient increases with its index
    pub coefficients: Vec<f64>,
    
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
        let mut result = 0.0;

        for iter in 0..self.coefficients.len()
        {
            result += self.coefficients[iter]*f64::powf(point, iter as f64);
        }

        return result;
    }
}

/// Given a set of X and Y points fit them into a polynomial equation. The degree of polynomial equation is input by the user
///
/// The regression results are in [`FitResult::coefficients`], which gives coefficients for the fit, such that the degree of coefficient increases with its index
/// If you don't care about the fit coefficients and want to use the predictor directly, use [`FitResult::predict`]
/// 
/// To inspect the regression metrics use:
/// [`FitResult::mean_absolute_error`]
/// [`FitResult::mean_squared_error`]
/// [`FitResult::root_mean_squared_error`]
/// [`FitResult::r_squared`]
/// [`FitResult::adjusted_r_squared`]
///
pub fn fit(x: Vec<f64>, y: Vec<f64>, polynomial_degree: usize) -> FitResult 
{
    assert!(x.len() > 0 && y.len() > 0, "points cannot be empty");

    assert!(x.len() == y.len(), "points have to be the same length");

    assert!(polynomial_degree > 0, "polynomial degree cannot be zero");

    let number_of_columns = polynomial_degree + 1;
    let number_of_rows = x.len();
    let mut x_matrix = na::DMatrix::zeros(number_of_rows, number_of_columns);

    for row in 0..number_of_rows
    {
        //first column is always 1.0
        x_matrix[(row, 0)] = 1.0;

        for column in 1..number_of_columns 
        {
            x_matrix[(row, column)] = x[row].powi(column as i32);
        }
    }

    let y_matrix = na::DVector::from_row_slice(&y);

    let svd_matrix = na::SVD::new(x_matrix, true, true);

    let coeff = svd_matrix.solve(&y_matrix, f64::EPSILON).unwrap().data.into();

    //now compute all the accuracy metrics

    let mut mae = 0.0;
    let mut mse = 0.0;
    let num_points = x.len() as f64;
    
    for iter in 0..num_points as usize
    {
        let predicted_y = get_predicted_y(x[iter], &coeff);
        
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
        let predicted_y = get_predicted_y(x[iter], &coeff);

        r2_numerator += f64::powf(predicted_y - y[iter], 2.0);
        r2_denominator += f64::powf(mae - y[iter], 2.0);
    }

    let r2 = 1.0 - r2_numerator/r2_denominator;
    let r2_adj = 1.0 - (1.0 - r2)*(num_points)/(num_points-2.0);


    let result = FitResult
    {
        coefficients: coeff,
        mean_absolute_error: mae,
        mean_squared_error: mse,
        root_mean_squared_error: rmse,
        r_squared: r2,
        adjusted_r_squared: r2_adj
    };

    return result;
}


fn get_predicted_y(point: f64, coeff: &Vec<f64>) -> f64
{
    let mut result = 0.0;

    for iter in 0..coeff.len()
    {
        result += coeff[iter]*f64::powf(point, iter as f64);
    }

    return result;
}