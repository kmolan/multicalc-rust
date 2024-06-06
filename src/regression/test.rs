use rand::Rng;
use crate::regression::linear as linear;
use crate::regression::polynomial as poly;

#[test]
fn test_linear_regression_1() 
{
    //create a list of 10,000 points with no noise, on a 2D plot this would be a straight line with slope = 1, intercept = 0
    let num_points: usize = 10_000;
    let mut x = vec![0.0; num_points];
    let mut y = vec![0.0; num_points];

    for iter in 0..num_points
    {
        x[iter] = iter as f64;
        y[iter] = x[iter];
    }

    let val = linear::fit(&x, &y) ;

    assert!(val.slope == 1.0);
    assert!(val.intercept == 0.0);

    assert!(val.root_mean_squared_error == 0.0);
    assert!(val.mean_absolute_error == 0.0);
    assert!(val.mean_squared_error == 0.0);
    assert!(val.r_squared == 1.0);
    assert!(val.adjusted_r_squared == 1.0);
}

#[test]
fn test_linear_regression_2() 
{
    //create a list of 10,000 points with noise = [0, 0.1). With no noise on a 2D plot this would be a straight line with slope = 1, intercept = 0
    let num_points: usize = 10_000;
    let mut x = vec![0.0; num_points];
    let mut y = vec![0.0; num_points];

    let mut random_generator = rand::thread_rng();

    for iter in 0..num_points
    {
        x[iter] = iter as f64;
        y[iter] = random_generator.gen_range(0.0..0.1) + x[iter];
    }

    let val = linear::fit(&x, &y);

    assert!(f64::abs(val.slope - 1.0) < 0.05);
    assert!(f64::abs(val.intercept - 0.0) < 0.1);

    assert!(val.root_mean_squared_error < 0.1);
    assert!(val.mean_absolute_error < 0.1);
    assert!(val.mean_squared_error < 0.1);
    assert!(val.r_squared > 0.99);
    assert!(val.adjusted_r_squared > 0.99);

    //expect a value of ~ -2.0
    assert!(f64::abs(val.predict(-2.0) + 2.0) < 2.0*val.root_mean_squared_error);    
}


#[test]
fn test_linear_regression_3() 
{
    //create a list of points with unequal number of points
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let result = std::panic::catch_unwind(||linear::fit(&x, &y));
    assert!(result.is_err());
}

#[test]
fn test_linear_regression_4() 
{
    //create a list of points with empty points
    let x = vec![];
    let y = vec![];

    //expect error
    let result = std::panic::catch_unwind(||linear::fit(&x, &y));
    assert!(result.is_err());    
}

#[test]
fn test_polyfit_1() 
{
    //quadratic fit with 10,000 data points with no noise
    let num_points: usize = 10_000;
    let mut x = vec![0.0; num_points];
    let mut y = vec![0.0; num_points];
    
    //the coefficients for the equation
    let coeff = vec![2.0, 3.0, -4.0];

    for iter in 0..num_points
    {
        x[iter] = iter as f64;
        y[iter] = coeff[0] + coeff[1]*x[iter] + coeff[2]*x[iter]*x[iter];
    }

    let val = poly::fit(x, y, 2);

    assert!(val.root_mean_squared_error < 0.00001);
    assert!(val.mean_absolute_error < 0.00001);
    assert!(val.mean_squared_error < 0.00001);
    assert!(val.r_squared == 1.00);
    assert!(val.adjusted_r_squared == 1.00);
}

#[test]
fn test_polyfit_2() 
{
    //quadratic fit with 10,000 data points with noise = [0, 0.1)
    let num_points: usize = 10_000;
    let mut x = vec![0.0; num_points];
    let mut y = vec![0.0; num_points];
    
    //the coefficients for the equation
    let coeff = vec![2.0, 3.0, -4.0];

    let mut random_generator = rand::thread_rng();

    for iter in 0..num_points
    {
        x[iter] = iter as f64;

        y[iter] = coeff[0] + coeff[1]*x[iter] + coeff[2]*x[iter]*x[iter];
        y[iter] += random_generator.gen_range(0.0..0.1);
    }

    let val = poly::fit(x, y, 2);
    
    assert!(val.root_mean_squared_error < 0.05);
    assert!(val.mean_absolute_error < 0.05);
    assert!(val.mean_squared_error < 0.0025);
    assert!(val.r_squared > 0.9999);
    assert!(val.adjusted_r_squared > 0.9999);

    //expect a value prediction of ~ -20
    assert!(f64::abs(val.predict(-2.0) + 20.0) < 2.0*val.root_mean_squared_error);
}

#[test]
fn test_polyfit_3() 
{
    //quintic fit with noise = [0, 0.1) with 1000 points
    let num_points: usize = 1_000;
    let mut x = vec![0.0; num_points];
    let mut y = vec![0.0; num_points];
    
    //the coefficients for the equation
    let coeff = vec![-0.85, 2.3, 3.45, -4.1, 0.0, 0.45];

    let mut random_generator = rand::thread_rng();

    for iter in 0..num_points
    {
        x[iter] = iter as f64;

        y[iter] = coeff[0] + coeff[1]*x[iter] + coeff[2]*x[iter]*x[iter];
        y[iter] += random_generator.gen_range(0.0..0.1);
    }

    let val = poly::fit(x, y, 5);
    
    assert!(val.coefficients.len() == 6);
    //with higher order apporximations, it is futile to compare against actual coefficients
    //this is because in such approximations the higher order coefficients can be different depending on the noise
    //however, the expected accuracy metrics value should still be within a required threshold, which guarantees prediction accuracy

    assert!(val.root_mean_squared_error < 0.05);
    assert!(val.mean_absolute_error < 0.05);
    assert!(val.mean_squared_error < 0.0025);
    assert!(val.r_squared > 0.9999);
    assert!(val.adjusted_r_squared > 0.9999);
}

#[test]
fn test_polyfit_4() 
{
    //create a list of points with unequal number of points
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    //expect error
    let result = std::panic::catch_unwind(||poly::fit(x, y, 2));
    assert!(result.is_err());
}

#[test]
fn test_polyfit_5() 
{
    //create a list of points with empty points
    let x = vec![];
    let y = vec![];

    //expect error
    let result = std::panic::catch_unwind(||poly::fit(x, y, 5));
    assert!(result.is_err());
}

#[test]
fn test_polyfit_6() 
{
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    //expect error because polynomial_degree passed is zero
    let result = std::panic::catch_unwind(||poly::fit(x, y, 0));
    assert!(result.is_err());
}