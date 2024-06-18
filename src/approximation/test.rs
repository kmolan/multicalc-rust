use crate::approximation::linear_approximation;
use crate::approximation::quadratic_approximation;
use rand::Rng;

#[test]
fn test_linear_approximation_1() 
{
    //function is x + y^2 + z^3, which we want to linearize
    let function_to_approximate = | args: &[f64; 3] | -> f64 
    { 
        return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
    };

    let point = [1.0, 2.0, 3.0]; //the point we want to linearize around

    let result = linear_approximation::get(&function_to_approximate, &point);
    assert!(f64::abs(function_to_approximate(&point) - result.get_prediction_value(&point)) < 1e-9);

    //now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
    //with random noise between [-0.1, +0.1) 
    let mut prediction_points = [[0.0; 3]; 1000];
    let mut random_generator = rand::thread_rng();

    for iter in 0..1000
    {
        let noise = random_generator.gen_range(-0.1..0.1);
        prediction_points[iter] = [point[0] + noise, point[1] + noise, point[2] + noise];
    }
    
    let prediction_metrics = result.get_prediction_metrics(&prediction_points, &function_to_approximate);

    assert!(prediction_metrics.root_mean_squared_error < 0.05);
    assert!(prediction_metrics.mean_absolute_error < 0.05);
    assert!(prediction_metrics.mean_squared_error < 0.05);
    assert!(prediction_metrics.r_squared > 0.99);
    assert!(prediction_metrics.adjusted_r_squared > 0.99);
    
}

#[test]
fn test_linear_approximation_2() 
{
    //function is x + y^2 + z^3, which we want to linearize
    let function_to_approximate = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
    };

    //the point we want to linearize around
    let point = [num_complex::c64(1.0, 1.0), num_complex::c64(2.0, 0.0), num_complex::c64(3.0, 0.5)];

    let result = linear_approximation::get(&function_to_approximate, &point);
    assert!(num_complex::ComplexFloat::abs(function_to_approximate(&point).re - result.get_prediction_value(&point).re) < 1e-9);
    assert!(num_complex::ComplexFloat::abs(function_to_approximate(&point).im - result.get_prediction_value(&point).im) < 1e-9);

    //now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
    //with random noise between [-0.1, +0.1) 
    let mut prediction_points = [[num_complex::c64(0.0, 1.0); 3]; 1000];
    let mut random_generator = rand::thread_rng();

    for iter in 0..1000
    {
        let noise = random_generator.gen_range(-0.1..0.1);
        prediction_points[iter] = [point[0] + noise, point[1] + noise, point[2] + noise];
    }

    let prediction_metrics = result.get_prediction_metrics(&prediction_points, &function_to_approximate);

    assert!(prediction_metrics.root_mean_squared_error < 0.05);
    assert!(prediction_metrics.mean_absolute_error < 0.05);
    assert!(prediction_metrics.mean_squared_error < 0.05);
    assert!(prediction_metrics.r_squared > 0.99);
    assert!(prediction_metrics.adjusted_r_squared > 0.99);
    
}


#[test]
fn test_quadratic_approximation_1() 
{
    //function is e^(x/2) + sin(y) + 2.0*z
    let function_to_approximate = | args: &[f64; 3] | -> f64 
    { 
        return f64::exp(args[0]/2.0) + f64::sin(args[1]) + 2.0*args[2];
    };

    let point = [0.0, 3.14/2.0, 10.0]; //the point we want to approximate around

    let result = quadratic_approximation::get(&function_to_approximate, &point);

    assert!(f64::abs(function_to_approximate(&point) - result.get_prediction_value(&point)) < 1e-9);

    //now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
    //with random noise between [-0.1, +0.1) 
    let mut prediction_points = [[0.0; 3]; 1000];
    let mut random_generator = rand::thread_rng();

    for iter in 0..1000
    {
        let noise = random_generator.gen_range(-0.1..0.1);
        prediction_points[iter] = [0.0 + noise, (3.14/2.0) + noise, 10.0 + noise];
    }

    let prediction_metrics = result.get_prediction_metrics(&prediction_points, &function_to_approximate);

    assert!(prediction_metrics.root_mean_squared_error < 0.01);
    assert!(prediction_metrics.mean_absolute_error < 0.01);
    assert!(prediction_metrics.mean_squared_error < 1e-5);
    assert!(prediction_metrics.r_squared > 0.9999);
    assert!(prediction_metrics.adjusted_r_squared > 0.9999);
}


#[test]
fn test_quadratic_approximation_2() 
{
    //function is e^(x/2) + sin(y) + 2.0*z
    let function_to_approximate = | args: &[num_complex::Complex64; 3] | -> num_complex::Complex64 
    { 
        return num_complex::ComplexFloat::exp(args[0]/2.0) + args[1].sin() + 2.0*args[2];
    };

    //the point we want to approximate around
    let point = [num_complex::c64(0.0, 1.0), num_complex::c64(3.12, 0.0), num_complex::c64(10.0, 0.5)];

    let result = quadratic_approximation::get(&function_to_approximate, &point);
    assert!(num_complex::ComplexFloat::abs(function_to_approximate(&point).re - result.get_prediction_value(&point).re) < 0.5);
    assert!(num_complex::ComplexFloat::abs(function_to_approximate(&point).im - result.get_prediction_value(&point).im) < 0.5);

    //now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
    //with random noise between [-0.1, +0.1) 
    let mut prediction_points = [[num_complex::c64(0.0, 0.0); 3]; 1000];
    let mut random_generator = rand::thread_rng();

    for iter in 0..1000
    {
        let noise = random_generator.gen_range(-0.1..0.1);
        prediction_points[iter] = [point[0] + noise, point[1] + noise, point[2] + noise];
    }

    let prediction_metrics = result.get_prediction_metrics(&prediction_points, &function_to_approximate);

    assert!(prediction_metrics.root_mean_squared_error < 0.25);
    assert!(prediction_metrics.mean_absolute_error < 0.25);
    assert!(prediction_metrics.mean_squared_error < 0.1);
    assert!(prediction_metrics.r_squared > 0.9);
    assert!(prediction_metrics.adjusted_r_squared > 0.9);
}