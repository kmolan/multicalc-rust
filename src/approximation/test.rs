use crate::approximation::linear_approximation::*;
use crate::approximation::quadratic_approximation::*;
use crate::numerical_derivative::finite_difference::FiniteDifferenceMulti;
use rand::Rng;

#[test]
fn test_linear_approximation_1() {
    //function is x + y^2 + z^3, which we want to linearize
    let function_to_approximate = |args: &[f64; 3]| -> f64 {
        return args[0] + args[1].powf(2.0) + args[2].powf(3.0);
    };

    let point = [1.0, 2.0, 3.0]; //the point we want to linearize around

    let approximator = LinearApproximator::<FiniteDifferenceMulti>::default();

    let result = approximator.get(&function_to_approximate, &point).unwrap();
    assert!(f64::abs(function_to_approximate(&point) - result.predict(&point)) < 1e-9);

    //now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
    //with random noise between [-0.1, +0.1)
    let mut prediction_points = [[0.0; 3]; 1000];
    let mut random_generator = rand::thread_rng();

    for iter in 0..1000 {
        let noise = random_generator.gen_range(-0.1..0.1);
        prediction_points[iter] = [point[0] + noise, point[1] + noise, point[2] + noise];
    }

    let prediction_metrics =
        result.get_prediction_metrics(&prediction_points, &function_to_approximate);

    assert!(prediction_metrics.root_mean_squared_error < 0.05);
    assert!(prediction_metrics.mean_absolute_error < 0.05);
    assert!(prediction_metrics.mean_squared_error < 0.05);
    assert!(prediction_metrics.r_squared > 0.99);
    assert!(prediction_metrics.adjusted_r_squared > 0.99);
}

#[test]
fn test_quadratic_approximation_1() {
    //function is e^(x/2) + sin(y) + 2.0*z
    let function_to_approximate = |args: &[f64; 3]| -> f64 {
        return f64::exp(args[0] / 2.0) + f64::sin(args[1]) + 2.0 * args[2];
    };

    let point = [0.0, 3.14 / 2.0, 10.0]; //the point we want to approximate around

    let approximator = QuadraticApproximator::<FiniteDifferenceMulti>::default();

    let result = approximator.get(&function_to_approximate, &point).unwrap();

    assert!(f64::abs(function_to_approximate(&point) - result.predict(&point)) < 1e-9);

    //now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
    //with random noise between [-0.1, +0.1)
    let mut prediction_points = [[0.0; 3]; 1000];
    let mut random_generator = rand::thread_rng();

    for iter in 0..1000 {
        let noise = random_generator.gen_range(-0.1..0.1);
        prediction_points[iter] = [0.0 + noise, (3.14 / 2.0) + noise, 10.0 + noise];
    }

    let prediction_metrics =
        result.get_prediction_metrics(&prediction_points, &function_to_approximate);

    assert!(prediction_metrics.root_mean_squared_error < 0.01);
    assert!(prediction_metrics.mean_absolute_error < 0.01);
    assert!(prediction_metrics.mean_squared_error < 1e-5);
    assert!(prediction_metrics.r_squared > 0.9999);
    assert!(prediction_metrics.adjusted_r_squared > 0.9999);
}

#[test]
fn test_linear_approximation_exact() {
    //an exactly-linear truth: 2x + 3y - z + 5. The linear approximation is exact
    //everywhere, so the fit is perfect (R² == 1, near-zero error).
    let function_to_approximate =
        |args: &[f64; 3]| -> f64 { 2.0 * args[0] + 3.0 * args[1] - args[2] + 5.0 };

    let point = [1.0, 2.0, 3.0];

    let approximator = LinearApproximator::<FiniteDifferenceMulti>::default();
    let result = approximator.get(&function_to_approximate, &point).unwrap();

    //prediction matches the truth away from the base point, not just at it
    let elsewhere = [4.0, -1.0, 0.5];
    assert!(
        f64::abs(function_to_approximate(&elsewhere) - result.predict(&elsewhere)) < 1e-6
    );

    //metrics on a spread of points where the truth genuinely varies
    let mut prediction_points = [[0.0; 3]; 10];
    for (iter, p) in prediction_points.iter_mut().enumerate() {
        let s = iter as f64;
        *p = [1.0 + s, 2.0 - s, 3.0 + 0.5 * s];
    }

    let metrics = result.get_prediction_metrics(&prediction_points, &function_to_approximate);
    assert!(metrics.mean_absolute_error < 1e-6);
    assert!(metrics.root_mean_squared_error < 1e-6);
    assert!(f64::abs(metrics.r_squared - 1.0) < 1e-6);
    assert!(f64::abs(metrics.adjusted_r_squared - 1.0) < 1e-6);
}
