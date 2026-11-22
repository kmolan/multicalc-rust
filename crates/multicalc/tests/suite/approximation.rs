#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::approximation::*;
use multicalc::numerical_derivative::AutoDiffMulti;
use multicalc::scalar::{Numeric, ScalarFnN, constant};
use multicalc::scalar_fn;
use proptest::prelude::*;
use rand::Rng;

/// A thousand points scattered around `centre` with independent noise in [-0.1, 0.1).
fn noisy_points_around(centre: [f64; 3]) -> [[f64; 3]; 1000] {
    let mut points = [[0.0; 3]; 1000];
    let mut random_generator = rand::thread_rng();
    for point in &mut points {
        let noise = random_generator.gen_range(-0.1..0.1);
        *point = [centre[0] + noise, centre[1] + noise, centre[2] + noise];
    }
    points
}

#[test]
fn linear_approximation_is_accurate_near_its_base_point() {
    //function is x + y^2 + z^3, which we want to linearize
    let truth = scalar_fn!(|vars: &[f64; 3]| vars[0] + vars[1].powi(2) + vars[2].powi(3));

    let point = [1.0, 2.0, 3.0]; //the point we want to linearize around

    let approximator = LinearApproximator::<AutoDiffMulti>::default();

    let model = approximator.approximate(&truth, &point).unwrap();
    assert!(f64::abs(truth.eval(&point) - model.predict(&point)) < 1e-9);

    //now test the prediction metrics, over a cloud of points around the one we linearized around
    let prediction_points = noisy_points_around(point);

    let prediction_metrics = model.prediction_metrics(&prediction_points, &truth);

    assert!(prediction_metrics.root_mean_squared_error < 0.05);
    assert!(prediction_metrics.mean_absolute_error < 0.05);
    assert!(prediction_metrics.mean_squared_error < 0.05);
    assert!(prediction_metrics.r_squared > 0.99);
    assert!(prediction_metrics.adjusted_r_squared > 0.99);
}

#[test]
fn quadratic_approximation_is_accurate_near_its_base_point() {
    //function is e^(x/2) + sin(y) + 2.0*z
    let truth = scalar_fn!(|vars: &[f64; 3]| (constant(0.5) * vars[0]).exp()
        + vars[1].sin()
        + constant(2.0) * vars[2]);

    let point = [0.0, core::f64::consts::FRAC_PI_2, 10.0]; //the point we want to approximate around

    let approximator = QuadraticApproximator::<AutoDiffMulti>::default();

    let model = approximator.approximate(&truth, &point).unwrap();

    assert!(f64::abs(truth.eval(&point) - model.predict(&point)) < 1e-9);

    //now test the prediction metrics, over a cloud of points around the one we approximated around
    let prediction_points = noisy_points_around(point);

    let prediction_metrics = model.prediction_metrics(&prediction_points, &truth);

    assert!(prediction_metrics.root_mean_squared_error < 0.01);
    assert!(prediction_metrics.mean_absolute_error < 0.01);
    assert!(prediction_metrics.mean_squared_error < 1e-5);
    assert!(prediction_metrics.r_squared > 0.9999);
    assert!(prediction_metrics.adjusted_r_squared > 0.9999);
}

#[test]
fn linear_approximation_is_exact_on_an_affine_truth() {
    //an exactly-linear truth: 2x + 3y - z + 5. The linear approximation is exact
    //everywhere, so the fit is perfect (R² == 1, near-zero error).
    let truth = scalar_fn!(|vars: &[f64; 3]| constant(5.0)
        + constant(2.0) * vars[0]
        + constant(3.0) * vars[1]
        - vars[2]);

    let point = [1.0, 2.0, 3.0];

    let approximator = LinearApproximator::<AutoDiffMulti>::default();
    let model = approximator.approximate(&truth, &point).unwrap();

    //prediction matches the truth away from the base point, not just at it
    let elsewhere = [4.0, -1.0, 0.5];
    assert!(f64::abs(truth.eval(&elsewhere) - model.predict(&elsewhere)) < 1e-9);

    //metrics on a spread of points where the truth genuinely varies
    let mut prediction_points = [[0.0; 3]; 10];
    for (index, prediction_point) in prediction_points.iter_mut().enumerate() {
        let offset = index as f64;
        *prediction_point = [1.0 + offset, 2.0 - offset, 3.0 + 0.5 * offset];
    }

    let metrics = model.prediction_metrics(&prediction_points, &truth);
    assert!(metrics.mean_absolute_error < 1e-9);
    assert!(metrics.root_mean_squared_error < 1e-9);
    assert!(f64::abs(metrics.r_squared - 1.0) < 1e-9);
    assert!(f64::abs(metrics.adjusted_r_squared - 1.0) < 1e-9);
}

#[test]
fn kahan_metrics_exact_on_affine() {
    // Same affine truth as the pairwise exactness test: Kahan path must also report
    // a perfect fit so the opt-in wire-in is exercised without changing defaults.
    let truth = scalar_fn!(|vars: &[f64; 3]| constant(5.0)
        + constant(2.0) * vars[0]
        + constant(3.0) * vars[1]
        - vars[2]);
    let point = [1.0, 2.0, 3.0];

    let model = LinearApproximator::<AutoDiffMulti>::default()
        .with_kahan_summation()
        .approximate(&truth, &point)
        .unwrap();

    let mut points = [[0.0; 3]; 10];
    for (index, sample) in points.iter_mut().enumerate() {
        let offset = index as f64;
        *sample = [1.0 + offset, 2.0 - offset, 3.0 + 0.5 * offset];
    }

    let metrics = model.prediction_metrics(&points, &truth);

    assert!(metrics.mean_absolute_error < 1e-9);
    assert!(metrics.root_mean_squared_error < 1e-9);
    assert!((metrics.r_squared - 1.0).abs() < 1e-9);
    assert!((metrics.adjusted_r_squared - 1.0).abs() < 1e-9);
}

#[test]
fn metrics_are_accurate_on_large_point_set() {
    //truth is x^2, so a linear approximation about `base_point` has the exact residual
    //-(x - base_point)^2. over a large point set this exercises the four running sums inside the
    //metrics; assert the returned metrics match the closed-form analytic values.
    const NUM_POINTS: usize = 10_000;
    let truth = scalar_fn!(|vars: &[f64; 1]| vars[0] * vars[0]);
    let base_point = 6.0;

    let approximator = LinearApproximator::<AutoDiffMulti>::default();
    let model = approximator.approximate(&truth, &[base_point]).unwrap();

    let mut prediction_points = [[0.0; 1]; NUM_POINTS];
    for (index, point) in prediction_points.iter_mut().enumerate() {
        point[0] = 1.0 + index as f64 * 0.001; //x spread over [1.0, 11.0)
    }

    let metrics = model.prediction_metrics(&prediction_points, &truth);

    //closed-form reference: residual(x) = -(x - base_point)^2 and y = x^2
    let count = NUM_POINTS as f64;
    let mut sum_of_absolute_residuals = 0.0;
    let mut residual_sum_of_squares = 0.0;
    let mut sum_of_truth = 0.0;
    for point in &prediction_points {
        let absolute_residual = (point[0] - base_point) * (point[0] - base_point);
        sum_of_absolute_residuals += absolute_residual;
        residual_sum_of_squares += absolute_residual * absolute_residual;
        sum_of_truth += point[0] * point[0];
    }
    let mean_truth = sum_of_truth / count;
    let mut total_sum_of_squares = 0.0;
    for point in &prediction_points {
        let deviation = point[0] * point[0] - mean_truth;
        total_sum_of_squares += deviation * deviation;
    }
    let expected_mean_absolute_error = sum_of_absolute_residuals / count;
    let expected_mean_squared_error = residual_sum_of_squares / count;
    let expected_root_mean_squared_error = expected_mean_squared_error.sqrt();
    let expected_r_squared = 1.0 - residual_sum_of_squares / total_sum_of_squares;

    let within_tolerance = |got: f64, want: f64| (got - want).abs() <= 1e-8 * want.abs().max(1.0);
    assert!(
        within_tolerance(metrics.mean_absolute_error, expected_mean_absolute_error),
        "mae {} vs {expected_mean_absolute_error}",
        metrics.mean_absolute_error
    );
    assert!(
        within_tolerance(metrics.mean_squared_error, expected_mean_squared_error),
        "mse {} vs {expected_mean_squared_error}",
        metrics.mean_squared_error
    );
    assert!(
        within_tolerance(
            metrics.root_mean_squared_error,
            expected_root_mean_squared_error
        ),
        "rmse {} vs {expected_root_mean_squared_error}",
        metrics.root_mean_squared_error
    );
    assert!(
        within_tolerance(metrics.r_squared, expected_r_squared),
        "r2 {} vs {expected_r_squared}",
        metrics.r_squared
    );
}

#[test]
fn linear_approximation_is_exact_on_an_affine_truth_at_f32() {
    //exactly-linear truth 2x + 3y - z + 5
    let truth = scalar_fn!(|vars: &[f64; 3]| constant(5.0)
        + constant(2.0) * vars[0]
        + constant(3.0) * vars[1]
        - vars[2]);

    let point = [1.0_f32, 2.0, 3.0];

    let approximator = LinearApproximator::<AutoDiffMulti<f32>>::default();
    let model = approximator.approximate(&truth, &point).unwrap();

    let nearby = [1.05_f32, 2.05, 2.95];
    let predicted = model.predict(&nearby);
    assert!(
        f32::abs(truth.eval(&nearby) - predicted) < 1e-4,
        "got {predicted}"
    );
}

struct Affine2 {
    intercept: f64,
    coef: [f64; 2],
}
impl ScalarFnN<2> for Affine2 {
    fn eval<S: Numeric>(&self, point: &[S; 2]) -> S {
        S::from_f64(self.intercept)
            + S::from_f64(self.coef[0]) * point[0]
            + S::from_f64(self.coef[1]) * point[1]
    }
}

struct Quad2 {
    constant: f64,
    grad: [f64; 2],
    h: [[f64; 2]; 2],
}
impl ScalarFnN<2> for Quad2 {
    fn eval<S: Numeric>(&self, point: &[S; 2]) -> S {
        let x = point[0];
        let y = point[1];
        let coefficient = |value| S::from_f64(value);
        coefficient(self.constant)
            + coefficient(self.grad[0]) * x
            + coefficient(self.grad[1]) * y
            + S::HALF
                * (coefficient(self.h[0][0]) * x * x
                    + coefficient(self.h[0][1]) * x * y
                    + coefficient(self.h[1][0]) * y * x
                    + coefficient(self.h[1][1]) * y * y)
    }
}

fn approx_tol(scale: f64, magnitude: f64) -> f64 {
    1e-9 * scale.max(1.0) * magnitude.abs().max(1.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn proptest_linear_exact_on_affine(
        intercept in -5.0f64..5.0,
        coef0 in -5.0f64..5.0, coef1 in -5.0f64..5.0,
        point_x in -2.0f64..2.0, point_y in -2.0f64..2.0,
        samples in prop::collection::vec((-2.0f64..2.0, -2.0f64..2.0), 8),
    ) {
        let affine = Affine2 { intercept, coef: [coef0, coef1] };
        let point = [point_x, point_y];
        let scale = 1.0 + intercept.abs() + coef0.abs() + coef1.abs();
        let tolerance = approx_tol(scale, 1.0);
        let model = LinearApproximator::<AutoDiffMulti>::default()
            .approximate(&affine, &point).unwrap();

        let mut points = [[0.0; 2]; 8];
        for (slot, &(x, y)) in points.iter_mut().zip(samples.iter()) {
            *slot = [x, y];
        }

        for sample in &points {
            let error = (model.predict(sample) - affine.eval(sample)).abs();
            prop_assert!(error < approx_tol(scale, affine.eval(sample)));
        }

        let metrics = model.prediction_metrics(&points, &affine);
        prop_assert!(metrics.mean_absolute_error < tolerance);
        prop_assert!(metrics.root_mean_squared_error < tolerance);
        prop_assert!(
            metrics.r_squared.is_nan()
                || (metrics.r_squared - 1.0).abs() < 1e-9 * scale
        );
        prop_assert!((metrics.root_mean_squared_error
            - metrics.mean_squared_error.sqrt()).abs() < 1e-12 * scale);
    }

    #[test]
    fn proptest_quadratic_exact_on_quadratic(
        constant in -5.0f64..5.0,
        grad0 in -5.0f64..5.0, grad1 in -5.0f64..5.0,
        h00 in -5.0f64..5.0, h01 in -5.0f64..5.0, h11 in -5.0f64..5.0,
        point_x in -2.0f64..2.0, point_y in -2.0f64..2.0,
        samples in prop::collection::vec((-2.0f64..2.0, -2.0f64..2.0), 8),
    ) {
        let quadratic = Quad2 {
            constant,
            grad: [grad0, grad1],
            h: [[h00, h01], [h01, h11]],
        };
        let point = [point_x, point_y];
        let scale = 1.0
            + constant.abs()
            + grad0.abs()
            + grad1.abs()
            + h00.abs()
            + h01.abs()
            + h11.abs();
        let tolerance = approx_tol(scale, 1.0);

        let model = QuadraticApproximator::<AutoDiffMulti>::default()
            .approximate(&quadratic, &point)
            .unwrap();

        let mut points = [[0.0; 2]; 8];
        for (slot, &(x, y)) in points.iter_mut().zip(samples.iter()) {
            *slot = [x, y];
        }

        for sample in &points {
            let error = (model.predict(sample) - quadratic.eval(sample)).abs();
            prop_assert!(error < approx_tol(scale, quadratic.eval(sample)));
        }

        let metrics = model.prediction_metrics(&points, &quadratic);
        prop_assert!(metrics.mean_absolute_error < tolerance);
        prop_assert!(metrics.root_mean_squared_error < tolerance);
        prop_assert!(
            metrics.r_squared.is_nan()
                || (metrics.r_squared - 1.0).abs() < 1e-9 * scale
        );
        prop_assert!(
            (metrics.root_mean_squared_error - metrics.mean_squared_error.sqrt()).abs()
                < 1e-12 * scale
        );
    }
}
