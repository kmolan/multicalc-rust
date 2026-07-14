#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::approximation::linear_approximation::*;
use multicalc::approximation::quadratic_approximation::*;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::{Numeric, ScalarFnN, c};
use multicalc::scalar_fn;
use proptest::prelude::*;
use rand::Rng;

#[test]
fn test_linear_approximation_1() {
    //function is x + y^2 + z^3, which we want to linearize
    let function_to_approximate = scalar_fn!(|v: &[f64; 3]| v[0] + v[1].powi(2) + v[2].powi(3));

    let point = [1.0, 2.0, 3.0]; //the point we want to linearize around

    let approximator = LinearApproximator::<AutoDiffMulti>::default();

    let result = approximator.get(&function_to_approximate, &point).unwrap();
    assert!(f64::abs(function_to_approximate.eval(&point) - result.predict(&point)) < 1e-9);

    //now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
    //with random noise between [-0.1, +0.1)
    let mut prediction_points = [[0.0; 3]; 1000];
    let mut random_generator = rand::thread_rng();

    for p in &mut prediction_points {
        let noise = random_generator.gen_range(-0.1..0.1);
        *p = [point[0] + noise, point[1] + noise, point[2] + noise];
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
    let function_to_approximate =
        scalar_fn!(|v: &[f64; 3]| (c(0.5) * v[0]).exp() + v[1].sin() + c(2.0) * v[2]);

    let point = [0.0, core::f64::consts::FRAC_PI_2, 10.0]; //the point we want to approximate around

    let approximator = QuadraticApproximator::<AutoDiffMulti>::default();

    let result = approximator.get(&function_to_approximate, &point).unwrap();

    assert!(f64::abs(function_to_approximate.eval(&point) - result.predict(&point)) < 1e-9);

    //now test the prediction metrics. For prediction, generate a list of 1000 points, all centered around the original point
    //with random noise between [-0.1, +0.1)
    let mut prediction_points = [[0.0; 3]; 1000];
    let mut random_generator = rand::thread_rng();

    for p in &mut prediction_points {
        let noise = random_generator.gen_range(-0.1..0.1);
        *p = [noise, core::f64::consts::FRAC_PI_2 + noise, 10.0 + noise];
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
        scalar_fn!(|v: &[f64; 3]| c(5.0) + c(2.0) * v[0] + c(3.0) * v[1] - v[2]);

    let point = [1.0, 2.0, 3.0];

    let approximator = LinearApproximator::<AutoDiffMulti>::default();
    let result = approximator.get(&function_to_approximate, &point).unwrap();

    //prediction matches the truth away from the base point, not just at it
    let elsewhere = [4.0, -1.0, 0.5];
    assert!(f64::abs(function_to_approximate.eval(&elsewhere) - result.predict(&elsewhere)) < 1e-9);

    //metrics on a spread of points where the truth genuinely varies
    let mut prediction_points = [[0.0; 3]; 10];
    for (iter, p) in prediction_points.iter_mut().enumerate() {
        let s = iter as f64;
        *p = [1.0 + s, 2.0 - s, 3.0 + 0.5 * s];
    }

    let metrics = result.get_prediction_metrics(&prediction_points, &function_to_approximate);
    assert!(metrics.mean_absolute_error < 1e-9);
    assert!(metrics.root_mean_squared_error < 1e-9);
    assert!(f64::abs(metrics.r_squared - 1.0) < 1e-9);
    assert!(f64::abs(metrics.adjusted_r_squared - 1.0) < 1e-9);
}

#[test]
fn kahan_metrics_exact_on_affine() {
    // Same affine truth as the pairwise exactness test: Kahan path must also report
    // a perfect fit so the opt-in wire-in is exercised without changing defaults.
    let truth = scalar_fn!(|v: &[f64; 3]| c(5.0) + c(2.0) * v[0] + c(3.0) * v[1] - v[2]);
    let point = [1.0, 2.0, 3.0];

    let model = LinearApproximator::<AutoDiffMulti>::default()
        .with_kahan_summation()
        .get(&truth, &point)
        .unwrap();

    let mut points = [[0.0; 3]; 10];
    for (i, p) in points.iter_mut().enumerate() {
        let s = i as f64;
        *p = [1.0 + s, 2.0 - s, 3.0 + 0.5 * s];
    }

    let metrics = model.get_prediction_metrics(&points, &truth);

    assert!(metrics.mean_absolute_error < 1e-9);
    assert!(metrics.root_mean_squared_error < 1e-9);
    assert!((metrics.r_squared - 1.0).abs() < 1e-9);
    assert!((metrics.adjusted_r_squared - 1.0).abs() < 1e-9);
}

#[test]
fn metrics_are_accurate_on_large_point_set() {
    //truth is x^2, so a linear approximation about `a` has the exact residual -(x - a)^2.
    //over a large point set this exercises the four running sums inside the metrics; assert the
    //returned metrics match the closed-form analytic values.
    const N: usize = 10_000;
    let truth = scalar_fn!(|v: &[f64; 1]| v[0] * v[0]);
    let a = 6.0;

    let approximator = LinearApproximator::<AutoDiffMulti>::default();
    let result = approximator.get(&truth, &[a]).unwrap();

    let mut prediction_points = [[0.0; 1]; N];
    for (i, p) in prediction_points.iter_mut().enumerate() {
        p[0] = 1.0 + i as f64 * 0.001; //x spread over [1.0, 11.0)
    }

    let metrics = result.get_prediction_metrics(&prediction_points, &truth);

    //closed-form reference: residual(x) = -(x - a)^2 and y = x^2
    let n = N as f64;
    let mut sum_abs = 0.0;
    let mut ss_res = 0.0;
    let mut sum_y = 0.0;
    for p in &prediction_points {
        let residual_sq = (p[0] - a) * (p[0] - a);
        sum_abs += residual_sq;
        ss_res += residual_sq * residual_sq;
        sum_y += p[0] * p[0];
    }
    let mean_y = sum_y / n;
    let mut ss_tot = 0.0;
    for p in &prediction_points {
        let d = p[0] * p[0] - mean_y;
        ss_tot += d * d;
    }
    let mae_ref = sum_abs / n;
    let mse_ref = ss_res / n;
    let rmse_ref = mse_ref.sqrt();
    let r2_ref = 1.0 - ss_res / ss_tot;

    let close = |got: f64, want: f64| (got - want).abs() <= 1e-8 * want.abs().max(1.0);
    assert!(
        close(metrics.mean_absolute_error, mae_ref),
        "mae {} vs {mae_ref}",
        metrics.mean_absolute_error
    );
    assert!(
        close(metrics.mean_squared_error, mse_ref),
        "mse {} vs {mse_ref}",
        metrics.mean_squared_error
    );
    assert!(
        close(metrics.root_mean_squared_error, rmse_ref),
        "rmse {} vs {rmse_ref}",
        metrics.root_mean_squared_error
    );
    assert!(
        close(metrics.r_squared, r2_ref),
        "r2 {} vs {r2_ref}",
        metrics.r_squared
    );
}

#[test]
fn test_linear_approximation_f32() {
    //exactly-linear truth 2x + 3y - z + 5
    let truth = scalar_fn!(|v: &[f64; 3]| c(5.0) + c(2.0) * v[0] + c(3.0) * v[1] - v[2]);

    let point = [1.0_f32, 2.0, 3.0];

    let approximator = LinearApproximator::<AutoDiffMulti<f32>>::default();
    let result = approximator.get(&truth, &point).unwrap();

    let nearby = [1.05_f32, 2.05, 2.95];
    let predicted = result.predict(&nearby);
    assert!(
        f32::abs(truth.eval(&nearby) - predicted) < 1e-4,
        "got {predicted}"
    );
}

struct Affine2 {
    b: f64,
    a: [f64; 2],
}
impl ScalarFnN<2> for Affine2 {
    fn eval<S: Numeric>(&self, p: &[S; 2]) -> S {
        S::from_f64(self.b) + S::from_f64(self.a[0]) * p[0] + S::from_f64(self.a[1]) * p[1]
    }
}

struct Quad2 {
    c: f64,
    g: [f64; 2],
    h: [[f64; 2]; 2],
}
impl ScalarFnN<2> for Quad2 {
    fn eval<S: Numeric>(&self, p: &[S; 2]) -> S {
        let (x, y) = (p[0], p[1]);
        let c = |v| S::from_f64(v);
        c(self.c)
            + c(self.g[0]) * x
            + c(self.g[1]) * y
            + S::HALF
                * (c(self.h[0][0]) * x * x
                    + c(self.h[0][1]) * x * y
                    + c(self.h[1][0]) * y * x
                    + c(self.h[1][1]) * y * y)
    }
}

fn approx_tol(scale: f64, mag: f64) -> f64 {
    1e-9 * scale.max(1.0) * mag.abs().max(1.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn proptest_linear_exact_on_affine(
        b in -5.0f64..5.0,
        a0 in -5.0f64..5.0, a1 in -5.0f64..5.0,
        px in -2.0f64..2.0, py in -2.0f64..2.0,
        samples in prop::collection::vec((-2.0f64..2.0, -2.0f64..2.0), 8),
    ) {
        let f = Affine2 { b, a: [a0, a1] };
        let point = [px, py];
        let scale = 1.0 + b.abs() + a0.abs() + a1.abs();
        let tol = approx_tol(scale, 1.0);
        let model = LinearApproximator::<AutoDiffMulti>::default()
            .get(&f, &point).unwrap();

        let mut points = [[0.0; 2]; 8];
        for (dst, &(x, y)) in points.iter_mut().zip(samples.iter()) {
            *dst = [x, y];
        }

        for p in &points {
            let err = (model.predict(p) - f.eval(p)).abs();
            prop_assert!(err < approx_tol(scale, f.eval(p)));
        }

        let metrics = model.get_prediction_metrics(&points, &f);
        prop_assert!(metrics.mean_absolute_error < tol);
        prop_assert!(metrics.root_mean_squared_error < tol);
        prop_assert!(
            metrics.r_squared.is_nan()
                || (metrics.r_squared - 1.0).abs() < 1e-9 * scale
        );
        prop_assert!((metrics.root_mean_squared_error
            - metrics.mean_squared_error.sqrt()).abs() < 1e-12 * scale);
    }

    #[test]
    fn proptest_quadratic_exact_on_quadratic(
        c in -5.0f64..5.0,
        g0 in -5.0f64..5.0, g1 in -5.0f64..5.0,
        h00 in -5.0f64..5.0, h01 in -5.0f64..5.0, h11 in -5.0f64..5.0,
        px in -2.0f64..2.0, py in -2.0f64..2.0,
        samples in prop::collection::vec((-2.0f64..2.0, -2.0f64..2.0), 8),
    ) {
        let f = Quad2 {
            c,
            g: [g0, g1],
            h: [[h00, h01], [h01, h11]],
        };
        let point = [px, py];
        let scale = 1.0
            + c.abs()
            + g0.abs()
            + g1.abs()
            + h00.abs()
            + h01.abs()
            + h11.abs();
        let tol = approx_tol(scale, 1.0);

        let model = QuadraticApproximator::<AutoDiffMulti>::default()
            .get(&f, &point)
            .unwrap();

        let mut points = [[0.0; 2]; 8];
        for (dst, &(x, y)) in points.iter_mut().zip(samples.iter()) {
            *dst = [x, y];
        }

        for p in &points {
            let err = (model.predict(p) - f.eval(p)).abs();
            prop_assert!(err < approx_tol(scale, f.eval(p)));
        }

        let metrics = model.get_prediction_metrics(&points, &f);
        prop_assert!(metrics.mean_absolute_error < tol);
        prop_assert!(metrics.root_mean_squared_error < tol);
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
