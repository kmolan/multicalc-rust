#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the Levenberg-Marquardt solver against MINPACK goldens.
//!
//! Each fixture names a problem; the matching residual comes from the shared
//! registry so it is identical to the one scipy solved. The comparison is the
//! recovered solution and the residual norm at that solution (a convention-free
//! quantity), not any library-specific cost scalar.

use multicalc::linear_algebra::Vector;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::optimization::LevenbergMarquardt;
use multicalc::scalar::VectorFn;
use multicalc_oracle::load::*;
use multicalc_oracle::problems::{CircleFit, GaussianPeaks, Rosenbrock, Trigonometric6};
use multicalc_oracle::schema::*;

fn run_lm<F: VectorFn<N, M>, const N: usize, const M: usize>(problem: &F, fx: &Fixture) {
    let x0 = to_vector::<N>(&fx.inputs["x0"]).into_array();
    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(problem, &x0)
        .unwrap();
    let t = fx.tolerances.get("f64", "host");

    assert_vector(
        &Vector::new(report.solution),
        &fx.expected["solution"],
        t,
        "solution",
    );

    let residual = problem.eval::<f64>(&report.solution);
    let norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert_scalar(norm, &fx.expected["residual_norm"], t, "residual_norm");
}

#[test]
fn optimization() {
    for fx in load_dir("fixtures/v1/optimization") {
        match fx.inputs["problem"].as_str() {
            "rosenbrock" => run_lm(&Rosenbrock, &fx),
            "trigonometric6" => run_lm(&Trigonometric6, &fx),
            "circle_fit" => run_lm(&CircleFit, &fx),
            "gaussian_peaks" => run_lm(&GaussianPeaks, &fx),
            other => panic!("unknown problem key {other}"),
        }
    }
}
