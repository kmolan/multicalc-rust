#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the Levenberg-Marquardt solver against MINPACK goldens.
//!
//! Each fixture names a problem; the matching residual comes from the shared
//! registry so it is identical to the one scipy solved. The comparison is the
//! recovered solution and the residual norm at that solution (a convention-free
//! quantity), not any library-specific cost scalar.

use multicalc::linear_algebra::Vector;
use multicalc::numerical_derivative::AutoDiffMulti;
use multicalc::optimization::LevenbergMarquardt;
use multicalc::scalar::VectorFn;
use multicalc_qa::load::*;
use multicalc_qa::problems::{CircleFit, GaussianPeaks, Rosenbrock, Trigonometric6};
use multicalc_qa::schema::*;

fn run_lm<F: VectorFn<N, M>, const N: usize, const M: usize>(problem: &F, fixture: &Fixture) {
    let initial_guess = to_vector::<N>(&fixture.inputs["x0"]).into_array();
    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(problem, &initial_guess)
        .unwrap();
    let tolerance = fixture.tolerances.f64;

    assert_vector(
        &Vector::new(report.solution),
        &fixture.expected["solution"],
        tolerance,
        "solution",
    );

    let residual = problem.eval::<f64>(&report.solution);
    let norm = residual
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    assert_scalar(
        norm,
        &fixture.expected["residual_norm"],
        tolerance,
        "residual_norm",
    );
}

#[test]
fn optimization() {
    for fixture in load_dir("optimization") {
        match fixture.inputs["problem"].as_str() {
            "rosenbrock" => run_lm(&Rosenbrock, &fixture),
            "trigonometric6" => run_lm(&Trigonometric6, &fixture),
            "circle_fit" => run_lm(&CircleFit, &fixture),
            "gaussian_peaks" => run_lm(&GaussianPeaks, &fixture),
            other => panic!("unknown problem key {other}"),
        }
    }
}
