#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks differentiation, Jacobians, Hessians, vector-field operators, and Taylor
//! approximation against closed-form analytic goldens (mpmath).
//!
//! Each fixture names an `op`; the matching problem comes from the shared registry
//! so its formula is identical to the one the generator differentiated. Every
//! function-valued case also re-checks `f_at_probe` (the function at its point),
//! which fails loudly if the Rust and Python formulas ever diverge.

use multicalc::approximation::LinearApproximator;
use multicalc::approximation::QuadraticApproximator;
use multicalc::linear_algebra::Vector;
use multicalc::numerical_derivative::Hessian;
use multicalc::numerical_derivative::Jacobian;
use multicalc::numerical_derivative::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::{DerivatorMultiVariable, DerivatorSingleVariable};
use multicalc::scalar::{ScalarFn, ScalarFnN, VectorFn};
use multicalc::scalar_fn;
use multicalc::vector_field::{curl_3d, divergence_3d, flux_integral_2d, line_integral_2d};
use multicalc_qa::load::*;
use multicalc_qa::problems::*;

// The unit-circle field [y, -x] and its parametrization (cos t, sin t), shared by
// the line- and flux-integral cases.
fn circle_field() -> [&'static dyn Fn(&[f64; 2]) -> f64; 2] {
    [&(|a: &[f64; 2]| a[1]), &(|a: &[f64; 2]| -a[0])]
}

fn circle_transforms() -> [&'static dyn Fn(f64) -> f64; 2] {
    [&(|time: f64| time.cos()), &(|time: f64| time.sin())]
}

#[test]
fn calculus() {
    for fixture in load_dir("calculus") {
        let tolerance = fixture.tolerances.f64;
        match fixture.inputs["op"].as_str() {
            "derivative" => {
                let point = fixture.inputs["point"].as_scalar();
                let order = fixture.inputs["order"].as_int() as usize;
                let cube = scalar_fn!(|x| x * x * x);
                let derivative = AutoDiffSingle::default()
                    .differentiate(order, &cube, point)
                    .unwrap();
                assert_scalar(
                    derivative,
                    &fixture.expected["derivative"],
                    tolerance,
                    "derivative",
                );
                assert_scalar(
                    cube.eval::<f64>(point),
                    &fixture.expected["f_at_probe"],
                    tolerance,
                    "f_at_probe",
                );
            }
            "partial" => {
                let point = to_vector::<3>(&fixture.inputs["point"]).into_array();
                let axes: Vec<usize> = fixture.inputs["axes"]
                    .as_str()
                    .split(',')
                    .map(|part| part.parse().unwrap())
                    .collect();
                let derivator = AutoDiffMulti::default();
                let val = match axes.as_slice() {
                    [a] => derivator.first_partial_derivative(&Transcendental, *a, &point),
                    [a, b] => derivator.differentiate(&Transcendental, &[*a, *b], &point),
                    [a, b, third] => {
                        derivator.differentiate(&Transcendental, &[*a, *b, *third], &point)
                    }
                    _ => panic!("unexpected axes {axes:?}"),
                }
                .unwrap();
                assert_scalar(val, &fixture.expected["partial"], tolerance, "partial");
                assert_scalar(
                    Transcendental.eval::<f64>(&point),
                    &fixture.expected["f_at_probe"],
                    tolerance,
                    "f_at_probe",
                );
            }
            "jacobian" => match fixture.inputs["func"].as_str() {
                "jac_23" => {
                    let point = to_vector::<3>(&fixture.inputs["point"]).into_array();
                    let j = Jacobian::<AutoDiffMulti>::default()
                        .evaluate(&Jac23, &point)
                        .unwrap();
                    assert_matrix(&j, &fixture.expected["jacobian"], tolerance, "jacobian");
                    assert_vector(
                        &Vector::new(Jac23.eval::<f64>(&point)),
                        &fixture.expected["f_at_probe"],
                        tolerance,
                        "f_at_probe",
                    );
                }
                "jac_66" => {
                    let point = to_vector::<6>(&fixture.inputs["point"]).into_array();
                    let j = Jacobian::<AutoDiffMulti>::default()
                        .evaluate(&Jac66, &point)
                        .unwrap();
                    assert_matrix(&j, &fixture.expected["jacobian"], tolerance, "jacobian");
                    assert_vector(
                        &Vector::new(Jac66.eval::<f64>(&point)),
                        &fixture.expected["f_at_probe"],
                        tolerance,
                        "f_at_probe",
                    );
                }
                other => panic!("unknown jacobian func {other}"),
            },
            "hessian" => {
                let point = to_vector::<3>(&fixture.inputs["point"]).into_array();
                let h = Hessian::<AutoDiffMulti>::default()
                    .evaluate(&HessianTarget, &point)
                    .unwrap();
                assert_matrix(&h, &fixture.expected["hessian"], tolerance, "hessian");
                assert_scalar(
                    HessianTarget.eval::<f64>(&point),
                    &fixture.expected["f_at_probe"],
                    tolerance,
                    "f_at_probe",
                );
            }
            "curl_div" => {
                let point = to_vector::<3>(&fixture.inputs["point"]).into_array();
                let curl = curl_3d(AutoDiffMulti::default(), &VField3d, &point).unwrap();
                assert_vector(
                    &Vector::new(curl),
                    &fixture.expected["curl"],
                    tolerance,
                    "curl",
                );
                let divergence =
                    divergence_3d(AutoDiffMulti::default(), &VField3d, &point).unwrap();
                assert_scalar(
                    divergence,
                    &fixture.expected["divergence"],
                    tolerance,
                    "divergence",
                );
                assert_vector(
                    &Vector::new(VField3d.eval::<f64>(&point)),
                    &fixture.expected["f_at_probe"],
                    tolerance,
                    "f_at_probe",
                );
            }
            "line_integral" => {
                let limits = fixture.inputs["limits"].as_vector();
                let val = line_integral_2d(
                    &circle_field(),
                    &circle_transforms(),
                    &[limits[0], limits[1]],
                )
                .unwrap();
                assert_scalar(
                    val,
                    &fixture.expected["line_integral"],
                    tolerance,
                    "line_integral",
                );
            }
            "flux_integral" => {
                let limits = fixture.inputs["limits"].as_vector();
                let val = flux_integral_2d(
                    &circle_field(),
                    &circle_transforms(),
                    &[limits[0], limits[1]],
                )
                .unwrap();
                assert_scalar(
                    val,
                    &fixture.expected["flux_integral"],
                    tolerance,
                    "flux_integral",
                );
            }
            "approx" => {
                let point = to_vector::<3>(&fixture.inputs["p"]).into_array();
                let query = to_vector::<3>(&fixture.inputs["q"]).into_array();
                let linear = LinearApproximator::<AutoDiffMulti>::default()
                    .approximate(&ApproxTarget, &point)
                    .unwrap()
                    .predict(&query);
                assert_scalar(
                    linear,
                    &fixture.expected["linear_predict"],
                    tolerance,
                    "linear_predict",
                );
                let quadratic = QuadraticApproximator::<AutoDiffMulti>::default()
                    .approximate(&ApproxTarget, &point)
                    .unwrap()
                    .predict(&query);
                assert_scalar(
                    quadratic,
                    &fixture.expected["quadratic_predict"],
                    tolerance,
                    "quadratic_predict",
                );
                assert_scalar(
                    ApproxTarget.eval::<f64>(&point),
                    &fixture.expected["f_at_probe"],
                    tolerance,
                    "f_at_probe",
                );
            }
            other => panic!("unknown op {other}"),
        }
    }
}
