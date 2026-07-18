#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks differentiation, Jacobians, Hessians, vector-field operators, and Taylor
//! approximation against closed-form analytic goldens (mpmath).
//!
//! Each fixture names an `op`; the matching problem comes from the shared registry
//! so its formula is identical to the one the generator differentiated. Every
//! function-valued case also re-checks `f_at_probe` (the function at its point),
//! which fails loudly if the Rust and Python formulas ever diverge.

use multicalc::approximation::linear_approximation::LinearApproximator;
use multicalc::approximation::quadratic_approximation::QuadraticApproximator;
use multicalc::linear_algebra::Vector;
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::derivator::{DerivatorMultiVariable, DerivatorSingleVariable};
use multicalc::numerical_derivative::hessian::Hessian;
use multicalc::numerical_derivative::jacobian::Jacobian;
use multicalc::scalar::{ScalarFn, ScalarFnN, VectorFn};
use multicalc::scalar_fn;
use multicalc::vector_field::{curl, divergence, flux_integral, line_integral};
use multicalc_qa::load::*;
use multicalc_qa::problems::*;

// The unit-circle field [y, -x] and its parametrization (cos t, sin t), shared by
// the line- and flux-integral cases.
fn circle_field() -> [&'static dyn Fn(&[f64; 2]) -> f64; 2] {
    [&(|a: &[f64; 2]| a[1]), &(|a: &[f64; 2]| -a[0])]
}

fn circle_transforms() -> [&'static dyn Fn(f64) -> f64; 2] {
    [&(|t: f64| t.cos()), &(|t: f64| t.sin())]
}

#[test]
fn calculus() {
    for fx in load_dir("fixtures/v1/calculus") {
        let t = fx.tolerances.get("f64", "host");
        match fx.inputs["op"].as_str() {
            "derivative" => {
                let point = fx.inputs["point"].as_scalar();
                let order = fx.inputs["order"].as_int() as usize;
                let cube = scalar_fn!(|x| x * x * x);
                let d = AutoDiffSingle::default().get(order, &cube, point).unwrap();
                assert_scalar(d, &fx.expected["derivative"], t, "derivative");
                assert_scalar(
                    cube.eval::<f64>(point),
                    &fx.expected["f_at_probe"],
                    t,
                    "f_at_probe",
                );
            }
            "partial" => {
                let point = to_vector::<3>(&fx.inputs["point"]).into_array();
                let axes: Vec<usize> = fx.inputs["axes"]
                    .as_str()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect();
                let d = AutoDiffMulti::default();
                let val = match axes.as_slice() {
                    [a] => d.get_single_partial(&G, *a, &point),
                    [a, b] => d.get(&G, &[*a, *b], &point),
                    [a, b, c] => d.get(&G, &[*a, *b, *c], &point),
                    _ => panic!("unexpected axes {axes:?}"),
                }
                .unwrap();
                assert_scalar(val, &fx.expected["partial"], t, "partial");
                assert_scalar(
                    G.eval::<f64>(&point),
                    &fx.expected["f_at_probe"],
                    t,
                    "f_at_probe",
                );
            }
            "jacobian" => match fx.inputs["func"].as_str() {
                "jac_23" => {
                    let p = to_vector::<3>(&fx.inputs["point"]).into_array();
                    let j = Jacobian::<AutoDiffMulti>::default()
                        .get(&Jac23, &p)
                        .unwrap();
                    assert_matrix(&j, &fx.expected["jacobian"], t, "jacobian");
                    assert_vector(
                        &Vector::new(Jac23.eval::<f64>(&p)),
                        &fx.expected["f_at_probe"],
                        t,
                        "f_at_probe",
                    );
                }
                "jac_66" => {
                    let p = to_vector::<6>(&fx.inputs["point"]).into_array();
                    let j = Jacobian::<AutoDiffMulti>::default()
                        .get(&Jac66, &p)
                        .unwrap();
                    assert_matrix(&j, &fx.expected["jacobian"], t, "jacobian");
                    assert_vector(
                        &Vector::new(Jac66.eval::<f64>(&p)),
                        &fx.expected["f_at_probe"],
                        t,
                        "f_at_probe",
                    );
                }
                other => panic!("unknown jacobian func {other}"),
            },
            "hessian" => {
                let p = to_vector::<3>(&fx.inputs["point"]).into_array();
                let h = Hessian::<AutoDiffMulti>::default()
                    .get(&HessianTarget, &p)
                    .unwrap();
                assert_matrix(&h, &fx.expected["hessian"], t, "hessian");
                assert_scalar(
                    HessianTarget.eval::<f64>(&p),
                    &fx.expected["f_at_probe"],
                    t,
                    "f_at_probe",
                );
            }
            "curl_div" => {
                let p = to_vector::<3>(&fx.inputs["point"]).into_array();
                let c = curl::get_3d(AutoDiffMulti::default(), &VField3d, &p).unwrap();
                assert_vector(&Vector::new(c), &fx.expected["curl"], t, "curl");
                let dv = divergence::get_3d(AutoDiffMulti::default(), &VField3d, &p).unwrap();
                assert_scalar(dv, &fx.expected["divergence"], t, "divergence");
                assert_vector(
                    &Vector::new(VField3d.eval::<f64>(&p)),
                    &fx.expected["f_at_probe"],
                    t,
                    "f_at_probe",
                );
            }
            "line_integral" => {
                let lv = fx.inputs["limits"].as_vector();
                let val =
                    line_integral::get_2d(&circle_field(), &circle_transforms(), &[lv[0], lv[1]])
                        .unwrap();
                assert_scalar(val, &fx.expected["line_integral"], t, "line_integral");
            }
            "flux_integral" => {
                let lv = fx.inputs["limits"].as_vector();
                let val =
                    flux_integral::get_2d(&circle_field(), &circle_transforms(), &[lv[0], lv[1]])
                        .unwrap();
                assert_scalar(val, &fx.expected["flux_integral"], t, "flux_integral");
            }
            "approx" => {
                let p = to_vector::<3>(&fx.inputs["p"]).into_array();
                let q = to_vector::<3>(&fx.inputs["q"]).into_array();
                let linear = LinearApproximator::<AutoDiffMulti>::default()
                    .get(&ApproxTarget, &p)
                    .unwrap()
                    .predict(&q);
                assert_scalar(linear, &fx.expected["linear_predict"], t, "linear_predict");
                let quadratic = QuadraticApproximator::<AutoDiffMulti>::default()
                    .get(&ApproxTarget, &p)
                    .unwrap()
                    .predict(&q);
                assert_scalar(
                    quadratic,
                    &fx.expected["quadratic_predict"],
                    t,
                    "quadratic_predict",
                );
                assert_scalar(
                    ApproxTarget.eval::<f64>(&p),
                    &fx.expected["f_at_probe"],
                    t,
                    "f_at_probe",
                );
            }
            other => panic!("unknown op {other}"),
        }
    }
}
