#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks single-variable quadrature against mpmath goldens.
//!
//! Each fixture names an integrand, a rule family and method, its parameter
//! (steps or order), and the limits. The integrand comes from the shared
//! registry; the golden is the exact integral. Finite-domain polynomial cases
//! also run an f32 pass against the f32 tolerance.

use multicalc::numerical_integration::GaussianSingle;
use multicalc::numerical_integration::IntegratorSingleVariable;
use multicalc::numerical_integration::IterativeSingle;
use multicalc::numerical_integration::{GaussianQuadratureMethod, IterativeMethod};
use multicalc_qa::load::*;
use multicalc_qa::problems::integrand;

#[must_use]
fn iterative_method(name: &str) -> IterativeMethod {
    match name {
        "Booles" => IterativeMethod::Booles,
        "Simpsons" => IterativeMethod::Simpsons,
        "Trapezoidal" => IterativeMethod::Trapezoidal,
        other => panic!("unknown iterative method {other}"),
    }
}

#[must_use]
fn gaussian_method(name: &str) -> GaussianQuadratureMethod {
    match name {
        "GaussLegendre" => GaussianQuadratureMethod::GaussLegendre,
        "GaussHermite" => GaussianQuadratureMethod::GaussHermite,
        "GaussLaguerre" => GaussianQuadratureMethod::GaussLaguerre,
        other => panic!("unknown gaussian method {other}"),
    }
}

#[test]
fn quadrature() {
    for fixture in load_dir("quadrature") {
        let integrand_key = fixture.inputs["integrand"].as_str();
        let family = fixture.inputs["family"].as_str();
        let method = fixture.inputs["method"].as_str();
        let param = fixture.inputs["param"].as_int();
        let limits_vec = fixture.inputs["limits"].as_vector();
        let limits = [limits_vec[0], limits_vec[1]];

        let f = integrand::<f64>(integrand_key);
        let value = match family {
            "iterative" => {
                IterativeSingle::<f64>::from_parameters(param as u64, iterative_method(method))
                    .single_integral(&f, &limits)
                    .unwrap()
            }
            "gaussian" => {
                GaussianSingle::<f64>::from_parameters(param as usize, gaussian_method(method))
                    .single_integral(&f, &limits)
                    .unwrap()
            }
            other => panic!("unknown family {other}"),
        };
        let tolerance = fixture.tolerances.f64;
        assert_scalar(
            value,
            &fixture.expected["integral"],
            tolerance,
            integrand_key,
        );

        // f32 pass for the finite-domain polynomial cases (those carry an f32 tolerance).
        if let Some(tolerance32) = fixture.tolerances.f32 {
            let f32_fn = integrand::<f32>(integrand_key);
            let limits32 = [limits[0] as f32, limits[1] as f32];
            let value32 = match family {
                "iterative" => {
                    IterativeSingle::<f32>::from_parameters(param as u64, iterative_method(method))
                        .single_integral(&f32_fn, &limits32)
                        .unwrap()
                }
                "gaussian" => {
                    GaussianSingle::<f32>::from_parameters(param as usize, gaussian_method(method))
                        .single_integral(&f32_fn, &limits32)
                        .unwrap()
                }
                other => panic!("unknown family {other}"),
            };
            let want = fixture.expected["integral"].as_scalar();
            assert!(
                close(value32 as f64, want, tolerance32),
                "{integrand_key} f32: got {value32}, want {want}, tol {tolerance32:?}"
            );
        }
    }
}
