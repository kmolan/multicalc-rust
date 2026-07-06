#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks single-variable quadrature against mpmath goldens.
//!
//! Each fixture names an integrand, a rule family and method, its parameter
//! (steps or order), and the limits. The integrand comes from the shared
//! registry; the golden is the exact integral. Finite-domain polynomial cases
//! also run an f32 pass against the f32 tolerance.

use multicalc::numerical_integration::gaussian_integration::GaussianSingle;
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::iterative_integration::IterativeSingle;
use multicalc::numerical_integration::mode::{GaussianQuadratureMethod, IterativeMethod};
use multicalc_oracle::load::*;
use multicalc_oracle::problems::{integrand_f32, integrand_f64};

fn iterative_method(s: &str) -> IterativeMethod {
    match s {
        "Booles" => IterativeMethod::Booles,
        "Simpsons" => IterativeMethod::Simpsons,
        "Trapezoidal" => IterativeMethod::Trapezoidal,
        other => panic!("unknown iterative method {other}"),
    }
}

fn gaussian_method(s: &str) -> GaussianQuadratureMethod {
    match s {
        "GaussLegendre" => GaussianQuadratureMethod::GaussLegendre,
        "GaussHermite" => GaussianQuadratureMethod::GaussHermite,
        "GaussLaguerre" => GaussianQuadratureMethod::GaussLaguerre,
        other => panic!("unknown gaussian method {other}"),
    }
}

#[test]
fn quadrature() {
    for fx in load_dir("fixtures/v1/quadrature") {
        let integrand = fx.inputs["integrand"].as_str();
        let family = fx.inputs["family"].as_str();
        let method = fx.inputs["method"].as_str();
        let param = fx.inputs["param"].as_int();
        let lv = fx.inputs["limits"].as_vector();
        let limits = [lv[0], lv[1]];

        let f = integrand_f64(integrand);
        let value = match family {
            "iterative" => {
                IterativeSingle::<f64>::from_parameters(param as u64, iterative_method(method))
                    .get_single(&f, &limits)
                    .unwrap()
            }
            "gaussian" => {
                GaussianSingle::<f64>::from_parameters(param as usize, gaussian_method(method))
                    .get_single(&f, &limits)
                    .unwrap()
            }
            other => panic!("unknown family {other}"),
        };
        let t = fx.tolerances.get("f64", "host");
        assert_scalar(value, &fx.expected["integral"], t, integrand);

        // f32 pass for the finite-domain polynomial cases (those carry an f32 tolerance).
        if fx.tolerances.table.contains_key("f32/host") {
            let f32_fn = integrand_f32(integrand);
            let limits32 = [limits[0] as f32, limits[1] as f32];
            let value32 = match family {
                "iterative" => {
                    IterativeSingle::<f32>::from_parameters(param as u64, iterative_method(method))
                        .get_single(&f32_fn, &limits32)
                        .unwrap()
                }
                "gaussian" => {
                    GaussianSingle::<f32>::from_parameters(param as usize, gaussian_method(method))
                        .get_single(&f32_fn, &limits32)
                        .unwrap()
                }
                other => panic!("unknown family {other}"),
            };
            let t32 = fx.tolerances.get("f32", "host");
            let want = fx.expected["integral"].as_scalar();
            assert!(
                close(value32 as f64, want, t32),
                "{integrand} f32: got {value32}, want {want}, tol {t32:?}"
            );
        }
    }
}
