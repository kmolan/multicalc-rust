#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks RK45 dense-output samples against scipy solve_ivp goldens.

use multicalc::linear_algebra::Vector;
use multicalc::ode::Rk45;
use multicalc_qa::load::*;
use multicalc_qa::problems::{ode_exp_decay, ode_harmonic, ode_two_body, ode_van_der_pol_mild};
use multicalc_qa::schema::Fixture;

fn run_case<const N: usize>(fx: &Fixture, f: &dyn Fn(f64, &Vector<N>) -> Vector<N>) {
    let problem = fx.inputs["problem"].as_str();
    let y0 = fx.inputs["y0"].as_vector();
    let t0 = fx.inputs["t0"].as_scalar();
    let times = fx.inputs["times"].as_vector();
    let (rows, cols, states) = fx.expected["states"].as_matrix();
    assert_eq!(cols, N, "{problem}: column count");
    assert_eq!(rows, times.len(), "{problem}: row count");
    let tol = fx.tolerances.get("f64", "host");

    let y0v = Vector::<N>::from_fn(|i| y0[i]);
    let mut out = vec![Vector::<N>::zeros(); times.len()];
    Rk45::<f64>::default()
        .with_rtol(1e-10)
        .with_atol(1e-12)
        .solve_on_grid(&f, t0, &y0v, &times, &mut out)
        .unwrap();
    for (i, y) in out.iter().enumerate() {
        let y = *y.as_array();
        for j in 0..N {
            let want = states[i * N + j];
            assert!(
                close(y[j], want, tol),
                "{problem}[t{i}][{j}]: got {}, want {want}, tol {tol:?}",
                y[j]
            );
        }
    }
}

#[test]
fn ode() {
    for fx in load_dir("fixtures/v1/ode") {
        match fx.inputs["problem"].as_str() {
            "exp_decay" => run_case(&fx, &ode_exp_decay),
            "harmonic" => run_case(&fx, &ode_harmonic),
            "two_body" => run_case(&fx, &ode_two_body),
            "van_der_pol_mild" => run_case(&fx, &ode_van_der_pol_mild),
            other => panic!("unknown ode problem {other}"),
        }
    }
}
