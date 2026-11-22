//! Gaussian quadrature: Gauss-Legendre (finite), Gauss-Hermite and Gauss-Laguerre (infinite).
//!
//! Also reproduces the Gaussian-quadrature accuracy figures in benches/calculus.md: these rules
//! are exact (to machine precision) for polynomial integrands, and lose accuracy fast on
//! non-polynomial ones.
//!
//! Run with: `cargo run -p multicalc-demos --example gaussian_integration`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_integration::GaussianQuadratureMethod;
use multicalc::numerical_integration::{GaussianMulti, GaussianSingle};
use multicalc::numerical_integration::{IntegratorMultiVariable, IntegratorSingleVariable};

fn report(label: &str, value: f64, exact: f64) {
    println!(
        "  {label:<26} = {value:>12.8}   (exact {exact:>9.6}, |err| {:.0e})",
        (value - exact).abs()
    );
}

fn main() {
    let sqrt_pi = std::f64::consts::PI.sqrt();

    // ---- Gauss-Legendre over a finite interval ----
    let legendre = GaussianSingle::from_parameters(5, GaussianQuadratureMethod::GaussLegendre);
    println!("Gauss-Legendre (finite limits):");
    // int_0^2 (4x^3 - 3x^2) dx = 8  (exact: order 5 handles degree <= 9)
    let poly: f64 = legendre
        .single_integral(&|x| 4.0 * x * x * x - 3.0 * x * x, &[0.0, 2.0])
        .unwrap();
    assert!(
        (poly - 8.0).abs() < 1e-9,
        "Legendre is exact for polynomials"
    );
    report("int_0^2 4x^3-3x^2", poly, 8.0);
    // non-polynomial integrand: accuracy falls
    report(
        "int_0^1 (sinx-sqrtx)e^-x",
        legendre
            .single_integral(&|x| (x.sin() - x.sqrt()) * (-x).exp(), &[0.0, 1.0])
            .unwrap(),
        -0.13311916,
    );

    // ---- Gauss-Hermite: int_-inf^inf f(x) e^(-x^2) dx ----
    // pass the BARE integrand f(x); the weights already carry the e^(-x^2) factor
    let node_count = 5;
    let hermite =
        GaussianSingle::from_parameters(node_count, GaussianQuadratureMethod::GaussHermite);
    let hermite_m =
        GaussianMulti::from_parameters(node_count, GaussianQuadratureMethod::GaussHermite);
    let real_line = [f64::NEG_INFINITY, f64::INFINITY];
    println!("\nGauss-Hermite (bare integrand; weights carry e^(-x^2)):");
    // int x^2 e^(-x^2) = sqrt(pi)/2
    report(
        "int x^2 e^-x^2",
        hermite.single_integral(&|x| x * x, &real_line).unwrap(),
        sqrt_pi / 2.0,
    );
    // multi-variable: int int x^2 y^2 e^(-x^2-y^2) = (sqrt(pi)/2)^2
    report(
        "int int x^2 y^2 e^-x^2-y^2",
        {
            let by_x_then_y = [0, 1];
            let plane = [real_line; 2];
            let origin = [0.0, 0.0];
            hermite_m
                .integrate(
                    by_x_then_y,
                    &|point: &[f64; 2]| point[0] * point[0] * point[1] * point[1],
                    &plane,
                    &origin,
                )
                .unwrap()
        },
        (sqrt_pi / 2.0) * (sqrt_pi / 2.0),
    );

    // ---- Gauss-Laguerre: int_0^inf f(x) e^(-x) dx ----
    let laguerre = GaussianSingle::from_parameters(5, GaussianQuadratureMethod::GaussLaguerre);
    let half_line = [0.0, f64::INFINITY];
    println!("\nGauss-Laguerre (bare integrand; weights carry e^(-x)):");
    // int x^2 e^(-x) = 2
    report(
        "int x^2 e^-x",
        laguerre.single_integral(&|x| x * x, &half_line).unwrap(),
        2.0,
    );
    // int (4x^3 - 3x^2) e^(-x) = 18
    report(
        "int (4x^3-3x^2) e^-x",
        laguerre
            .single_integral(&|x| 4.0 * x * x * x - 3.0 * x * x, &half_line)
            .unwrap(),
        18.0,
    );
}
