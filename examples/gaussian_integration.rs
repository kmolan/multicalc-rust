//! Gaussian quadrature: Gauss-Legendre (finite), Gauss-Hermite and Gauss-Laguerre (infinite).
//!
//! Also reproduces the accuracy figures in BENCHMARKS.md section 4: these rules are exact (to
//! machine precision) for polynomial integrands, and lose accuracy fast on non-polynomial ones.
//!
//! Run with: `cargo run --example gaussian_integration`

use multicalc::numerical_integration::gaussian_integration::{GaussianMulti, GaussianSingle};
use multicalc::numerical_integration::integrator::{
    IntegratorMultiVariable, IntegratorSingleVariable,
};
use multicalc::numerical_integration::mode::GaussianQuadratureMethod;

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
    report(
        "int_0^2 4x^3-3x^2",
        legendre
            .get_single(&|x| 4.0 * x * x * x - 3.0 * x * x, &[0.0, 2.0])
            .unwrap(),
        8.0,
    );
    // non-polynomial integrand: accuracy falls
    report(
        "int_0^1 (sinx-sqrtx)e^-x",
        legendre
            .get_single(&|x| (x.sin() - x.sqrt()) * (-x).exp(), &[0.0, 1.0])
            .unwrap(),
        -0.13311916,
    );

    // ---- Gauss-Hermite: int_-inf^inf f(x) e^(-x^2) dx ----
    // pass the BARE integrand f(x); the weights already carry the e^(-x^2) factor
    let hermite = GaussianSingle::from_parameters(5, GaussianQuadratureMethod::GaussHermite);
    let hermite_m = GaussianMulti::from_parameters(5, GaussianQuadratureMethod::GaussHermite);
    let real_line = [f64::NEG_INFINITY, f64::INFINITY];
    println!("\nGauss-Hermite (bare integrand; weights carry e^(-x^2)):");
    // int x^2 e^(-x^2) = sqrt(pi)/2
    report(
        "int x^2 e^-x^2",
        hermite.get_single(&|x| x * x, &real_line).unwrap(),
        sqrt_pi / 2.0,
    );
    // multi-variable: int int x^2 y^2 e^(-x^2-y^2) = (sqrt(pi)/2)^2
    report(
        "int int x^2 y^2 e^-x^2-y^2",
        hermite_m
            .get(
                [0, 1],
                &|v: &[f64; 2]| v[0] * v[0] * v[1] * v[1],
                &[real_line; 2],
                &[0.0, 0.0],
            )
            .unwrap(),
        (sqrt_pi / 2.0) * (sqrt_pi / 2.0),
    );

    // ---- Gauss-Laguerre: int_0^inf f(x) e^(-x) dx ----
    let laguerre = GaussianSingle::from_parameters(5, GaussianQuadratureMethod::GaussLaguerre);
    let half_line = [0.0, f64::INFINITY];
    println!("\nGauss-Laguerre (bare integrand; weights carry e^(-x)):");
    // int x^2 e^(-x) = 2
    report(
        "int x^2 e^-x",
        laguerre.get_single(&|x| x * x, &half_line).unwrap(),
        2.0,
    );
    // int (4x^3 - 3x^2) e^(-x) = 18
    report(
        "int (4x^3-3x^2) e^-x",
        laguerre
            .get_single(&|x| 4.0 * x * x * x - 3.0 * x * x, &half_line)
            .unwrap(),
        18.0,
    );
}
