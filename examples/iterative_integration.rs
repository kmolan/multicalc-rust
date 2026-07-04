//! Numerical integration with the iterative composite rules (Boole, Simpson, Trapezoidal),
//! including the 0.6.0 infinite / semi-infinite limits.
//!
//! Also reproduces the iterative-integration accuracy figures in benches/calculus.md: Boole is
//! the highest-order rule and most accurate, Simpson is intermediate, Trapezoidal is lowest order.
//!
//! Run with: `cargo run --example iterative_integration`

use multicalc::numerical_integration::integrator::{
    IntegratorMultiVariable, IntegratorSingleVariable,
};
use multicalc::numerical_integration::iterative_integration::{IterativeMulti, IterativeSingle};
use multicalc::numerical_integration::mode::IterativeMethod;

fn report(label: &str, value: f64, exact: f64) {
    println!(
        "  {label:<13} = {value:>12.8}   (exact {exact:>9.6}, |err| {:.0e})",
        (value - exact).abs()
    );
}

fn main() {
    // ---- single variable: int_0^2 2x dx = 4 ----
    let f = |x: f64| 2.0 * x;
    let integrator = IterativeSingle::default(); // Boole's rule, 120 intervals
    println!(
        "int_0^2 2x dx = {:.8}   (exact 4)",
        integrator.get_single(&f, &[0.0, 2.0]).unwrap()
    );

    // ---- compare the three rules on the same integrand ----
    // int_0^1 (yz x^2 e^x) dx folded three times, with y*z = 6  ->  6*(e - 2)
    let g = |v: &[f64; 3]| v[1] * v[2] * v[0] * v[0] * v[0].exp();
    let exact = 6.0 * (std::f64::consts::E - 2.0);
    let point = [1.0, 2.0, 3.0];
    println!("\nint int int (yz x^2 e^x) dx dx dx  (default 120 intervals):");
    for (name, method) in [
        ("Boole", IterativeMethod::Booles),
        ("Simpson", IterativeMethod::Simpsons),
        ("Trapezoid", IterativeMethod::Trapezoidal),
    ] {
        let solver = IterativeMulti::from_parameters(120, method);
        let val = solver.get([0, 0, 0], &g, &[[0.0, 1.0]; 3], &point).unwrap();
        report(name, val, exact);
    }

    // ---- infinite / semi-infinite limits (for convergent, decaying integrands) ----
    println!("\ninfinite limits (Boole):");
    let bell = |x: f64| (-x * x).exp();
    report(
        "e^(-x^2)",
        integrator
            .get_single(&bell, &[f64::NEG_INFINITY, f64::INFINITY])
            .unwrap(),
        std::f64::consts::PI.sqrt(),
    );
    report(
        "e^(-x)",
        integrator
            .get_single(&|x| (-x).exp(), &[0.0, f64::INFINITY])
            .unwrap(),
        1.0,
    );
    report(
        "x^(-2)",
        integrator
            .get_single(&|x| 1.0 / (x * x), &[1.0, f64::INFINITY])
            .unwrap(),
        1.0,
    );

    // ---- multi-variable partial integral ----
    // integrate (2x + yz) over x in [0, 1] with (x, y, z) = (1, 2, 3); result is 7
    let h = |v: &[f64; 3]| 2.0 * v[0] + v[1] * v[2];
    let multi = IterativeMulti::default();
    println!("\nint_0^1 (2x + yz) dx at (1, 2, 3):");
    report(
        "partial",
        multi
            .get_single_partial(&h, 0, &[0.0, 1.0], &point)
            .unwrap(),
        7.0,
    );
}
