//! Single- and multi-variable differentiation.
//! The derivative order for a partial is just the number of indices passed.
//!
//! Run with: `cargo run -p multicalc-demos --example differentiation`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::derivator::{DerivatorMultiVariable, DerivatorSingleVariable};
use multicalc::scalar_fn;

fn report(label: &str, value: f64, exact: f64) {
    assert!((value - exact).abs() < 1e-6, "{label}: |err| too large");
    println!(
        "  {label:<18} = {value:>13.8}   (exact {exact:>13.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

fn main() {
    // ---- single variable: f(x) = x^2 sin(x) at x = 1, by autodiff (exact) ----
    let f = scalar_fn!(|x| x * x * x.sin());
    let derivator = AutoDiffSingle::default();
    let x = 1.0_f64;
    let (s, c) = (x.sin(), x.cos());

    println!("f(x) = x^2 sin(x)  at x = {x}");
    report(
        "f'",
        derivator.get(1, &f, x).unwrap(),
        2.0 * x * s + x * x * c,
    );
    report(
        "f''",
        derivator.get(2, &f, x).unwrap(),
        2.0 * s + 4.0 * x * c - x * x * s,
    );
    report(
        "f'''",
        derivator.get(3, &f, x).unwrap(),
        6.0 * c - 6.0 * x * s - x * x * c,
    );

    // convenience wrappers exist for the 1st and 2nd derivative
    let _ = derivator.get_single(&f, x).unwrap();
    let _ = derivator.get_double(&f, x).unwrap();

    // ---- multi variable: g(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z at (1, 2, 3) ----
    let g =
        scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin() + v[0] * v[1].cos() + v[0] * v[1] * v[2].exp());
    let multi = AutoDiffMulti::default();
    let p = [1.0, 2.0, 3.0];
    let (e3, sin2, cos2) = (3.0_f64.exp(), 2.0_f64.sin(), 2.0_f64.cos());

    println!("\ng(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z  at {p:?}");

    // a single partial derivative, dg/dx = y*cos(x) + cos(y) + y*e^z
    report(
        "dg/dx",
        multi.get_single_partial(&g, 0, &p).unwrap(),
        2.0 * c + cos2 + 2.0 * e3,
    );

    // the derivative order is the number of indices, so no separate "order" argument is needed:
    // d2g/dx2 = -y*sin(x)
    report("d2g/dx2", multi.get(&g, &[0, 0], &p).unwrap(), -2.0 * s);
    // mixed partial d(dg/dx)/dy = cos(x) - sin(y) + e^z
    report(
        "d2g/dx dy",
        multi.get(&g, &[0, 1], &p).unwrap(),
        c - sin2 + e3,
    );

    // third-order mixed partial d2(dg/dy)/dx2 = -sin(x)
    report("d3g/dx2 dy", multi.get(&g, &[0, 0, 1], &p).unwrap(), -s);
}
