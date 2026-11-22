//! Single- and multi-variable differentiation.
//! The derivative order for a partial is just the number of indices passed.
//!
//! Run with: `cargo run -p multicalc-demos --example differentiation`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::{AutoDiffMulti, AutoDiffSingle};
use multicalc::numerical_derivative::{DerivatorMultiVariable, DerivatorSingleVariable};
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
    let (sine, cosine) = (x.sin(), x.cos());
    let (first_order, second_order, third_order) = (1, 2, 3);

    println!("f(x) = x^2 sin(x)  at x = {x}");
    report(
        "f'",
        derivator.differentiate(first_order, &f, x).unwrap(),
        2.0 * x * sine + x * x * cosine,
    );
    report(
        "f''",
        derivator.differentiate(second_order, &f, x).unwrap(),
        2.0 * sine + 4.0 * x * cosine - x * x * sine,
    );
    report(
        "f'''",
        derivator.differentiate(third_order, &f, x).unwrap(),
        6.0 * cosine - 6.0 * x * sine - x * x * cosine,
    );

    // convenience wrappers exist for the 1st and 2nd derivative
    let _ = derivator.first_derivative(&f, x).unwrap();
    let _ = derivator.second_derivative(&f, x).unwrap();

    // ---- multi variable: g(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z at (1, 2, 3) ----
    let func = scalar_fn!(|point: &[f64; 3]| point[1] * point[0].sin()
        + point[0] * point[1].cos()
        + point[0] * point[1] * point[2].exp());
    let multi = AutoDiffMulti::default();
    let point_vals = [1.0, 2.0, 3.0];
    let (exp3, sin2, cos2) = (3.0_f64.exp(), 2.0_f64.sin(), 2.0_f64.cos());

    // Which variable to differentiate by, and in what order. x is index 0, y is 1, z is 2.
    let x_index = 0;
    let twice_by_x = [0, 0];
    let by_x_then_y = [0, 1];
    let twice_by_x_then_y = [0, 0, 1];

    println!("\ng(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z  at {point_vals:?}");

    // a single partial derivative, dg/dx = y*cos(x) + cos(y) + y*e^z
    report(
        "dg/dx",
        multi
            .first_partial_derivative(&func, x_index, &point_vals)
            .unwrap(),
        2.0 * cosine + cos2 + 2.0 * exp3,
    );

    // the derivative order is the number of indices, so no separate "order" argument is needed:
    // d2g/dx2 = -y*sin(x)
    report(
        "d2g/dx2",
        multi
            .differentiate(&func, &twice_by_x, &point_vals)
            .unwrap(),
        -2.0 * sine,
    );
    // mixed partial d(dg/dx)/dy = cos(x) - sin(y) + e^z
    report(
        "d2g/dx dy",
        multi
            .differentiate(&func, &by_x_then_y, &point_vals)
            .unwrap(),
        cosine - sin2 + exp3,
    );

    // third-order mixed partial d2(dg/dy)/dx2 = -sin(x)
    report(
        "d3g/dx2 dy",
        multi
            .differentiate(&func, &twice_by_x_then_y, &point_vals)
            .unwrap(),
        -sine,
    );
}
