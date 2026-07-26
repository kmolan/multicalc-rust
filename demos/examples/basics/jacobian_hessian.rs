//! Jacobian and Hessian matrices of multi-variable functions.
//!
//! Run with: `cargo run -p multicalc-demos --example jacobian_hessian`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::Hessian;
use multicalc::numerical_derivative::Jacobian;
use multicalc::scalar::c;
use multicalc::{scalar_fn, scalar_fn_vec};

fn main() {
    // ---- Jacobian of the vector function (x*y*z, x^2 + y^2) ----
    let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    let point = [1.0, 2.0, 3.0];

    let jacobian: Jacobian = Jacobian::default();
    let result = jacobian.evaluate(&f, &point).unwrap();

    println!("Jacobian of (x*y*z, x^2 + y^2) at {point:?}:");
    for row in 0..2 {
        println!(
            "  [{:.4}, {:.4}, {:.4}]",
            result[(row, 0)],
            result[(row, 1)],
            result[(row, 2)]
        );
    }
    println!("  (exact [[6, 3, 2], [2, 4, 0]])");
    let exact = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for (i, row) in exact.iter().enumerate() {
        for (j, &want) in row.iter().enumerate() {
            assert!((result[(i, j)] - want).abs() < 1e-9);
        }
    }

    // ---- Hessian of f(x, y) = y*sin(x) + 2*x*e^y ----
    let g = scalar_fn!(|v: &[f64; 2]| v[1] * v[0].sin() + c(2.0) * v[0] * v[1].exp());
    let hessian_point = [1.0, 2.0];
    let hessian: Hessian = Hessian::default();
    let result = hessian.evaluate(&g, &hessian_point).unwrap();

    println!("\nHessian of y*sin(x) + 2*x*e^y at {hessian_point:?}:");
    for row in 0..2 {
        println!("  [{:.4}, {:.4}]", result[(row, 0)], result[(row, 1)]);
    }
    // only the upper triangle is evaluated; the symmetric entries are mirrored
}
