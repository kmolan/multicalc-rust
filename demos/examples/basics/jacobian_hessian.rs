//! Jacobian and Hessian matrices of multi-variable functions.
//!
//! Run with: `cargo run -p multicalc-demos --example jacobian_hessian`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::hessian::Hessian;
use multicalc::numerical_derivative::jacobian::Jacobian;
use multicalc::scalar::c;
use multicalc::{scalar_fn, scalar_fn_vec};

fn main() {
    // ---- Jacobian of the vector function (x*y*z, x^2 + y^2) ----
    let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
    let point = [1.0, 2.0, 3.0];

    let jacobian: Jacobian = Jacobian::default();
    let result = jacobian.get(&f, &point).unwrap();

    println!("Jacobian of (x*y*z, x^2 + y^2) at {point:?}:");
    for row in &result {
        println!("  [{:.4}, {:.4}, {:.4}]", row[0], row[1], row[2]);
    }
    println!("  (exact [[6, 3, 2], [2, 4, 0]])");
    let exact = [[6.0, 3.0, 2.0], [2.0, 4.0, 0.0]];
    for i in 0..2 {
        for j in 0..3 {
            assert!((result[i][j] - exact[i][j]).abs() < 1e-9);
        }
    }

    // ---- Hessian of f(x, y) = y*sin(x) + 2*x*e^y ----
    let g = scalar_fn!(|v: &[f64; 2]| v[1] * v[0].sin() + c(2.0) * v[0] * v[1].exp());
    let hessian: Hessian = Hessian::default();
    let result = hessian.get(&g, &[1.0, 2.0]).unwrap();

    println!("\nHessian of y*sin(x) + 2*x*e^y at [1, 2]:");
    for row in &result {
        println!("  [{:.4}, {:.4}]", row[0], row[1]);
    }
    // only the upper triangle is evaluated; the symmetric entries are mirrored
}
