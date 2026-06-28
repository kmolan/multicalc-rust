//! Jacobian and Hessian matrices of multi-variable functions.
//!
//! Run with: `cargo run --example jacobian_hessian`

use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
use multicalc::numerical_derivative::hessian::Hessian;
use multicalc::numerical_derivative::jacobian::Jacobian;

fn main() {
    // ---- Jacobian of the vector function (x*y*z, x^2 + y^2) ----
    let f1 = |v: &[f64; 3]| v[0] * v[1] * v[2];
    let f2 = |v: &[f64; 3]| v[0] * v[0] + v[1] * v[1];
    // a heterogeneous array of functions is the one place a `&dyn Fn` array is used
    let functions: [&dyn Fn(&[f64; 3]) -> f64; 2] = [&f1, &f2];
    let point = [1.0, 2.0, 3.0];

    let jacobian = Jacobian::<FiniteDifferenceMulti>::default();
    let result = jacobian.get(&functions, &point).unwrap();

    println!("Jacobian of (x*y*z, x^2 + y^2) at {point:?}:");
    for row in &result {
        println!("  [{:.4}, {:.4}, {:.4}]", row[0], row[1], row[2]);
    }
    println!("  (exact [[6, 3, 2], [2, 4, 0]])");

    // ---- Hessian of f(x, y) = y*sin(x) + 2*x*e^y ----
    let g = |v: &[f64; 2]| v[1] * v[0].sin() + 2.0 * v[0] * v[1].exp();
    let hessian = Hessian::<FiniteDifferenceMulti>::default();
    let result = hessian.get(&g, &[1.0, 2.0]).unwrap();

    println!("\nHessian of y*sin(x) + 2*x*e^y at [1, 2]:");
    for row in &result {
        println!("  [{:.4}, {:.4}]", row[0], row[1]);
    }
    // only the upper triangle is evaluated; the symmetric entries are mirrored
}
