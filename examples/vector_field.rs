//! Vector-field calculus: curl, divergence, line integrals and flux integrals.
//!
//! Run with: `cargo run --example vector_field`

use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
use multicalc::vector_field::{curl, divergence, flux_integral, line_integral};

fn main() {
    let derivator = FiniteDifferenceMulti::default();

    // ---- curl & divergence of the 2D field (2xy, 3cos y) ----
    let vf_x = |v: &[f64; 2]| 2.0 * v[0] * v[1];
    let vf_y = |v: &[f64; 2]| 3.0 * v[1].cos();
    let field: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&vf_x, &vf_y];
    let point = [1.0, std::f64::consts::PI];

    let c = curl::get_2d(derivator, &field, &point).unwrap();
    let d = divergence::get_2d(derivator, &field, &point).unwrap();
    println!("field (2xy, 3cos y) at {point:?}");
    println!("  curl       = {c:.4}   (exact -2)");
    println!(
        "  divergence = {d:.4}   (exact 2*pi = {:.4})",
        std::f64::consts::TAU
    );

    // ---- line & flux integral of the field (y, -x) over the unit circle ----
    // the field components take the curve position [x, y]; the transforms map t -> x, t -> y
    let g: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&(|v: &[f64; 2]| v[1]), &(|v: &[f64; 2]| -v[0])];
    let curve: [&dyn Fn(f64) -> f64; 2] = [&(|t: f64| t.cos()), &(|t: f64| t.sin())];
    let limit = [0.0, 2.0 * std::f64::consts::PI];

    let line = line_integral::get_2d(&g, &curve, &limit).unwrap();
    let flux = flux_integral::get_2d(&g, &curve, &limit).unwrap();
    println!("\nfield (y, -x) over the unit circle");
    println!(
        "  line integral = {line:.4}   (exact -2*pi = {:.4})",
        -2.0 * std::f64::consts::PI
    );
    println!("  flux integral = {flux:.4}   (exact 0)");
}
