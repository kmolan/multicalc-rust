//! Vector-field calculus: curl, divergence, line integrals and flux integrals.
//!
//! Run with: `cargo run --example vector_field`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::c;
use multicalc::scalar_fn_vec;
use multicalc::vector_field::{curl, divergence, flux_integral, line_integral};

fn main() {
    // ---- curl & divergence of the 2D field (2xy, 3cos y), by autodiff (exact) ----
    let field = scalar_fn_vec!(|v: &[f64; 2]| [c(2.0) * v[0] * v[1], c(3.0) * v[1].cos()]);
    let point = [1.0, std::f64::consts::PI];

    let curl_2d = curl::get_2d(AutoDiffMulti::default(), &field, &point).unwrap();
    let div_2d = divergence::get_2d(AutoDiffMulti::default(), &field, &point).unwrap();
    println!("field (2xy, 3cos y) at {point:?}");
    println!("  curl       = {curl_2d:.4}   (exact -2)");
    println!(
        "  divergence = {div_2d:.4}   (exact 2*pi = {:.4})",
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
