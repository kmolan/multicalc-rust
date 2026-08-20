//! Vector-field calculus: curl, divergence, line integrals and flux integrals.
//!
//! Run with: `cargo run -p multicalc-demos --example vector_field`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::AutoDiffMulti;
use multicalc::scalar::constant;
use multicalc::scalar_fn_vec;
use multicalc::vector_field::{curl_2d, divergence_2d, flux_integral_2d, line_integral_2d};

fn main() {
    // ---- curl & divergence of the 2D field (2xy, 3cos y), by autodiff (exact) ----
    let field = scalar_fn_vec!(|point: &[f64; 2]| [
        constant(2.0) * point[0] * point[1],
        constant(3.0) * point[1].cos()
    ]);
    let point = [1.0, std::f64::consts::PI];

    let curl_2d = curl_2d(AutoDiffMulti::default(), &field, &point).unwrap();
    let div_2d = divergence_2d(AutoDiffMulti::default(), &field, &point).unwrap();
    assert!((curl_2d + 2.0).abs() < 1e-9, "curl");
    assert!((div_2d - std::f64::consts::TAU).abs() < 1e-9, "divergence");
    println!("field (2xy, 3cos y) at {point:?}");
    println!("  curl       = {curl_2d:.4}   (exact -2)");
    println!(
        "  divergence = {div_2d:.4}   (exact 2*pi = {:.4})",
        std::f64::consts::TAU
    );

    // ---- line & flux integral of the field (y, -x) over the unit circle ----
    // the field components take the curve position [x, y]; the transforms map t -> x, t -> y
    let components: [&dyn Fn(&[f64; 2]) -> f64; 2] = [
        &(|point: &[f64; 2]| point[1]),
        &(|point: &[f64; 2]| -point[0]),
    ];
    let curve: [&dyn Fn(f64) -> f64; 2] = [&(|time: f64| time.cos()), &(|time: f64| time.sin())];
    let limit = [0.0, 2.0 * std::f64::consts::PI];

    let line = line_integral_2d(&components, &curve, &limit).unwrap();
    let flux = flux_integral_2d(&components, &curve, &limit).unwrap();
    println!("\nfield (y, -x) over the unit circle");
    println!(
        "  line integral = {line:.4}   (exact -2*pi = {:.4})",
        -2.0 * std::f64::consts::PI
    );
    println!("  flux integral = {flux:.4}   (exact 0)");
}
