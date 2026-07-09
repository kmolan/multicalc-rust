//! Use autodiff scalar types (Dual, HyperDual) directly.
//!
//! Run with: cargo run -p multicalc --example autodiff_scalars

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::{Dual, HyperDual, Numeric};

/// f(x) = x^2 * sin(x) — written once, generic over the scalar type.
fn f<T: Numeric>(x: T) -> T {
    x * x * x.sin()
}

fn main() {
    let x = 1.0_f64;

    // (1) Dual: value + first derivative in one pass
    let d = f(Dual::variable(x));
    // exact: f = x^2 sin x, f' = 2x sin x + x^2 cos x
    let (s, c) = (x.sin(), x.cos());
    let f_exact = x * x * s;
    let fp_exact = 2.0 * x * s + x * x * c;
    println!(\"Dual at x = {x}\");
    println!(\"  f  = {:.12}   (exact {:.12})\", d.value, f_exact);
    println!(\"  f' = {:.12}   (exact {:.12})\", d.deriv, fp_exact);
    assert!((d.value - f_exact).abs() < 1e-14);
    assert!((d.deriv - fp_exact).abs() < 1e-14);

    // (2) HyperDual: value + first + second derivative
    let h = f(HyperDual::variable(x));
    // f'' = 2 sin x + 4x cos x - x^2 sin x
    let fpp_exact = 2.0 * s + 4.0 * x * c - x * x * s;
    println!(\"\\nHyperDual at x = {x}\");
    println!(\"  f   = {:.12}   (exact {:.12})\", h.real, f_exact);
    println!(\"  f'  = {:.12}   (exact {:.12})\", h.eps1, fp_exact);
    println!(\"  f'' = {:.12}   (exact {:.12})\", h.eps1eps2, fpp_exact);
    assert!((h.real - f_exact).abs() < 1e-14);
    assert!((h.eps1 - fp_exact).abs() < 1e-14);
    assert!((h.eps1eps2 - fpp_exact).abs() < 1e-14);

    // (3) Generic-over-scalar: same  with plain f64 and Dual
    let plain = f(x);
    let dual_val = f(Dual::variable(x)).value;
    println!(\"\\nGeneric scalar path\");
    println!(\"  f::<f64>(x)        = {:.12}\", plain);
    println!(\"  f::<Dual<_>>(x).value = {:.12}\", dual_val);
    assert!((plain - dual_val).abs() < 1e-14);

    println!(\"\\nAll autodiff scalar checks passed.\");
}
