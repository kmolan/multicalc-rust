//! Using autodiff scalar types directly (`Dual`, `HyperDual`).
//!
//! Run with: `cargo run -p multicalc-demos --example autodiff_scalars`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::scalar::{Dual, HyperDual, Numeric};

/// f(x) = x^2 + sin(x) — generic over the scalar, so one definition drives f64,
/// Dual, and HyperDual on the same code path.
#[must_use]
fn f<T: Numeric>(x: T) -> T {
    x * x * x.sin()
}

fn report(label: &str, value: f64, exact: f64) {
    println!(
        "  {label:<22} = {value:>13.8}   (exact {exact:>13.8}, |err| {:.0e})",
        (value - exact).abs()
    );
}

fn main() {
    let x = 1.0_f64;
    let (sine, cosine) = (x.sin(), x.cos());
    let (f_val, f_prime, fpp) = (
        x * x * sine,
        2.0 * x * sine + x * x * cosine,
        2.0 * sine + 4.0 * x * cosine - x * x * sine,
    );

    println!("f(x) = x^2 sin(x) at x = {x}");

    // (1) Dual: one pass gives f and f'
    let dual = f(Dual::variable(x));
    report("f (Dual.value)", dual.value, f_val);
    report("f' (Dual.deriv)", dual.deriv, f_prime);

    // (2) HyperDual: one pass give f, f', and f''
    let hyper = f(HyperDual::variable(x));
    report("f (HyperDual.real)", hyper.real, f_val);
    report("f' (HyperDual.eps1)", hyper.eps1, f_prime);
    report("f'' (HyperDual.eps1eps2)", hyper.eps1eps2, fpp);

    // (3) Generic over Numeric: plain f64 and Dual share the same function
    let plain = f(x);
    let dual = f(Dual::variable(x));

    println!("\nGeneric over Numeric - same fn, two scalar types:");
    report("f(1.0)", plain, f_val);
    report("Dual.value", dual.value, f_val);
    report("Dual.deriv", dual.deriv, f_prime);

    assert!((plain - dual.value).abs() < 1e-12);
    assert!((dual.deriv - f_prime).abs() < 1e-12);
}
