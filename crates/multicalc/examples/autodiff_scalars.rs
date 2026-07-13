//! Using autodiff scalar types directly (`Dual`, `HyperDual`).
//!
//! Run with: `cargo run --example autodiff_scalars`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::scalar::{Dual, HyperDual, Numeric};

/// f(x) = x^2 + sin(x) — generic over the scalar, so one definition drives f64,
/// Dual, and HyperDual on the same code path.
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
    let (s, c) = (x.sin(), x.cos());
    let (fv, fp, fpp) = (
        x * x * s,
        2.0 * x * s + x * x * c,
        2.0 * s + 4.0 * x * c - x * x * s,
    );

    println!("f(x) = x^2 sin(x) at x = {x}");

    // (1) Dual: one pass gives f and f'
    let dual = f(Dual::variable(x));
    report("f (Dual.value)", dual.value, fv);
    report("f' (Dual.deriv)", dual.deriv, fp);

    // (2) HyperDual: one pass give f, f', and f''
    let hyper = f(HyperDual::variable(x));
    report("f (HyperDual.real)", hyper.real, fv);
    report("f' (HyperDual.eps1)", hyper.eps1, fp);
    report("f'' (HyperDual.eps1eps2)", hyper.eps1eps2, fpp);

    // (3) Generic over Numeric: plain f64 and Dual share the same function
    let plain = f(x);
    let dual = f(Dual::variable(x));

    println!("\nGeneric over Numeric - same fn, two scalar types:");
    report("f(1.0)", plain, fv);
    report("Dual.value", dual.value, fv);
    report("Dual.deriv", dual.deriv, fp);

    assert!((plain - dual.value).abs() < 1e-12);
    assert!((dual.deriv - fp).abs() < 1e-12);
}
