//! Root finding: bracketed bisection, Newton with exact derivatives, damped Newton, and a
//! square-system Newton solve. Each result prints against its known root with the `|err|`.
//!
//! Run with: `cargo run --example root_finding`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::root_finding::{Bisection, Newton, NewtonSystem};
use multicalc::scalar::c;
use multicalc::{scalar_fn, scalar_fn_vec};

fn main() {
    // ---- Bisection: Wien's displacement law, x - 5 + 5*e^-x = 0 ----
    // The root 4.965114231744276 fixes the peak of the blackbody spectrum.
    let wien = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let wien_true = 4.965114231744276_f64;
    let r = Bisection::default().solve(&wien, 1.0, 10.0).unwrap();
    println!("Bisection  Wien's displacement root on [1, 10]");
    println!(
        "  root = {:.15}   |err| = {:.1e}   ({} iters, {:?})",
        r.root,
        (r.root - wien_true).abs(),
        r.iterations,
        r.termination
    );

    // ---- Newton: x^2 - 2 = 0, exact derivative via Dual numbers ----
    let f = scalar_fn!(|x| c(-2.0) + x * x);
    let sqrt2 = 2.0_f64.sqrt();
    let r = Newton::<AutoDiffSingle>::default().solve(&f, 2.0).unwrap();
    println!("\nNewton  x^2 - 2 = 0 (exact derivative, x0 = 2)");
    println!(
        "  root = {:.15}   |err| = {:.1e}   ({} iters, {:?})",
        r.root,
        (r.root - sqrt2).abs(),
        r.iterations,
        r.termination
    );

    // ---- Damped Newton: x / sqrt(1 + x^2), root at 0 ----
    // The plain Newton map is x -> -x^3, so from x0 = 2 it diverges. The
    // backtracking line search halves the step until |f| decreases.
    let g = scalar_fn!(|x| x / (c(1.0) + x * x).sqrt());
    let plain = Newton::<AutoDiffSingle>::default().solve(&g, 2.0);
    let damped = Newton::<AutoDiffSingle>::default()
        .with_backtracking(true)
        .solve(&g, 2.0)
        .unwrap();
    println!("\nDamped Newton  x / sqrt(1 + x^2), root at 0, from x0 = 2");
    println!("  plain Newton  -> {plain:?}");
    println!(
        "  damped Newton -> root = {:.3e}   |err| = {:.1e}   ({} iters)",
        damped.root,
        damped.root.abs(),
        damped.iterations
    );

    // ---- Newton system: x^2 + y^2 = 4 and x*y = 1 (circle and hyperbola) ----
    let system =
        scalar_fn_vec!(
            |v: &[f64; 2]| [c(-4.0) + v[0] * v[0] + v[1] * v[1], c(-1.0) + v[0] * v[1],]
        );
    let x_true = (2.0_f64 + 3.0_f64.sqrt()).sqrt();
    let y_true = (2.0_f64 - 3.0_f64.sqrt()).sqrt();
    let r = NewtonSystem::<AutoDiffMulti>::default()
        .solve(&system, &[1.5, 0.8])
        .unwrap();
    let err = (r.root[0] - x_true).abs().max((r.root[1] - y_true).abs());
    println!("\nNewton system  x^2 + y^2 = 4 and x*y = 1");
    println!("  root = [{:.12}, {:.12}]", r.root[0], r.root[1]);
    println!(
        "  |err| = {err:.1e}   norm(F) = {:.1e}   ({} iters, {:?})",
        r.residual_norm, r.iterations, r.termination
    );
}
