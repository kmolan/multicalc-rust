//! Root finding: bracketed bisection, Newton with exact derivatives, damped Newton, and a
//! square-system Newton solve. Each result prints against its known root with the `|err|`.
//!
//! Run with: `cargo run -p multicalc-demos --example root_finding`

use multicalc::{Bisection, CalcError, Newton, NewtonSystem, c};
use multicalc::{scalar_fn, scalar_fn_vec};

// Every solve below propagates with `?`. A solver that runs out of iterations, or is handed a
// bracket that does not straddle a root, returns `SolveError`, which converts into the `CalcError`
// umbrella on the way out.
fn main() -> Result<(), CalcError> {
    // ---- Bisection: Wien's displacement law, x - 5 + 5*e^-x = 0 ----
    // The root 4.965114231744276 fixes the peak of the blackbody spectrum.
    let wien = scalar_fn!(|x| c(-5.0) + x + c(5.0) * (-x).exp());
    let wien_true = 4.965114231744276_f64;
    let r = Bisection::default().solve(&wien, 1.0, 10.0)?;
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
    let r = Newton::new().solve(&f, 2.0)?;
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
    // The plain solve is expected to fail, so its `Result` is printed rather than propagated.
    let plain = Newton::new().solve(&g, 2.0);
    let damped = Newton::new().with_backtracking(true).solve(&g, 2.0)?;
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
    let r = NewtonSystem::new().solve(&system, &[1.5, 0.8])?;
    let err = (r.root[0] - x_true).abs().max((r.root[1] - y_true).abs());
    assert!(
        err < 1e-9,
        "Newton system should converge to the intersection"
    );
    println!("\nNewton system  x^2 + y^2 = 4 and x*y = 1");
    println!("  root = [{:.12}, {:.12}]", r.root[0], r.root[1]);
    println!(
        "  |err| = {err:.1e}   norm(F) = {:.1e}   ({} iters, {:?})",
        r.residual_norm, r.iterations, r.termination
    );

    Ok(())
}
