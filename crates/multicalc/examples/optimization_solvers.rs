//! Gauss-Newton least-squares example (well-conditioned linear residual).
//!
//! When to use which solver:
//! - **Gauss-Newton**: undamped, fast near a well-conditioned solution (this example).
//! - **Levenberg-Marquardt** (see curve_fit.rs): damped / trust-region style; prefer when
//!   far from the solution or the Jacobian is poorly conditioned.
//!
//! Run: cargo run -p multicalc --example optimization_solvers

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::GaussNewton;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::{Numeric, VectorFn};

// Fit y = a + b*t to points on y = 2t + 1; linear residuals converge in one GN step.
// Expected: a = 1.0, b = 2.0
struct LineFit;

impl VectorFn<2, 3> for LineFit {
    fn eval<S: Numeric>(&self, p: &[S; 2]) -> [S; 3] {
        // residuals at t = 0,1,2 with y = 1,3,5
        let (a, b) = (p[0], p[1]);
        [
            a + b * S::from_f64(0.0) - S::from_f64(1.0),
            a + b * S::from_f64(1.0) - S::from_f64(3.0),
            a + b * S::from_f64(2.0) - S::from_f64(5.0),
        ]
    }
}

fn main() {
    let report = GaussNewton::<AutoDiffMulti>::default()
        .minimize(&LineFit, &[0.0, 0.0])
        .expect("gauss-newton did not converge");

    let (a, b) = (report.solution[0], report.solution[1]);
    println!("Gauss-Newton fit y = a + b*t");
    println!("  a = {a:.9}   (expect 1.0)");
    println!("  b = {b:.9}   (expect 2.0)");
    println!(
        "  objective = {:.1e}   ({} evals, {:?})",
        report.objective_function, report.evaluations, report.termination
    );

    assert!((a - 1.0).abs() < 1e-9 && (b - 2.0).abs() < 1e-9);
}
