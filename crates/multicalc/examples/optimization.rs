//! Gauss-Newton vs trust-region (Levenberg–Marquardt) on the same residual problem.
//!
//! Reach for [`GaussNewton`] when the problem is well-conditioned and you are already near the
//! solution — it takes undamped steps and is usually the fastest of the two.
//! Reach for [`LevenbergMarquardt`] (trust-region / damped) when the start is far from the
//! optimum or the Jacobian may be ill-conditioned; damping keeps steps inside a trusted region.
//!
//! Run with: `cargo run -p multicalc --example optimization`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::GaussNewton;
use multicalc::LevenbergMarquardt;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::{Numeric, VectorFn};

const A_TRUE: f64 = 100.0;
const B_TRUE: f64 = -0.5;
const PARAM_TOL: f64 = 1e-8;

/// Fit y = a·e^(b·t) residuals, generic over the scalar for autodiff Jacobians.
struct ExpFit<const M: usize> {
    t: [f64; M],
    y: [f64; M],
}

impl<const M: usize> VectorFn<2, M> for ExpFit<M> {
    fn eval<S: Numeric>(&self, p: &[S; 2]) -> [S; M] {
        let (a, b) = (p[0], p[1]);
        core::array::from_fn(|i| a * (b * S::from_f64(self.t[i])).exp() - S::from_f64(self.y[i]))
    }
}

fn main() {
    let t: [f64; 8] = core::array::from_fn(|i| i as f64);
    let y: [f64; 8] = core::array::from_fn(|i| A_TRUE * (B_TRUE * i as f64).exp());
    let problem = ExpFit { t, y };
    let start = [80.0, -0.3];

    // --- Gauss-Newton (undamped): prefer when well-conditioned / near solution ---
    let gn = GaussNewton::<AutoDiffMulti>::default()
        .with_backtracking(true)
        .minimize(&problem, &start)
        .expect("Gauss-Newton did not converge");
    println!("Gauss-Newton (undamped, with backtracking)");
    println!(
        "  a = {:.9}   b = {:.9}   objective = {:.1e}   ({:?})",
        gn.solution[0], gn.solution[1], gn.objective_function, gn.termination
    );
    assert!((gn.solution[0] - A_TRUE).abs() < PARAM_TOL);
    assert!((gn.solution[1] - B_TRUE).abs() < PARAM_TOL);

    // --- Levenberg-Marquardt (trust-region / damped): prefer for robustness ---
    let lm = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&problem, &start)
        .expect("Levenberg-Marquardt did not converge");
    println!("\nLevenberg-Marquardt (trust-region / damped)");
    println!(
        "  a = {:.9}   b = {:.9}   objective = {:.1e}   ({:?})",
        lm.solution[0], lm.solution[1], lm.objective_function, lm.termination
    );
    assert!((lm.solution[0] - A_TRUE).abs() < PARAM_TOL);
    assert!((lm.solution[1] - B_TRUE).abs() < PARAM_TOL);

    println!("\nBoth solvers recovered a = {A_TRUE}, b = {B_TRUE}.");
}
