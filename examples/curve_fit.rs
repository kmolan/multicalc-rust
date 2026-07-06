//! Sensor-calibration curve fit: recover a and b in y = a·e^(b·t) from samples using
//! Levenberg–Marquardt with exact autodiff Jacobians (no hand-derived derivatives), zero heap.
//! The identical fit builds for bare-metal Cortex-M — see examples/embedded_curve_fit.rs.
//!
//! Run with: cargo run --example curve_fit   (or: cargo fit)

use multicalc::LevenbergMarquardt;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::{Numeric, VectorFn};

const A_TRUE: f64 = 100.0;
const B_TRUE: f64 = -0.5;
const PARAM_TOL: f64 = 1e-9;
const OBJ_TOL: f64 = 1e-12;

// y = a·e^(b·t), generic over the scalar so autodiff differentiates the residuals for free.
struct SensorFit<const M: usize> {
    t: [f64; M],
    y: [f64; M],
}

impl<const M: usize> VectorFn<2, M> for SensorFit<M> {
    fn eval<S: Numeric>(&self, p: &[S; 2]) -> [S; M] {
        let (a, b) = (p[0], p[1]);
        core::array::from_fn(|i| a * (b * S::from_f64(self.t[i])).exp() - S::from_f64(self.y[i]))
    }
}

fn main() {
    let t: [f64; 8] = core::array::from_fn(|i| i as f64);
    let y: [f64; 8] = core::array::from_fn(|i| A_TRUE * (B_TRUE * i as f64).exp());
    let problem = SensorFit { t, y };

    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&problem, &[80.0, -0.3])
        .expect("curve fit did not converge");

    let (a, b) = (report.solution[0], report.solution[1]);
    let (da, db) = ((a - A_TRUE).abs(), (b - B_TRUE).abs());
    println!("fit y = a·e^(b·t) to 8 samples");
    println!("  a = {a:.9}   |err| = {da:.1e}");
    println!("  b = {b:.9}   |err| = {db:.1e}");
    println!(
        "  objective = {:.1e}   ({} evals, {:?})",
        report.objective_function, report.evaluations, report.termination
    );

    let converged = da < PARAM_TOL && db < PARAM_TOL && report.objective_function < OBJ_TOL;
    assert!(converged, "fit missed the shared tolerance");
}
