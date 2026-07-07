//! Streams a Levenberg-Marquardt curve fit (`y = a·e^(b·t)`) to a live Rerun viewer.
//!
//! Requires the `rerun` viewer (version 0.33.1) on PATH; see showcase/viz/README.md.
//! Run with: cargo run -p multicalc-viz --example curve_fit_live

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::LevenbergMarquardt;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc_viz::{RerunSink, VizError, VizSink};

const A_TRUE: f64 = 100.0;
const B_TRUE: f64 = -0.5;
const M: usize = 8;

// Residuals of y = a·e^(b·t), generic over the scalar so autodiff supplies the Jacobian.
struct SensorFit {
    t: [f64; M],
    y: [f64; M],
}

impl VectorFn<2, M> for SensorFit {
    fn eval<S: Numeric>(&self, p: &[S; 2]) -> [S; M] {
        let (a, b) = (p[0], p[1]);
        core::array::from_fn(|i| a * (b * S::from_f64(self.t[i])).exp() - S::from_f64(self.y[i]))
    }
}

fn main() -> Result<(), VizError> {
    let t: [f64; M] = core::array::from_fn(|i| i as f64);
    let y: [f64; M] = core::array::from_fn(|i| A_TRUE * (B_TRUE * i as f64).exp());
    let problem = SensorFit { t, y };

    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&problem, &[80.0, -0.3])
        .expect("curve fit did not converge");
    let (a, b) = (report.solution[0], report.solution[1]);
    let fit = |tt: f64| a * (b * tt).exp();

    // Spawns the viewer and streams data scatter, fitted curve, and residual series.
    let mut rr = RerunSink::live("multicalc-viz/curve-fit")?;

    let data_pts: Vec<[f64; 2]> = (0..M).map(|i| [t[i], y[i]]).collect();
    rr.points2d("data", &data_pts)?;

    let steps = 100;
    let (t0, t1) = (t[0], t[M - 1]);
    let curve: Vec<[f64; 2]> = (0..=steps)
        .map(|k| {
            let tt = t0 + (t1 - t0) * (k as f64) / (steps as f64);
            [tt, fit(tt)]
        })
        .collect();
    rr.points2d("fit", &curve)?;

    for i in 0..M {
        rr.set_sequence("sample", i as i64);
        rr.scalar("residual", fit(t[i]) - y[i])?;
    }
    rr.scalar("objective", report.objective_function)?;
    rr.flush()?;
    Ok(())
}
