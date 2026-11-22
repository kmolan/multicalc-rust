//! Streams a Levenberg-Marquardt curve fit (`y = a·e^(b·t)`) to a live Rerun viewer.
//!
//! Requires the `rerun` viewer (version 0.33.1) on PATH; see demos/README.md.
//! Run with: cargo run -p multicalc-demos --example curve_fit_live

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::LevenbergMarquardt;
use multicalc::numerical_derivative::AutoDiffMulti;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc_demos::{RerunSink, VizError, VizSink};

const A_TRUE: f64 = 100.0;
const B_TRUE: f64 = -0.5;
const N_SAMPLES: usize = 8;

// Residuals of y = a·e^(b·t), generic over the scalar so autodiff supplies the Jacobian.
struct SensorFit {
    times: [f64; N_SAMPLES],
    y: [f64; N_SAMPLES],
}

impl VectorFn<2, N_SAMPLES> for SensorFit {
    fn eval<S: Numeric>(&self, params: &[S; 2]) -> [S; N_SAMPLES] {
        let (a, b) = (params[0], params[1]);
        core::array::from_fn(|i| {
            a * (b * S::from_f64(self.times[i])).exp() - S::from_f64(self.y[i])
        })
    }
}

fn main() -> Result<(), VizError> {
    let times: [f64; N_SAMPLES] = core::array::from_fn(|i| i as f64);
    let y: [f64; N_SAMPLES] = core::array::from_fn(|i| A_TRUE * (B_TRUE * i as f64).exp());
    let problem = SensorFit { times, y };

    // Deliberately away from the truth, so the fit has work to do.
    let initial_guess = [80.0, -0.3];

    let report = LevenbergMarquardt::<AutoDiffMulti>::default()
        .minimize(&problem, &initial_guess)
        .expect("curve fit did not converge");
    let (a, b) = (report.solution[0], report.solution[1]);
    let fit = |sample_t: f64| a * (b * sample_t).exp();

    // Spawns the viewer and streams data scatter, fitted curve, and residual series.
    let mut sink = RerunSink::live("multicalc-demos/curve-fit")?;

    let data_pts: Vec<[f64; 2]> = (0..N_SAMPLES).map(|i| [times[i], y[i]]).collect();
    sink.points2d("data", &data_pts)?;

    let steps = 100;
    let (t_lo, t_hi) = (times[0], times[N_SAMPLES - 1]);
    let curve: Vec<[f64; 2]> = (0..=steps)
        .map(|k| {
            let sample_t = t_lo + (t_hi - t_lo) * (k as f64) / (steps as f64);
            [sample_t, fit(sample_t)]
        })
        .collect();
    sink.points2d("fit", &curve)?;

    for i in 0..N_SAMPLES {
        sink.set_sequence("sample", i as i64);
        sink.scalar("residual", fit(times[i]) - y[i])?;
    }
    sink.scalar("objective", report.objective_function)?;
    sink.flush()?;
    Ok(())
}
