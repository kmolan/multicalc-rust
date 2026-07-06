//! Sensor-calibration curve fit on bare metal: the same y = a·e^(b·t) Levenberg–Marquardt fit as
//! examples/curve_fit.rs, built for Cortex-M with zero heap, no unsafe, and exact autodiff
//! Jacobians. There is no console on-chip, so the recovered parameters and a `converged` flag are
//! kept live via `black_box` (inspect under a debugger); the program then idles.
//!
//! WCET / cost model (worst case per call):
//!   ≤ MAX_ITERS × (1 residual eval + 1 Jacobian [8×2, via 2 dual evals] + QR(8,2) + ≤10 LMPAR solves)
//! MAX_ITERS caps the outer loop, so work is bounded. Run as a ONE-SHOT calibration OFF the control
//! loop (startup / low-rate task) — never inside a real-time tick.
//!
//! Build (does NOT run on host; needs a Cortex-M target — add them once with `rustup target add`):
//!   cargo build --example embedded_curve_fit --no-default-features --features embedded \
//!     --target thumbv7em-none-eabi     # soft-float   (or: cargo fit-eabi)
//!   cargo build --example embedded_curve_fit --no-default-features --features embedded \
//!     --target thumbv7em-none-eabihf   # hardware-FPU (or: cargo fit-eabihf)

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use multicalc::LevenbergMarquardt;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::{Numeric, VectorFn};

const A_TRUE: f64 = 100.0;
const B_TRUE: f64 = -0.5;
const PARAM_TOL: f64 = 1e-9;
const OBJ_TOL: f64 = 1e-12;
const MAX_ITERS: usize = 50;

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

#[entry]
fn main() -> ! {
    let t: [f64; 8] = core::array::from_fn(|i| i as f64);
    let y: [f64; 8] = core::array::from_fn(|i| A_TRUE * (B_TRUE * i as f64).exp());
    let problem = SensorFit { t, y };

    let solver = LevenbergMarquardt::<AutoDiffMulti>::default().with_patience(MAX_ITERS);

    // Panic-free: no unwrap/expect on the library path; a bad fit just leaves converged = false.
    let mut converged = false;
    if let Ok(report) = solver.minimize(&problem, &[80.0, -0.3]) {
        let da = (report.solution[0] - A_TRUE).abs();
        let db = (report.solution[1] - B_TRUE).abs();
        converged = da < PARAM_TOL && db < PARAM_TOL && report.objective_function < OBJ_TOL;
        core::hint::black_box(&report.solution);
    }
    core::hint::black_box(converged);

    loop {}
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
