//! 1 kHz IK servo arm (optimization showcase).
//!
//! A planar 3-link arm chases a moving target. Every tick it runs a complete Levenberg-Marquardt
//! solve with exact autodiff Jacobians, and the panel shows that math costing a few percent of the
//! 1 ms budget. Two position residuals plus three posture regularizers keep the system
//! over-determined (LM needs M >= N) and resolve the arm's redundant degree of freedom, so the
//! pose is unique and chatter-free.
//!
//! Timing model: the simulation advances on logical time (a fixed 1 ms per tick), so the numbers
//! are deterministic — the pacer only decides *when* a tick is displayed. `plots/solve_us` is
//! multicalc's math cost; host-OS scheduling lateness is measured too but shown only as a hud
//! percentile (not a plot), since it is the OS, not the library, and never perturbs the computed
//! result. The headline is the math cost and its headroom under the 1 ms budget, not a claim of
//! hard real-time on a desktop OS.
//!
//! Streams live to a Rerun viewer; see showcase/viz/README.md for the WSL setup.
//! Run with: cargo run --release -p multicalc-viz --example ik_servo

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::LevenbergMarquardt;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc_viz::loop_util::{LatencyRing, Pacer};
use multicalc_viz::{RerunSink, Rgba, VizError, VizSink};
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::time::Instant;

// Palette (§2), dark-mode column, sRGB with alpha.
const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // arm, joints, solved pose
const TARGET: Rgba = [0xc9, 0x85, 0x00, 0xff]; // target dot
const ERROR: Rgba = [0xe6, 0x67, 0x67, 0xff]; // residual series
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 140]; // ghost trail
const CHROME: Rgba = [0x89, 0x87, 0x81, 0xff]; // grid, reach circle
const TARGET_PATH: Rgba = [0xc9, 0x85, 0x00, 90]; // upcoming target path

const L: [f64; 3] = [1.0, 0.8, 0.6]; // link lengths; reach 2.4
const REG: f64 = 0.05; // sqrt of posture weight

const GEOM_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats
const TRAIL_MAX: usize = 187; // ~3 s of ee positions at 60 Hz
const PATH_PTS: usize = 200; // decimated upcoming-path samples
const PATH_HORIZON: f64 = 2.0; // seconds of target path drawn ahead
const REACH_SEGS: usize = 128;

/// The IK residual system: 2 position residuals + 3 posture regularizers.
struct IkProblem {
    target: [f64; 2],
    prev: [f64; 3],
}

impl VectorFn<3, 5> for IkProblem {
    fn eval<S: Numeric>(&self, q: &[S; 3]) -> [S; 5] {
        let mut a = S::ZERO; // cumulative angle
        let (mut x, mut y) = (S::ZERO, S::ZERO);
        for i in 0..3 {
            a += q[i];
            x += S::from_f64(L[i]) * a.cos();
            y += S::from_f64(L[i]) * a.sin();
        }
        let w = S::from_f64(REG);
        [
            x - S::from_f64(self.target[0]),
            y - S::from_f64(self.target[1]),
            w * (q[0] - S::from_f64(self.prev[0])),
            w * (q[1] - S::from_f64(self.prev[1])),
            w * (q[2] - S::from_f64(self.prev[2])),
        ]
    }
}

/// Target position on the two-tone Lissajous (max radius 2.0 < reach 2.4, always reachable).
fn lissajous(t: f64) -> [f64; 2] {
    [
        1.6 * (TAU * 0.13 * t).sin(),
        1.2 * (TAU * 0.21 * t + 0.5).sin(),
    ]
}

/// The 4 chain nodes (base, joint1, joint2, ee) for the given joint angles.
fn fk_nodes(q: &[f64; 3]) -> [[f64; 2]; 4] {
    let mut nodes = [[0.0; 2]; 4];
    let (mut a, mut x, mut y) = (0.0, 0.0, 0.0);
    for i in 0..3 {
        a += q[i];
        x += L[i] * a.cos();
        y += L[i] * a.sin();
        nodes[i + 1] = [x, y];
    }
    nodes
}

/// Static 1-unit grid over [-2.6, 2.6]^2 as a batch of 2-point strips.
fn grid_strips() -> Vec<Vec<[f64; 2]>> {
    let (lo, hi) = (-2.6, 2.6);
    let mut strips = Vec::new();
    for k in -2..=2 {
        let c = k as f64;
        strips.push(vec![[c, lo], [c, hi]]);
        strips.push(vec![[lo, c], [hi, c]]);
    }
    strips
}

/// The reach circle (r = 2.4) as one closed strip.
fn reach_circle() -> Vec<[f64; 2]> {
    (0..=REACH_SEGS)
        .map(|i| {
            let th = TAU * i as f64 / REACH_SEGS as f64;
            [2.4 * th.cos(), 2.4 * th.sin()]
        })
        .collect()
}

/// The next `PATH_HORIZON` seconds of the target path from time `t`, decimated to `PATH_PTS`.
fn target_path(t: f64) -> Vec<[f64; 2]> {
    (0..=PATH_PTS)
        .map(|i| lissajous(t + PATH_HORIZON * i as f64 / PATH_PTS as f64))
        .collect()
}

fn main() -> Result<(), VizError> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-viz --example ik_servo"
        );
    }

    let mut rr = RerunSink::live("multicalc-viz/ik-servo")?;

    // Statics: stamp at tick 0 so they forward-fill across the run (see rerun-viz-gotchas).
    rr.set_sequence("tick", 0);
    rr.line_strips2d("world/grid", &grid_strips(), &[CHROME], &[0.004])?;
    rr.line_strips2d("world/reach", &[reach_circle()], &[CHROME], &[0.006])?;
    rr.series_style("plots/residual_norm", ERROR, "position residual ‖r‖", 2.0)?;

    let lm = LevenbergMarquardt::<AutoDiffMulti>::default().with_patience(40);
    let mut problem = IkProblem {
        target: lissajous(0.0),
        prev: [0.4, 0.4, 0.4], // a bent, non-singular start pose
    };

    let mut pacer = Pacer::new();
    let mut solve_ring = LatencyRing::new(1024);
    let mut trail: VecDeque<[f64; 2]> = VecDeque::with_capacity(TRAIL_MAX);

    let mut last_residual_norm = 0.0; // sqrt(2*objective), full system
    let mut last_pos_residual = 0.0; // ||ee - target||, the accuracy readout
    let mut evals_live: usize = 0; // residual evaluations on the latest solve
    let mut evals_sum: u64 = 0; // running total for the average
    let mut evals_n: u64 = 0;

    let mut n: i64 = 0;
    loop {
        pacer.wait(); // pace to the next 1 ms boundary
        n += 1;
        let t = n as f64 / 1000.0;
        rr.set_sequence("tick", n);

        problem.target = lissajous(t);
        let x0 = problem.prev; // copy out before the borrow, keeps the update below borrow-clean
        let t0 = Instant::now();
        let result = lm.minimize(&problem, &x0);
        let solve_us = t0.elapsed().as_micros() as f64;

        // On Err, hold the last pose (never panic); Ok updates the pose and readouts.
        if let Ok(rep) = result {
            problem.prev = rep.solution;
            last_residual_norm = (2.0 * rep.objective_function).sqrt();
            evals_live = rep.evaluations;
            evals_sum += rep.evaluations as u64;
            evals_n += 1;
            let ee = fk_nodes(&rep.solution)[3];
            let dx = ee[0] - problem.target[0];
            let dy = ee[1] - problem.target[1];
            last_pos_residual = (dx * dx + dy * dy).sqrt();
        }

        if n > WARMUP_TICKS {
            solve_ring.push(solve_us);
        }

        // Residual-norm series (solve time is summarized in the hud, not plotted).
        rr.scalar("plots/residual_norm", last_residual_norm)?;

        // Spatial geometry at ~60 Hz.
        if n % GEOM_EVERY == 0 {
            let nodes = fk_nodes(&problem.prev);
            let ee = nodes[3];
            if trail.len() == TRAIL_MAX {
                trail.pop_front();
            }
            trail.push_back(ee);

            rr.line_strips2d("world/arm", &[nodes.to_vec()], &[HERO], &[0.035])?;
            rr.points2d_styled(
                "world/arm/joints",
                &nodes,
                &[HERO],
                &[0.06, 0.05, 0.05, 0.04],
            )?;
            rr.points2d_styled("world/target", &[problem.target], &[TARGET], &[0.07])?;
            rr.line_strips2d(
                "world/target/path",
                &[target_path(t)],
                &[TARGET_PATH],
                &[0.008],
            )?;
            let trail_pts: Vec<[f64; 2]> = trail.iter().copied().collect();
            rr.line_strips2d("world/trail", &[trail_pts], &[ACCENT], &[0.012])?;
        }

        // Hud headline at 1 Hz.
        if n % HUD_EVERY == 0
            && let Some(s) = solve_ring.summary()
        {
            let evals_avg = evals_sum as f64 / evals_n.max(1) as f64;
            let md = format!(
                "## ik_servo — multicalc live demo\n\
                 ### full IK solve (Levenberg–Marquardt, exact autodiff Jacobian): median {:.0} µs — {:.2} % of the 1 ms tick\n\
                 ### accuracy: position residual ‖ee−target‖ = {:.1e}\n\
                 ### LM residual evaluations: {} this tick, {:.1} average",
                s.median,
                s.median / 10.0,
                last_pos_residual,
                evals_live,
                evals_avg,
            );
            rr.text("hud/stats", &md)?;
        }
    }
}
