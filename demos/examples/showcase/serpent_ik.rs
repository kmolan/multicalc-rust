//! 1 kHz SE(3) tentacle IK (spatial + autodiff showcase).
//!
//! An 8-link tentacle chases a moving 3D target in position and orientation. Every millisecond a
//! full Levenberg-Marquardt solve runs whose Jacobian is a single `Dual` pushed through the entire
//! Lie composition — exp, log, compose — with no hand-derived kinematics. The panel shows the solve
//! cost against the 1 ms budget.
//!
//! The forward kinematics is written once, generic over the scalar type: the same code produces the
//! pose in `f64` and its derivative in `Dual`. Eight posture regularizers keep the system
//! over-determined (LM needs M >= N) and bias weakly toward the warm-start pose for continuity.
//!
//! Streams live to a Rerun viewer; see demos/README.md for the WSL setup.
//! Run with: cargo run --release -p multicalc-demos --example serpent_ik

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::linear_algebra::Vector;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::{Numeric, VectorFn};
use multicalc::spatial::{Quaternion, SE3, SO3};
use multicalc::{LevenbergMarquardt, SolveError};
use multicalc_demos::loop_util::{LatencyRing, Pacer};
use multicalc_demos::{RerunSink, Rgba, VizError, VizSink};
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::time::Instant;

// Palette (§2), sRGB with alpha.
const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // the solved arm, ee gnomon
const TARGET: Rgba = [0xc9, 0x85, 0x00, 0xff]; // target frame
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 140]; // ee trail
const CHROME: Rgba = [0x89, 0x87, 0x81, 0xff]; // reach envelope

const N_JOINTS: usize = 8;
const LINK: f64 = 0.25; // per-link length; reach = N_JOINTS * LINK = 2.0
const K_ORI: f64 = 0.5; // orientation-residual weight
// Sqrt of the posture-regularizer weight, kept small so it holds M >= N and biases weakly toward
// the warm-start without fighting position tracking.
const K_REG: f64 = 3e-4;
const CYCLE: f64 = 20.0; // orientation keyframe cycle (s), 5 s per segment
const PATIENCE: usize = 60; // LM outer-iteration budget

const GEOM_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text + ad-vs-fd cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats
const TRAIL_MAX: usize = 180; // ~3 s of ee positions at 60 Hz
const REACH_SEGS: usize = 128;
const GNOMON: f64 = 0.18; // ee/target frame arrow base length

/// Four target-orientation keyframes (tunable); adjacent geodesic separation stays clear of the
/// `log` θ = π branch. Slerped over `CYCLE`.
fn keyframes() -> [Quaternion<f64>; 4] {
    [
        Quaternion::from_euler_zyx(0.0, 0.0, 0.0),
        Quaternion::from_euler_zyx(0.24, 0.36, 0.48),
        Quaternion::from_euler_zyx(-0.30, -0.42, 0.72),
        Quaternion::from_euler_zyx(0.18, 0.18, -0.54),
    ]
}

/// End-effector forward kinematics, generic over the scalar `S` — the same code yields the pose in
/// `f64` and its derivative in `Dual`. Joint `i` rotates about the body x-axis (even) or y-axis
/// (odd), then a fixed `LINK` translation advances along the body z-axis.
fn fk<S: Numeric>(q: &[S; N_JOINTS]) -> SE3<S> {
    let step = SE3::from_parts(
        SO3::identity(),
        Vector::new([S::ZERO, S::ZERO, S::from_f64(LINK)]),
    );
    let mut t = SE3::<S>::identity();
    for (i, &qi) in q.iter().enumerate() {
        let axis = if i % 2 == 0 {
            Vector::new([S::ONE, S::ZERO, S::ZERO])
        } else {
            Vector::new([S::ZERO, S::ONE, S::ZERO])
        };
        let rot = SO3::exp(axis.scale(qi));
        t = t * SE3::from_parts(rot, Vector::zeros()) * step;
    }
    t
}

/// The cumulative pose at the end of each link, for rendering.
fn chain_poses(q: &[f64; N_JOINTS]) -> [SE3<f64>; N_JOINTS] {
    let step = SE3::from_parts(SO3::identity(), Vector::new([0.0, 0.0, LINK]));
    let mut t = SE3::<f64>::identity();
    let mut poses = [SE3::<f64>::identity(); N_JOINTS];
    for (i, &qi) in q.iter().enumerate() {
        let axis = if i % 2 == 0 {
            Vector::new([1.0, 0.0, 0.0])
        } else {
            Vector::new([0.0, 1.0, 0.0])
        };
        let rot = SO3::exp(axis.scale(qi));
        t = t * SE3::from_parts(rot, Vector::zeros()) * step;
        poses[i] = t;
    }
    poses
}

/// The IK residual system: 3 position + 3 orientation residuals, plus 8 posture regularizers.
struct Serpent {
    target_pos: [f64; 3],
    target_quat: [f64; 4],
    prev: [f64; N_JOINTS],
}

impl VectorFn<8, 14> for Serpent {
    fn eval<S: Numeric>(&self, q: &[S; 8]) -> [S; 14] {
        let ee = fk(q);
        let p = ee.translation();
        let r_ee = ee.rotation();
        let tq = self.target_quat;
        let r_tgt = SO3::from_quaternion(Quaternion::<S>::new(
            S::from_f64(tq[0]),
            S::from_f64(tq[1]),
            S::from_f64(tq[2]),
            S::from_f64(tq[3]),
        ));
        let e = (r_tgt.inverse() * r_ee).log(); // orientation error in so(3)
        let ko = S::from_f64(K_ORI);
        let kr = S::from_f64(K_REG);
        [
            p[0] - S::from_f64(self.target_pos[0]),
            p[1] - S::from_f64(self.target_pos[1]),
            p[2] - S::from_f64(self.target_pos[2]),
            ko * e[0],
            ko * e[1],
            ko * e[2],
            kr * (q[0] - S::from_f64(self.prev[0])),
            kr * (q[1] - S::from_f64(self.prev[1])),
            kr * (q[2] - S::from_f64(self.prev[2])),
            kr * (q[3] - S::from_f64(self.prev[3])),
            kr * (q[4] - S::from_f64(self.prev[4])),
            kr * (q[5] - S::from_f64(self.prev[5])),
            kr * (q[6] - S::from_f64(self.prev[6])),
            kr * (q[7] - S::from_f64(self.prev[7])),
        ]
    }
}

/// Lissajous target position; max base-distance ≈ 1.5 < reach 2.0, leaving length slack so the
/// arm can meet the target orientation without running out of reach.
fn lissajous_pos(t: f64) -> [f64; 3] {
    [
        0.585 * (TAU * 0.11 * t).sin(),
        0.585 * (TAU * 0.17 * t + 0.4).sin(),
        1.0 + 0.26 * (TAU * 0.07 * t).sin(),
    ]
}

/// Target orientation: the keyframes slerped over `CYCLE`, 5 s per segment.
fn target_orientation(t: f64) -> Quaternion<f64> {
    let keys = keyframes();
    let seg = CYCLE / 4.0;
    let phase = (t % CYCLE) / seg; // 0..4
    let i = phase.floor() as usize % 4;
    let frac = phase - phase.floor();
    keys[i].slerp(keys[(i + 1) % 4], frac)
}

/// Position and orientation residual of the converged pose against the current target.
fn residuals(problem: &Serpent) -> (f64, f64) {
    let ee = fk(&problem.prev);
    let p = ee.translation();
    let tp = problem.target_pos;
    let pos = ((p[0] - tp[0]).powi(2) + (p[1] - tp[1]).powi(2) + (p[2] - tp[2]).powi(2)).sqrt();
    let tq = problem.target_quat;
    let r_tgt = SO3::from_quaternion(Quaternion::new(tq[0], tq[1], tq[2], tq[3]));
    let ori = (r_tgt.inverse() * ee.rotation()).log().norm();
    (pos, ori)
}

/// The reach envelope: three great circles of radius `N_JOINTS * LINK` in the coordinate planes.
fn reach_circles() -> Vec<Vec<[f64; 3]>> {
    let r = N_JOINTS as f64 * LINK;
    let circle = |plane: usize| -> Vec<[f64; 3]> {
        (0..=REACH_SEGS)
            .map(|i| {
                let a = TAU * i as f64 / REACH_SEGS as f64;
                match plane {
                    0 => [r * a.cos(), r * a.sin(), 0.0],
                    1 => [0.0, r * a.cos(), r * a.sin()],
                    _ => [r * a.cos(), 0.0, r * a.sin()],
                }
            })
            .collect()
    };
    vec![circle(0), circle(1), circle(2)]
}

/// A frame gnomon: arrows of length ratio 1 : 0.75 : 0.5 along local x, y, z.
fn gnomon() -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let origins = [[0.0; 3]; 3];
    let vectors = [
        [GNOMON, 0.0, 0.0],
        [0.0, GNOMON * 0.75, 0.0],
        [0.0, 0.0, GNOMON * 0.5],
    ];
    (origins, vectors)
}

/// Brightness ramp along the chain, HERO base.
fn link_color(i: usize) -> Rgba {
    let f = 0.65 + 0.05 * i as f64;
    let s = |c: u8| (c as f64 * f).min(255.0) as u8;
    [s(HERO[0]), s(HERO[1]), s(HERO[2]), 0xff]
}

/// Worst-case position residual over one orientation cycle at the live 1 ms cadence, warm-started —
/// a startup reachability check. Samples at the same step size the loop runs, after a short warmup
/// so the cold-start transient is excluded. Returns the max residual seen.
fn reachability_sweep(lm: &LevenbergMarquardt<AutoDiffMulti>) -> f64 {
    let mut problem = Serpent {
        target_pos: lissajous_pos(0.0),
        target_quat: target_orientation(0.0).as_array(),
        prev: [0.1; N_JOINTS],
    };
    let steps = (CYCLE * 1000.0) as i64; // 1 ms spacing
    let mut worst = 0.0_f64;
    for n in 1..=steps {
        let t = n as f64 / 1000.0;
        problem.target_pos = lissajous_pos(t);
        problem.target_quat = target_orientation(t).as_array();
        let x0 = problem.prev;
        if let Ok(rep) = lm.minimize(&problem, &x0) {
            problem.prev = rep.solution;
        }
        if n > 200 {
            worst = worst.max(residuals(&problem).0);
        }
    }
    worst
}

fn main() -> Result<(), VizError> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example serpent_ik"
        );
    }

    let lm = LevenbergMarquardt::<AutoDiffMulti>::default().with_patience(PATIENCE);

    let worst_reach = reachability_sweep(&lm);
    eprintln!("reachability sweep: worst position residual over one cycle = {worst_reach:.2e} m");

    let mut rr = RerunSink::live("multicalc-demos/serpent-ik")?;

    // Statics: stamp at tick 0 so they forward-fill across the run (see rerun-viz-gotchas).
    rr.set_sequence("tick", 0);
    rr.line_strips3d("world/reach", &reach_circles(), &[CHROME], &[0.004])?;
    let (g_o, g_v) = gnomon();
    rr.arrows3d("world/target/gnomon", &g_o, &g_v, &[TARGET])?;
    rr.arrows3d("world/arm/link7/gnomon", &g_o, &g_v, &[HERO])?;
    // One tapered box per link, in the link's local frame (spans back toward the previous joint).
    for i in 0..N_JOINTS {
        let hs = 0.06 - 0.04 * i as f64 / (N_JOINTS - 1) as f64;
        rr.boxes3d(
            &format!("world/arm/link{i}/box"),
            &[[0.0, 0.0, -LINK / 2.0]],
            &[[hs, hs, LINK / 2.0]],
            &[link_color(i)],
        )?;
    }

    let mut problem = Serpent {
        target_pos: lissajous_pos(0.0),
        target_quat: target_orientation(0.0).as_array(),
        prev: [0.1; N_JOINTS], // a gently curled, non-singular start pose
    };

    let mut pacer = Pacer::new();
    let mut solve_ring = LatencyRing::new(1024);
    let mut trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);

    let mut residual_pos = 0.0;
    let mut residual_ori = 0.0;

    let mut n: i64 = 0;
    loop {
        pacer.wait();
        n += 1;
        let t = n as f64 / 1000.0;
        rr.set_sequence("tick", n);

        problem.target_pos = lissajous_pos(t);
        problem.target_quat = target_orientation(t).as_array();
        let x0 = problem.prev; // copy out before the borrow
        let t0 = Instant::now();
        let result = lm.minimize(&problem, &x0);
        let solve_us = t0.elapsed().as_micros() as f64;

        match result {
            Ok(rep) => {
                problem.prev = rep.solution;
                let (rp, ro) = residuals(&problem);
                residual_pos = rp;
                residual_ori = ro;
            }
            Err(SolveError::DidNotConverge { .. }) => {
                // Hold the pose, then nudge deterministically to break a stuck configuration.
                for v in problem.prev.iter_mut() {
                    *v += 1e-3;
                }
            }
            Err(_) => {} // hold the pose
        }

        if n > WARMUP_TICKS {
            solve_ring.push(solve_us);
        }

        // Spatial geometry at ~60 Hz.
        if n % GEOM_EVERY == 0 {
            let poses = chain_poses(&problem.prev);
            for (i, pose) in poses.iter().enumerate() {
                rr.transform3d(
                    &format!("world/arm/link{i}"),
                    pose.translation().into_array(),
                    pose.rotation().quaternion().as_array(),
                )?;
            }
            rr.transform3d("world/target", problem.target_pos, problem.target_quat)?;

            let ee = poses[N_JOINTS - 1].translation().into_array();
            if trail.len() == TRAIL_MAX {
                trail.pop_front();
            }
            trail.push_back(ee);
            rr.line_strips3d(
                "world/trail",
                &[trail.iter().copied().collect()],
                &[ACCENT],
                &[0.006],
            )?;
        }

        // Hud at 1 Hz.
        if n % HUD_EVERY == 0
            && let Some(s) = solve_ring.summary()
        {
            let md = format!(
                "## serpent_ik — multicalc live demo\n\
                 ### full SE(3) IK solve (Levenberg–Marquardt, autodiff Lie Jacobian): median {:.0} µs · p99 {:.0} µs ({:.1} % of the 1 ms tick)\n\
                 ### tracking: position error {:.3} µm, orientation error {:.3} µrad",
                s.median,
                s.p99,
                s.p99 / 10.0,
                residual_pos * 1e6,
                residual_ori * 1e6,
            );
            rr.text("hud/stats", &md)?;
        }
    }
}
