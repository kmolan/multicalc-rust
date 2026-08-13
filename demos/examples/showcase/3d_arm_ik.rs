//! 1 kHz 3D arm SE(3) IK (kinematics showcase).
//!
//! An 8-link arm chases a moving 3D target in position and orientation. Every millisecond a full
//! damped-least-squares solve runs against the arm's analytic geometric Jacobian, holding every
//! hinge inside its travel and spending the freedom the task leaves over on keeping a comfortable
//! posture. The panel shows the solve cost against the 1 ms budget.
//!
//! Streams live to a Rerun viewer; see demos/README.md for the WSL setup.
//! Run with: cargo run --release -p multicalc-demos --example 3d_arm_ik

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::kinematics::{
    InverseKinematics, InverseKinematicsTermination, Joint, JointParent, KinematicTree,
    SecondaryObjective,
};
use multicalc::linear_algebra::Vector;
use multicalc::scalar::Numeric;
use multicalc::spatial::{Quaternion, SE3, SO3};
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
const N_FRAMES: usize = N_JOINTS + 1; // the joint frames plus the welded tool frame
const LINK: f64 = 0.25; // per-link length; reach = N_JOINTS * LINK = 2.0
const CYCLE: f64 = 20.0; // orientation keyframe cycle (s), 5 s per segment
const MAXIMUM_ITERATIONS: usize = 40; // IK budget per tick
const SECONDARY_GAIN: f64 = 0.2; // how hard the comfortable posture pulls
const JOINT_LIMIT: f64 = 2.6; // travel per hinge, radians
/// The posture the arm drifts back toward with whatever freedom the task leaves it.
const RESTING_POSTURE: f64 = 0.1;

const GEOM_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats
const TRAIL_MAX: usize = 180; // ~3 s of ee positions at 60 Hz
const REACH_SEGS: usize = 128;
const GNOMON: f64 = 0.18; // ee/target frame arrow base length

/// Four target-orientation keyframes (tunable); adjacent geodesic separation stays clear of the
/// `log` θ = π branch. Slerped over `CYCLE`.
#[must_use]
fn keyframes() -> [Quaternion<f64>; 4] {
    [
        Quaternion::from_euler_zyx(0.0, 0.0, 0.0),
        Quaternion::from_euler_zyx(0.24, 0.36, 0.48),
        Quaternion::from_euler_zyx(-0.30, -0.42, 0.72),
        Quaternion::from_euler_zyx(0.18, 0.18, -0.54),
    ]
}

/// The arm as a model, generic over the scalar `S`: joint `i` rotates about the body x-axis (even)
/// or y-axis (odd), and each joint's origin advances `LINK` along its parent's z-axis. Joint
/// `N_JOINTS` is a weld carrying the tool frame one link past the last hinge.
///
/// Each joint frame therefore sits at the *start* of the link it carries, and that link runs along
/// the frame's +z. The renderer relies on this.
#[must_use]
fn arm<S: Numeric>() -> KinematicTree<N_FRAMES, S> {
    let step = SE3::from_parts(
        SO3::identity(),
        Vector::new([S::ZERO, S::ZERO, S::from_f64(LINK)]),
    );
    let mut tree = KinematicTree::<N_FRAMES, S>::new();
    for i in 0..N_FRAMES {
        // The first joint sits at the world origin; every later frame starts one link further on.
        let origin = if i == 0 { SE3::identity() } else { step };
        let parent = if i == 0 {
            JointParent::World
        } else {
            JointParent::Joint(i - 1)
        };
        let joint = if i == N_JOINTS {
            Joint::fixed(origin)
        } else {
            let axis = if i % 2 == 0 {
                Vector::new([S::ONE, S::ZERO, S::ZERO])
            } else {
                Vector::new([S::ZERO, S::ONE, S::ZERO])
            };
            Joint::revolute(axis, origin)
                .with_limits(S::from_f64(-JOINT_LIMIT), S::from_f64(JOINT_LIMIT))
        };
        tree.push(joint, parent)
            .unwrap_or_else(|_| unreachable!("the arm is a valid tree"));
    }
    tree
}

/// The joint readings, with a zero in the welded tool frame's slot.
fn readings<S: Numeric>(q: &[S; N_JOINTS]) -> Vector<N_FRAMES, S> {
    Vector::from_fn(|i| if i < N_JOINTS { q[i] } else { S::ZERO })
}

/// Every frame of the arm from one solve: the eight joint frames, then the tool.
#[must_use]
fn link_poses(
    tree: &KinematicTree<N_FRAMES, f64>,
    q: &Vector<N_FRAMES, f64>,
) -> [SE3<f64>; N_FRAMES] {
    let state = tree
        .forward_kinematics(q)
        .unwrap_or_else(|_| unreachable!("finite readings"));
    core::array::from_fn(|i| {
        state
            .pose(i)
            .unwrap_or_else(|| unreachable!("every frame was settled"))
    })
}

/// Startup check on the frame convention: at rest the tool sits one full reach up z, and folding
/// the first hinge a quarter turn about x swings it onto -y.
fn verify_frame_convention(tree: &KinematicTree<N_FRAMES, f64>) {
    let reach = N_JOINTS as f64 * LINK;
    let tool = |q: &[f64; N_JOINTS]| link_poses(tree, &readings(q))[N_JOINTS];

    let at_rest = tool(&[0.0; N_JOINTS]).translation().into_array();
    assert!(
        (at_rest[0]).abs() < 1e-12
            && at_rest[1].abs() < 1e-12
            && (at_rest[2] - reach).abs() < 1e-12,
        "tool at rest: {at_rest:?}, expected [0, 0, {reach}]"
    );

    let mut folded = [0.0; N_JOINTS];
    folded[0] = std::f64::consts::FRAC_PI_2;
    let quarter_turn = tool(&folded).translation().into_array();
    assert!(
        quarter_turn[0].abs() < 1e-12
            && (quarter_turn[1] + reach).abs() < 1e-12
            && quarter_turn[2].abs() < 1e-12,
        "tool at q0 = pi/2: {quarter_turn:?}, expected [0, {}, 0]",
        -reach
    );
}

/// Lissajous target position; max base-distance ≈ 1.5 < reach 2.0, leaving length slack so the
/// arm can meet the target orientation without running out of reach.
#[must_use]
fn lissajous_pos(t: f64) -> [f64; 3] {
    [
        0.585 * (TAU * 0.11 * t).sin(),
        0.585 * (TAU * 0.17 * t + 0.4).sin(),
        1.0 + 0.26 * (TAU * 0.07 * t).sin(),
    ]
}

/// Target orientation: the keyframes slerped over `CYCLE`, 5 s per segment.
#[must_use]
fn target_orientation(t: f64) -> Quaternion<f64> {
    let keys = keyframes();
    let seg = CYCLE / 4.0;
    let phase = (t % CYCLE) / seg; // 0..4
    let i = phase.floor() as usize % 4;
    let frac = phase - phase.floor();
    keys[i].slerp(keys[(i + 1) % 4], frac)
}

/// The reach envelope: three great circles of radius `N_JOINTS * LINK` in the coordinate planes.
#[must_use]
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
#[must_use]
fn gnomon() -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let origins = [[0.0; 3]; 3];
    let vectors = [
        [GNOMON, 0.0, 0.0],
        [0.0, GNOMON * 0.75, 0.0],
        [0.0, 0.0, GNOMON * 0.5],
    ];
    (origins, vectors)
}

/// Brightness ramp along the arm, HERO base.
#[must_use]
fn link_color(i: usize) -> Rgba {
    let f = 0.65 + 0.05 * i as f64;
    let s = |c: u8| (c as f64 * f).min(255.0) as u8;
    [s(HERO[0]), s(HERO[1]), s(HERO[2]), 0xff]
}

/// The target pose at time `t`: the Lissajous position carrying the slerped keyframe orientation.
#[must_use]
fn target_pose(t: f64) -> SE3<f64> {
    let quaternion = target_orientation(t).as_array();
    SE3::from_parts(
        SO3::from_quaternion(Quaternion::new(
            quaternion[0],
            quaternion[1],
            quaternion[2],
            quaternion[3],
        )),
        Vector::new(lissajous_pos(t)),
    )
}

/// Worst-case position residual over one orientation cycle at the live 1 ms cadence, warm-started —
/// a startup reachability check. Samples at the same step size the loop runs, after a short warmup
/// so the cold-start transient is excluded. Returns the max residual seen.
#[must_use]
fn reachability_sweep(
    solver: &InverseKinematics<N_FRAMES, f64>,
    tree: &KinematicTree<N_FRAMES, f64>,
) -> f64 {
    let mut joint_readings = readings(&[RESTING_POSTURE; N_JOINTS]);
    let steps = (CYCLE * 1000.0) as i64; // 1 ms spacing
    let mut worst = 0.0_f64;
    for n in 1..=steps {
        let t = n as f64 / 1000.0;
        let Ok(report) = solver.solve(tree, N_JOINTS, target_pose(t), &joint_readings) else {
            continue;
        };
        joint_readings = report.joint_positions;
        if n > 200 {
            worst = worst.max(report.position_error);
        }
    }
    worst
}

fn main() -> Result<(), VizError> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example 3d_arm_ik"
        );
    }

    // Built once: the model is fixed, and rebuilding it inside the loop would land in the timing.
    let tree = arm::<f64>();
    verify_frame_convention(&tree);

    let solver = InverseKinematics::<N_FRAMES, f64>::new()
        .with_maximum_iterations(MAXIMUM_ITERATIONS)
        .with_position_tolerance(1e-6)
        .with_orientation_tolerance(1e-6)
        .with_secondary_objective(SecondaryObjective::PreferredPosture(readings(
            &[RESTING_POSTURE; N_JOINTS],
        )))
        .with_secondary_gain(SECONDARY_GAIN);

    let worst_reach = reachability_sweep(&solver, &tree);
    eprintln!("reachability sweep: worst position residual over one cycle = {worst_reach:.2e} m");

    let mut rr = RerunSink::live("multicalc-demos/3d-arm-ik")?;

    // Statics: stamp at tick 0 so they forward-fill across the run (see rerun-viz-gotchas).
    rr.set_sequence("tick", 0);
    rr.line_strips3d("world/reach", &reach_circles(), &[CHROME], &[0.004])?;
    let (g_o, g_v) = gnomon();
    rr.arrows3d("world/target/gnomon", &g_o, &g_v, &[TARGET])?;
    rr.arrows3d("world/arm/tool/gnomon", &g_o, &g_v, &[HERO])?;
    // One tapered box per link, in its joint's frame (spanning forward toward the next joint).
    for i in 0..N_JOINTS {
        let hs = 0.06 - 0.04 * i as f64 / (N_JOINTS - 1) as f64;
        rr.boxes3d(
            &format!("world/arm/link{i}/box"),
            &[[0.0, 0.0, LINK / 2.0]],
            &[[hs, hs, LINK / 2.0]],
            &[link_color(i)],
        )?;
    }

    // A gently curled, non-singular start pose.
    let mut joint_readings = readings(&[RESTING_POSTURE; N_JOINTS]);

    let mut pacer = Pacer::new();
    let mut solve_ring = LatencyRing::new(1024);
    let mut trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);

    let mut residual_pos = 0.0;
    let mut residual_ori = 0.0;
    let mut stalled_ticks: u64 = 0;

    let mut n: i64 = 0;
    loop {
        let _ = pacer.wait();
        n += 1;
        let t = n as f64 / 1000.0;
        rr.set_sequence("tick", n);

        let target = target_pose(t);
        let t0 = Instant::now();
        let result = solver.solve(&tree, N_JOINTS, target, &joint_readings);
        let solve_us = t0.elapsed().as_micros() as f64;

        // A solve that ran out of budget or stalled still hands back the nearest pose it managed,
        // which is what a control loop wants; only a malformed request is an error, and there the
        // arm holds where it is.
        if let Ok(report) = result {
            joint_readings = report.joint_positions;
            residual_pos = report.position_error;
            residual_ori = report.orientation_error;
            if report.termination == InverseKinematicsTermination::Stalled {
                stalled_ticks += 1;
            }
        }

        if n > WARMUP_TICKS {
            solve_ring.push(solve_us);
        }

        // Spatial geometry at ~60 Hz.
        if n % GEOM_EVERY == 0 {
            let poses = link_poses(&tree, &joint_readings);
            for (i, pose) in poses.iter().take(N_JOINTS).enumerate() {
                rr.transform3d(
                    &format!("world/arm/link{i}"),
                    pose.translation().into_array(),
                    pose.rotation().quaternion().as_array(),
                )?;
            }
            let tool = poses[N_JOINTS];
            rr.transform3d(
                "world/arm/tool",
                tool.translation().into_array(),
                tool.rotation().quaternion().as_array(),
            )?;
            rr.transform3d(
                "world/target",
                target.translation().into_array(),
                target.rotation().quaternion().as_array(),
            )?;

            let ee = tool.translation().into_array();
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
                "## 3d_arm_ik — multicalc live demo\n\
                 ### full SE(3) IK solve (damped least squares, analytic Jacobian): median {:.0} µs · p99 {:.0} µs ({:.1} % of the 1 ms tick)\n\
                 ### tracking: position error {:.3} µm, orientation error {:.3} µrad · {} stalled ticks",
                s.median,
                s.p99,
                s.p99 / 10.0,
                residual_pos * 1e6,
                residual_ori * 1e6,
                stalled_ticks,
            );
            rr.text("hud/stats", &md)?;
        }
    }
}
