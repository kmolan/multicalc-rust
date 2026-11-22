//! 1 kHz Franka Panda SE(3) IK, driven through a joint PD (kinematics + dynamics showcase).
//!
//! Panda from its MJCF file. Damped-least-squares IK on the analytic geometric Jacobian tracks a
//! moving SE(3) target inside the model's joint limits, redundancy spent on a preferred posture.
//! The target sweeps the reach shell — ±140° azimuth, 0.38..0.66 m reach, the tool yawing with the
//! sweep and pitching and rolling on top; `reachability_sweep` sets those amplitudes at startup.
//!
//! The solve is a reference, not the state: `τ = kp⊙(q_d − q) + kd⊙(q̇_d − q̇) + G(q)` with
//! `q̇_d = 0` drives the articulated-body dynamics. The headline is the gap between commanded and
//! reached tool position — tracking lag, proportional to tool speed since the derivative term is
//! pure damping. Panel reads the whole tick against the 1 ms budget.
//!
//! Streams to a Rerun viewer; see demos/README.md for WSL setup.
//! Run: cargo run --release -p multicalc-demos --example 3d_arm_ik

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::control::{JointPdController, JointReference};
use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{
    InverseKinematics, InverseKinematicsTermination, JointKind, KinematicTree, SecondaryObjective,
};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3};
use multicalc_demos::loop_util::{LatencyRing, Pacer};
use multicalc_demos::{RerunSink, Rgba, VizSink};
use std::collections::{HashMap, VecDeque};
use std::f64::consts::TAU;
use std::path::Path;
use std::time::Instant;

// Palette  sRGB with alpha.
const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // arm, ee gnomon
const TARGET: Rgba = [0xc9, 0x85, 0x00, 0xff]; // target frame
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 140]; // ee trail
const CHROME: Rgba = [0x89, 0x87, 0x81, 0xff]; // target path
const ERROR: Rgba = [0xe6, 0x67, 0x67, 0xff]; // tracking lag plot
const TARGET_FAINT: Rgba = [0xc9, 0x85, 0x00, 110]; // commanded tool trail

const MODEL_FILE: &str = "../third_party/menagerie/franka_emika_panda/panda.xml";
/// Mesh directory (panda.xml `<compiler meshdir="assets"/>`).
const MESH_DIR: &str = "../third_party/menagerie/franka_emika_panda/assets";
/// Tip frame: hand, one weld past the last hinge.
const TIP: &str = "hand";
const N_FRAMES: usize = 9; // link0, seven hinges, hand
const N_JOINTS: usize = 7; // movable joints
const TOOL_INDEX: usize = 8;
/// Resting posture, from panda.xml's `home` keyframe (not `PI/2`/`PI/4`-rounded).
#[allow(clippy::approx_constant)]
const HOME_POSTURE: [f64; N_JOINTS] = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853];

/// Target path period (s). Every harmonic below is an integer multiple of `1/CYCLE`, so the whole
/// pose trajectory — position and orientation — closes exactly and `target_path` can draw it.
const CYCLE: f64 = 60.0;
const MAXIMUM_ITERATIONS: usize = 40; // IK budget per tick
const SECONDARY_GAIN: f64 = 0.2; // secondary-objective weight

// Target position, in shoulder-centred spherical coordinates: azimuth, elevation and reach.
// Spherical, not cylindrical: the workspace is a shell, so the far-high corner of a box in
// (radius, height) sits off the reach sphere. Ranges come from the startup sweep, not the bare
// reach — holding a tool orientation costs workspace.
/// Shoulder height: panda.xml's `link1` body origin, checked against the model in `verify_model`.
const SHOULDER_HEIGHT: f64 = 0.333;
const AZIMUTH_SWEEP: f64 = 2.45; // ±140°, inside joint 1's ±166° travel
const ELEVATION_MID: f64 = 0.18;
const ELEVATION_SWEEP: f64 = 0.44; // −15° .. +36° off the shoulder plane
const REACH_MID: f64 = 0.52;
const REACH_SWEEP: f64 = 0.14; // 0.38 .. 0.66 m from the shoulder

// Target orientation: yaw follows the sweep, since the wrist runs out of travel before joint 1
// does; pitch and roll ride on top, holding the tool off the radial direction.
const YAW_LEAD: f64 = 0.45; // yaw offset from the radial direction (rad)
const PITCH_SWEEP: f64 = 0.50;
const ROLL_SWEEP: f64 = 0.38;

const GRAVITY: f64 = -9.81;
const TICK: f64 = 1e-3; // loop period (s), matching `Pacer`
/// Closed-loop bandwidth per joint. Bounded by the loop rate; 50 rad/s against a 1 kHz tick leaves
/// an order of magnitude of headroom.
const BANDWIDTH: f64 = 50.0;
const DAMPING_RATIO: f64 = 1.0; // critically damped: no overshoot

const GEOM_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats
const TRAIL_MAX: usize = 180; // ~3 s of ee positions at 60 Hz
const TARGET_PATH_SEGS: usize = 1024; // the path is a full workspace sweep
const GNOMON: f64 = 0.18; // ee/target frame arrow base length

/// Franka Panda with mass: base-to-hand chain under earth gravity.
///
/// `ArticulatedBody::tree` returns the chain the IK runs on, so solver and plant share one model.
fn arm() -> Result<ArticulatedBody<N_FRAMES, N_FRAMES, f64>, Box<dyn std::error::Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_FILE);
    let model = multicalc_robot_model::mjcf::load_path(&path)?;
    Ok(model.articulated_body_to::<N_FRAMES, N_FRAMES>(TIP, Vector::new([0.0, 0.0, GRAVITY]))?)
}

/// Per-joint PD gains from the arm's own inertia at the home posture.
///
/// Gravity cancelled, the closed loop linearizes per joint to
/// `H_ii·ë + (kd + damping)·ė + kp·e = 0`, so `kp = H_ii·ω²` and `kd = 2·ζ·ω·H_ii − damping`.
/// `H_ii` carries the armature, CRBA putting it on the diagonal.
///
/// Taken at one posture: `H(q)` varies across the workspace, so the realized bandwidth drifts as
/// the arm moves — what a fixed-gain joint servo does.
fn gains(
    body: &ArticulatedBody<N_FRAMES, N_FRAMES, f64>,
) -> (
    Vector<N_FRAMES, f64>,
    Vector<N_FRAMES, f64>,
    [f64; N_JOINTS],
) {
    let inertia = body
        .joint_space_inertia_at(&readings(&HOME_POSTURE))
        .unwrap_or_else(|_| unreachable!("the home posture is finite"));

    let mut diagonal = [0.0; N_JOINTS];
    let mut position_gains = Vector::<N_FRAMES, f64>::zeros();
    let mut velocity_gains = Vector::<N_FRAMES, f64>::zeros();
    for slot in 1..=N_JOINTS {
        let effective_inertia = inertia[(slot, slot)];
        assert!(
            effective_inertia > 0.0,
            "joint-space inertia at slot {slot} is not positive"
        );
        let damping = body
            .tree()
            .joint(slot)
            .unwrap_or_else(|| unreachable!())
            .damping();

        diagonal[slot - 1] = effective_inertia;
        position_gains[slot] = effective_inertia * BANDWIDTH * BANDWIDTH;
        // The model's damping can already exceed what the ratio asks; a negative gain would
        // drive the joint rather than damp it, and the controller rejects it.
        velocity_gains[slot] =
            (2.0 * DAMPING_RATIO * BANDWIDTH * effective_inertia - damping).max(0.0);
    }
    (position_gains, velocity_gains, diagonal)
}

/// Joint readings; zero at the welded base/hand slots.
fn readings(joints: &[f64; N_JOINTS]) -> Vector<N_FRAMES, f64> {
    Vector::from_fn(|i| {
        if (1..=N_JOINTS).contains(&i) {
            joints[i - 1]
        } else {
            0.0
        }
    })
}

/// All 9 frame poses for one configuration.
#[must_use]
fn link_poses(
    tree: &KinematicTree<N_FRAMES, N_FRAMES, f64>,
    joints: &Vector<N_FRAMES, f64>,
) -> [SE3<f64>; N_FRAMES] {
    let state = tree
        .forward_kinematics(joints)
        .unwrap_or_else(|_| unreachable!("finite readings"));
    core::array::from_fn(|i| {
        state
            .pose(i)
            .unwrap_or_else(|| unreachable!("every frame was settled"))
    })
}

/// Startup check: model structure and default-class joint settings.
///
/// Loose bounds by design — catches a mis-parsed chain without duplicating the fixture test.
fn verify_model(tree: &KinematicTree<N_FRAMES, N_FRAMES, f64>) {
    assert_eq!(tree.len(), N_FRAMES);
    assert_eq!(
        tree.joint(0).unwrap_or_else(|| unreachable!()).kind(),
        JointKind::Fixed
    );
    assert_eq!(
        tree.joint(TOOL_INDEX)
            .unwrap_or_else(|| unreachable!())
            .kind(),
        JointKind::Fixed
    );
    for slot in 1..=N_JOINTS {
        let joint = tree.joint(slot).unwrap_or_else(|| unreachable!());
        assert_eq!(joint.kind(), JointKind::Revolute);
        assert!(
            (joint.armature() - 0.1).abs() < 1e-12,
            "armature at slot {slot}"
        );
        assert!(
            (joint.damping() - 1.0).abs() < 1e-12,
            "damping at slot {slot}"
        );
        assert!(joint.limits().is_some(), "no travel limits at slot {slot}");
    }
    // The path is laid out on a shell around the shoulder, so the constant has to be the model's.
    let shoulder = link_poses(tree, &Vector::zeros())[1].translation();
    assert!(
        (shoulder[2] - SHOULDER_HEIGHT).abs() < 1e-9,
        "shoulder height: model says {}",
        shoulder[2]
    );

    // q = 0 rest pose: straight up.
    let hand = link_poses(tree, &Vector::zeros())[TOOL_INDEX];
    let [x, y, z] = *hand.translation().as_array();
    assert!(z > 0.8 && x.hypot(y) < 0.3, "hand at rest: {:?}", [x, y, z]);
}

/// Azimuth about the base axis at `time`: one sweep out and back per `CYCLE`.
#[must_use]
fn target_azimuth(time: f64) -> f64 {
    AZIMUTH_SWEEP * (TAU * time / CYCLE).sin()
}

/// Target position: a Lissajous on the reach shell — azimuth once per `CYCLE`, elevation twice,
/// reach three times, the offset phases keeping successive passes apart.
#[must_use]
fn target_position(time: f64) -> [f64; 3] {
    let azimuth = target_azimuth(time);
    let elevation = ELEVATION_MID + ELEVATION_SWEEP * (2.0 * TAU * time / CYCLE + 0.7).sin();
    let reach = REACH_MID + REACH_SWEEP * (3.0 * TAU * time / CYCLE + 1.3).sin();
    [
        reach * elevation.cos() * azimuth.cos(),
        reach * elevation.cos() * azimuth.sin(),
        SHOULDER_HEIGHT + reach * elevation.sin(),
    ]
}

/// Target orientation: home orientation yawed to follow the sweep, then pitched and rolled in the
/// tool frame.
///
/// Yaw tracks the azimuth because the wrist runs out of travel long before joint 1 does — the
/// difference between a 120° sweep and a 250° one. `YAW_LEAD` and the tool-frame turns hold the
/// tool off the radial direction.
#[must_use]
fn target_orientation(home: SO3<f64>, time: f64) -> SO3<f64> {
    let yaw = target_azimuth(time) + YAW_LEAD * (2.0 * TAU * time / CYCLE + 0.9).sin();
    let pitch = PITCH_SWEEP * (2.0 * TAU * time / CYCLE + 0.3).sin();
    let roll = ROLL_SWEEP * (3.0 * TAU * time / CYCLE + 1.1).sin();
    SO3::exp(Vector::new([0.0, 0.0, yaw])) * home * SO3::exp(Vector::new([roll, pitch, 0.0]))
}

/// Target pose at `time`.
#[must_use]
fn target_pose(home: SE3<f64>, time: f64) -> SE3<f64> {
    SE3::from_parts(
        target_orientation(home.rotation(), time),
        Vector::new(target_position(time)),
    )
}

/// Target path over one cycle, closed.
#[must_use]
fn target_path(home: SE3<f64>) -> Vec<[f64; 3]> {
    (0..=TARGET_PATH_SEGS)
        .map(|i| {
            let time = CYCLE * i as f64 / TARGET_PATH_SEGS as f64;
            target_pose(home, time).translation().into_array()
        })
        .collect()
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

// panda.xml `<asset>` materials.
const MAT_WHITE: Rgba = [255, 255, 255, 255];
const MAT_OFF_WHITE: Rgba = [230, 235, 237, 255];
const MAT_BLACK: Rgba = [64, 64, 64, 255];
const MAT_GREEN: Rgba = [0, 255, 0, 255];
const MAT_LIGHT_BLUE: Rgba = [10, 138, 199, 255];

/// One body's visual sub-meshes: filename (relative to `MESH_DIR`, `.obj` implied) and material
/// color, from panda.xml's `<geom class="visual">` entries. No geom has its own `pos`/`quat`.
#[must_use]
fn body_meshes(slot: usize) -> &'static [(&'static str, Rgba)] {
    match slot {
        0 => &[
            ("link0_0", MAT_OFF_WHITE),
            ("link0_1", MAT_BLACK),
            ("link0_2", MAT_OFF_WHITE),
            ("link0_3", MAT_BLACK),
            ("link0_4", MAT_OFF_WHITE),
            ("link0_5", MAT_BLACK),
            ("link0_7", MAT_WHITE),
            ("link0_8", MAT_WHITE),
            ("link0_9", MAT_BLACK),
            ("link0_10", MAT_OFF_WHITE),
            ("link0_11", MAT_WHITE),
        ],
        1 => &[("link1", MAT_WHITE)],
        2 => &[("link2", MAT_WHITE)],
        3 => &[
            ("link3_0", MAT_WHITE),
            ("link3_1", MAT_WHITE),
            ("link3_2", MAT_WHITE),
            ("link3_3", MAT_BLACK),
        ],
        4 => &[
            ("link4_0", MAT_WHITE),
            ("link4_1", MAT_WHITE),
            ("link4_2", MAT_BLACK),
            ("link4_3", MAT_WHITE),
        ],
        5 => &[
            ("link5_0", MAT_BLACK),
            ("link5_1", MAT_WHITE),
            ("link5_2", MAT_WHITE),
        ],
        6 => &[
            ("link6_0", MAT_OFF_WHITE),
            ("link6_1", MAT_WHITE),
            ("link6_2", MAT_BLACK),
            ("link6_3", MAT_WHITE),
            ("link6_4", MAT_WHITE),
            ("link6_5", MAT_WHITE),
            ("link6_6", MAT_WHITE),
            ("link6_7", MAT_LIGHT_BLUE),
            ("link6_8", MAT_LIGHT_BLUE),
            ("link6_9", MAT_BLACK),
            ("link6_10", MAT_BLACK),
            ("link6_11", MAT_WHITE),
            ("link6_12", MAT_GREEN),
            ("link6_13", MAT_WHITE),
            ("link6_14", MAT_BLACK),
            ("link6_15", MAT_BLACK),
            ("link6_16", MAT_WHITE),
        ],
        7 => &[
            ("link7_0", MAT_WHITE),
            ("link7_1", MAT_BLACK),
            ("link7_2", MAT_BLACK),
            ("link7_3", MAT_BLACK),
            ("link7_4", MAT_BLACK),
            ("link7_5", MAT_BLACK),
            ("link7_6", MAT_BLACK),
            ("link7_7", MAT_WHITE),
        ],
        8 => &[
            ("hand_0", MAT_OFF_WHITE),
            ("hand_1", MAT_BLACK),
            ("hand_2", MAT_BLACK),
            ("hand_3", MAT_WHITE),
            ("hand_4", MAT_OFF_WHITE),
        ],
        _ => unreachable!("nine slots"),
    }
}

/// Parses one `.obj` into an indexed triangle mesh.
///
/// Faces are `v//vn` (vertex + normal index, no texcoord); a vertex is a (position, normal) pair
/// since the two indices can differ, so positions may repeat across output vertices. Faces are
/// always triangles. `mtllib`/`usemtl` are skipped (the referenced `.mtl` isn't shipped) — color
/// comes from `body_meshes` instead.
fn read_obj(path: &Path) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("reading mesh file {path:?}: {err}"));

    let mut positions = Vec::new();
    let mut raw_normals = Vec::new();
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut triangles = Vec::new();
    let mut seen: HashMap<(u32, u32), u32> = HashMap::new();

    for line in text.lines() {
        let mut fields = line.split_ascii_whitespace();
        match fields.next() {
            Some("v") => {
                let xyz: Vec<f64> = fields.map(|f| f.parse().unwrap()).collect();
                positions.push([xyz[0], xyz[1], xyz[2]]);
            }
            Some("vn") => {
                let xyz: Vec<f64> = fields.map(|f| f.parse().unwrap()).collect();
                raw_normals.push([xyz[0], xyz[1], xyz[2]]);
            }
            Some("f") => {
                let mut triangle = [0u32; 3];
                for (corner, field) in fields.enumerate() {
                    let mut parts = field.split('/');
                    let vertex_index: u32 = parts.next().unwrap().parse().unwrap();
                    let _texture_coordinate = parts.next(); // always empty
                    let normal_index: u32 = parts.next().unwrap().parse().unwrap();
                    let key = (vertex_index, normal_index);
                    let index = *seen.entry(key).or_insert_with(|| {
                        vertices.push(positions[(vertex_index - 1) as usize]);
                        normals.push(raw_normals[(normal_index - 1) as usize]);
                        (vertices.len() - 1) as u32
                    });
                    triangle[corner] = index;
                }
                triangles.push(triangle);
            }
            _ => {}
        }
    }
    (vertices, normals, triangles)
}

/// Merges one body's visual sub-meshes into one vertex/normal/triangle/color set.
#[must_use]
fn load_body_mesh(
    mesh_dir: &Path,
    slot: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[u32; 3]>, Vec<Rgba>) {
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut triangles = Vec::new();
    let mut colors = Vec::new();

    for &(file, color) in body_meshes(slot) {
        let (part_vertices, part_normals, part_triangles) =
            read_obj(&mesh_dir.join(format!("{file}.obj")));
        let offset = vertices.len() as u32;
        colors.extend(std::iter::repeat_n(color, part_vertices.len()));
        vertices.extend(part_vertices);
        normals.extend(part_normals);
        triangles.extend(
            part_triangles
                .into_iter()
                .map(|[i, j, k]| [i + offset, j + offset, k + offset]),
        );
    }
    (vertices, normals, triangles, colors)
}

/// What one cycle of the target path costs the solver, and how much of the arm it uses.
struct Sweep {
    /// Worst position residual over the cycle, metres.
    position_residual: f64,
    /// Worst orientation residual over the cycle, radians.
    orientation_residual: f64,
    /// Ticks the solver stalled on.
    stalled: u64,
    /// Per-joint travel used, as a fraction of what the model allows.
    travel_used: [f64; N_JOINTS],
    /// Straight-line tool speed, peak, m/s.
    peak_speed: f64,
}

/// Startup reachability check: walks the target path at the loop cadence, post-warmup.
///
/// The sweep amplitudes are set against what this reports: an amplitude the solver cannot hold is
/// a torque spike once the arm is driven rather than posed.
#[must_use]
fn reachability_sweep(
    solver: &InverseKinematics<N_FRAMES, f64>,
    tree: &KinematicTree<N_FRAMES, N_FRAMES, f64>,
    home_pose: SE3<f64>,
) -> Sweep {
    let mut joint_readings = readings(&HOME_POSTURE);
    let steps = (CYCLE * 1000.0) as i64; // 1 ms spacing
    let mut sweep = Sweep {
        position_residual: 0.0,
        orientation_residual: 0.0,
        stalled: 0,
        travel_used: [0.0; N_JOINTS],
        peak_speed: 0.0,
    };
    let mut lowest = [f64::INFINITY; N_JOINTS];
    let mut highest = [f64::NEG_INFINITY; N_JOINTS];
    let mut previous = Vector::new(target_position(0.0));

    for n in 1..=steps {
        let time = n as f64 / 1000.0;
        let position = Vector::new(target_position(time));
        let Ok(report) = solver.solve(
            tree,
            TOOL_INDEX,
            target_pose(home_pose, time),
            &joint_readings,
        ) else {
            continue;
        };
        joint_readings = report.joint_positions;
        if n > 200 {
            sweep.position_residual = sweep.position_residual.max(report.position_error);
            sweep.orientation_residual = sweep.orientation_residual.max(report.orientation_error);
            sweep.peak_speed = sweep.peak_speed.max((position - previous).norm() / TICK);
            if report.termination == InverseKinematicsTermination::Stalled {
                sweep.stalled += 1;
            }
            for slot in 1..=N_JOINTS {
                lowest[slot - 1] = lowest[slot - 1].min(joint_readings[slot]);
                highest[slot - 1] = highest[slot - 1].max(joint_readings[slot]);
            }
        }
        previous = position;
    }

    for slot in 1..=N_JOINTS {
        let (lower, upper) = tree
            .joint(slot)
            .unwrap_or_else(|| unreachable!())
            .limits()
            .unwrap_or_else(|| unreachable!("verify_model checked every joint has travel"));
        sweep.travel_used[slot - 1] = (highest[slot - 1] - lowest[slot - 1]) / (upper - lower);
    }
    sweep
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example 3d_arm_ik"
        );
    }

    // Built once: rebuilding per-tick would pollute the timing.
    let body = arm()?;
    let tree = body.tree();
    verify_model(tree);

    let (position_gains, velocity_gains, effective_inertia) = gains(&body);
    let controller =
        JointPdController::new(position_gains, velocity_gains)?.with_gravity_compensation(true);

    let solver = InverseKinematics::<N_FRAMES, f64>::new()
        .with_maximum_iterations(MAXIMUM_ITERATIONS)
        .with_position_tolerance(1e-6)
        .with_orientation_tolerance(1e-6)
        .with_secondary_objective(SecondaryObjective::PreferredPosture(readings(
            &HOME_POSTURE,
        )))
        .with_secondary_gain(SECONDARY_GAIN);

    let home_pose = link_poses(tree, &readings(&HOME_POSTURE))[TOOL_INDEX];

    let sweep = reachability_sweep(&solver, tree, home_pose);
    eprintln!(
        "target path: azimuth ±{:.0}°, elevation {:+.0}..{:+.0}°, reach {:.2}..{:.2} m, \
         period {CYCLE:.0} s, peak tool speed {:.2} m/s",
        AZIMUTH_SWEEP.to_degrees(),
        (ELEVATION_MID - ELEVATION_SWEEP).to_degrees(),
        (ELEVATION_MID + ELEVATION_SWEEP).to_degrees(),
        REACH_MID - REACH_SWEEP,
        REACH_MID + REACH_SWEEP,
        sweep.peak_speed,
    );
    eprintln!(
        "reachability sweep over one cycle: worst residual {:.2e} m / {:.2e} rad, {} stalled ticks",
        sweep.position_residual, sweep.orientation_residual, sweep.stalled,
    );
    eprint!("  joint travel used:");
    for slot in 1..=N_JOINTS {
        eprint!(" j{slot} {:.0}%", sweep.travel_used[slot - 1] * 100.0);
    }
    eprintln!();
    assert!(
        sweep.position_residual < 1e-4 && sweep.orientation_residual < 1e-4,
        "the target path leaves the workspace the solver can hold: reduce the sweep amplitudes"
    );

    eprintln!(
        "joint PD from the model's own inertia at the home posture \
         (bandwidth {BANDWIDTH:.0} rad/s, damping ratio {DAMPING_RATIO:.1}):"
    );
    for slot in 1..=N_JOINTS {
        eprintln!(
            "  joint {slot}: H = {:7.4} kg·m²  ->  kp = {:9.2} N·m/rad, kd = {:8.2} N·m·s/rad",
            effective_inertia[slot - 1],
            position_gains[slot],
            velocity_gains[slot],
        );
    }

    let mut sink = RerunSink::live("multicalc-demos/3d-arm-ik")?;

    // Statics: tick 0, forward-fill (see rerun-viz-gotchas).
    sink.set_sequence("tick", 0);
    sink.line_strips3d(
        "world/target_path",
        &[target_path(home_pose)],
        &[CHROME],
        &[0.003],
    )?;
    let (g_o, g_v) = gnomon();
    sink.arrows3d("world/target/gnomon", &g_o, &g_v, &[TARGET])?;
    sink.arrows3d("world/arm/tool/gnomon", &g_o, &g_v, &[HERO])?;

    // Base is a world weld: static pose, logged once.
    let rest_poses = link_poses(tree, &Vector::zeros());
    sink.transform3d(
        "world/arm/base",
        rest_poses[0].translation().into_array(),
        rest_poses[0].rotation().quaternion().as_array(),
    )?;

    // Real geometry: loaded once, logged under each frame's path so per-tick transform3d moves it.
    let mesh_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(MESH_DIR);
    let mesh_path = |slot: usize| match slot {
        0 => "world/arm/base/mesh".to_owned(),
        TOOL_INDEX => "world/arm/tool/mesh".to_owned(),
        slot => format!("world/arm/link{slot}/mesh"),
    };
    for slot in 0..N_FRAMES {
        let (vertices, normals, triangles, colors) = load_body_mesh(&mesh_dir, slot);
        sink.mesh3d(&mesh_path(slot), &vertices, &triangles, &normals, &colors)?;
    }

    sink.series_style("plots/tool_error", ERROR, "tracking lag (m)", 2.0)?;
    sink.series_style("plots/tool_speed", TARGET, "tool speed (m/s)", 2.0)?;

    // Non-singular start pose, at rest.
    let mut joint_positions = readings(&HOME_POSTURE);
    let mut joint_velocities = Vector::<N_FRAMES, f64>::zeros();

    let mut pacer = Pacer::new();
    let mut solve_ring = LatencyRing::new(1024);
    let mut trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);
    let mut commanded_trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);

    let mut desired_positions = readings(&HOME_POSTURE);
    let mut residual_pos = 0.0;
    let mut stalled_ticks: u64 = 0;
    let mut previous_target = target_pose(home_pose, 0.0).translation();
    let mut lag_ring = LatencyRing::new(1024);

    let mut n: i64 = 0;
    loop {
        let _ = pacer.wait();
        n += 1;
        let time = n as f64 / 1000.0;
        sink.set_sequence("tick", n);

        let target = target_pose(home_pose, time);
        let tool_speed = (target.translation() - previous_target).norm() / TICK;
        previous_target = target.translation();

        let tick_start = Instant::now();

        // One FK solve, shared by the metric, the controller and the plant.
        let solved = tree.forward_kinematics(&joint_positions)?;
        let tool = solved
            .pose(TOOL_INDEX)
            .unwrap_or_else(|| unreachable!("every frame was settled"));
        let tracking_lag = (tool.translation() - target.translation()).norm();

        // Seeded from the measured configuration, not the last solve, so the solver stays near
        // where the arm is rather than near an ideal it never reached.
        let result = solver.solve(tree, TOOL_INDEX, target, &joint_positions);

        // Budget-out and stalled solves still return the nearest pose (control-loop semantics);
        // only a malformed request errors, and there the reference holds.
        if let Ok(report) = result {
            desired_positions = report.joint_positions;
            residual_pos = report.position_error;
            if report.termination == InverseKinematicsTermination::Stalled {
                stalled_ticks += 1;
            }
        }

        // The solve is a reference; the arm has to be driven to it.
        let reference = JointReference::at_rest(desired_positions);
        let torque = controller.torque(
            &body,
            &solved,
            &joint_positions,
            &joint_velocities,
            &reference,
        )?;
        let acceleration = body.forward_dynamics(&solved, &joint_velocities, &torque)?;

        // Semi-implicit Euler: the new rate carries the position, stable where the explicit form
        // drifts.
        joint_velocities += acceleration.scale(TICK);
        joint_positions += joint_velocities.scale(TICK);

        let tick_us = tick_start.elapsed().as_micros() as f64;

        // Fails at the tick a divergence happens rather than painting a scribble.
        assert!(
            torque.is_finite() && joint_positions.is_finite() && joint_velocities.is_finite(),
            "tick {n}: the loop diverged"
        );
        for slot in 1..=N_JOINTS {
            let (lower, upper) = tree
                .joint(slot)
                .unwrap_or_else(|| unreachable!())
                .limits()
                .unwrap_or_else(|| unreachable!("verify_model checked every joint has travel"));
            let reading = joint_positions[slot];
            assert!(
                reading >= lower - 0.05 && reading <= upper + 0.05,
                "tick {n}: joint {slot} left its travel at {reading}"
            );
        }

        if n > WARMUP_TICKS {
            solve_ring.push(tick_us);
            lag_ring.push(tracking_lag);
        }

        sink.scalar("plots/tool_error", tracking_lag)?;
        sink.scalar("plots/tool_speed", tool_speed)?;

        // Spatial geometry at ~60 Hz.
        if n % GEOM_EVERY == 0 {
            let poses = link_poses(tree, &joint_positions);
            for (slot, pose) in poses.iter().enumerate().skip(1).take(N_JOINTS) {
                sink.transform3d(
                    &format!("world/arm/link{slot}"),
                    pose.translation().into_array(),
                    pose.rotation().quaternion().as_array(),
                )?;
            }
            let drawn_tool = poses[TOOL_INDEX];
            sink.transform3d(
                "world/arm/tool",
                drawn_tool.translation().into_array(),
                drawn_tool.rotation().quaternion().as_array(),
            )?;
            sink.transform3d(
                "world/target",
                target.translation().into_array(),
                target.rotation().quaternion().as_array(),
            )?;

            // Skeleton: line through every frame origin.
            let skeleton: Vec<[f64; 3]> = poses
                .iter()
                .map(|pose| pose.translation().into_array())
                .collect();
            sink.line_strips3d("world/arm/skeleton", &[skeleton], &[HERO], &[0.012])?;

            // Commanded and reached: the gap between the two trails is the lag at life size.
            if trail.len() == TRAIL_MAX {
                trail.pop_front();
                commanded_trail.pop_front();
            }
            trail.push_back(drawn_tool.translation().into_array());
            commanded_trail.push_back(target.translation().into_array());
            sink.line_strips3d(
                "world/trail",
                &[trail.iter().copied().collect()],
                &[ACCENT],
                &[0.006],
            )?;
            sink.line_strips3d(
                "world/commanded_trail",
                &[commanded_trail.iter().copied().collect()],
                &[TARGET_FAINT],
                &[0.004],
            )?;
        }

        // Hud at 1 Hz.
        if n % HUD_EVERY == 0
            && let Some(solve_stats) = solve_ring.summary()
        {
            let lag = lag_ring.summary().map_or(0.0, |stats| stats.median);
            let meta = format!(
                "## 3d_arm_ik — Franka Panda, live from its model file\n\
                 ### SE(3) IK (damped least squares, analytic Jacobian) + gravity-compensated joint PD + articulated-body dynamics: median {:.0} µs · p99 {:.0} µs ({:.1} % of the 1 ms tick)\n\
                 ### tracking lag {:.2} mm at {:.2} m/s tool speed · IK residual {:.3} µm · {} stalled ticks",
                solve_stats.median,
                solve_stats.p99,
                solve_stats.p99 / 10.0,
                lag * 1e3,
                tool_speed,
                residual_pos * 1e6,
                stalled_ticks,
            );
            sink.text("hud/stats", &meta)?;
        }
    }
}
