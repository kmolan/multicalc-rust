//! 1 kHz Franka Panda SE(3) IK (kinematics showcase).
//!
//! Franka Panda, loaded from its MuJoCo model file, tracks a moving 3D target in position and
//! orientation via damped-least-squares IK against the analytic geometric Jacobian, at 1 kHz,
//! respecting the model's joint limits. Secondary objective: hold a comfortable posture with
//! leftover DOF. Panel shows solve cost against the 1 ms budget.
//!
//! Streams to a Rerun viewer; see demos/README.md for WSL setup.
//! Run: cargo run --release -p multicalc-demos --example 3d_arm_ik

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::kinematics::{
    InverseKinematics, InverseKinematicsTermination, JointKind, KinematicTree, SecondaryObjective,
};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{Quaternion, SE3, SO3};
use multicalc_demos::loop_util::{LatencyRing, Pacer};
use multicalc_demos::{RerunSink, Rgba, VizSink};
use std::collections::{HashMap, VecDeque};
use std::f64::consts::TAU;
use std::path::Path;
use std::time::Instant;

// Palette (§2), sRGB with alpha.
const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // arm, ee gnomon
const TARGET: Rgba = [0xc9, 0x85, 0x00, 0xff]; // target frame
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 140]; // ee trail
const CHROME: Rgba = [0x89, 0x87, 0x81, 0xff]; // target path

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

const CYCLE: f64 = 20.0; // orientation keyframe cycle (s), 5 s/segment
const MAXIMUM_ITERATIONS: usize = 40; // IK budget per tick
const SECONDARY_GAIN: f64 = 0.2; // secondary-objective weight

const GEOM_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats
const TRAIL_MAX: usize = 180; // ~3 s of ee positions at 60 Hz
const TARGET_PATH_SEGS: usize = 256;
const GNOMON: f64 = 0.18; // ee/target frame arrow base length

/// Target-orientation keyframes: small turns (wrist stays in travel), slerped over `CYCLE`,
/// clear of the `log` θ = π branch.
#[must_use]
fn keyframes() -> [Quaternion<f64>; 4] {
    [
        Quaternion::from_euler_zyx(0.0, 0.0, 0.0),
        Quaternion::from_euler_zyx(0.20, 0.15, 0.10),
        Quaternion::from_euler_zyx(-0.18, 0.22, -0.12),
        Quaternion::from_euler_zyx(0.10, -0.20, 0.25),
    ]
}

/// Loads the Franka Panda model: base-to-hand chain.
fn arm() -> Result<KinematicTree<N_FRAMES, f64>, Box<dyn std::error::Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_FILE);
    let model = multicalc_mjcf::load_path(&path)?;
    Ok(model.kinematic_tree_to::<N_FRAMES>(TIP)?)
}

/// Joint readings; zero at the welded base/hand slots.
fn readings(q: &[f64; N_JOINTS]) -> Vector<N_FRAMES, f64> {
    Vector::from_fn(|i| {
        if (1..=N_JOINTS).contains(&i) {
            q[i - 1]
        } else {
            0.0
        }
    })
}

/// All 9 frame poses for one configuration.
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

/// Startup check: model structure and default-class joint settings loaded correctly.
///
/// Bounds are loose by design — catches a mis-parsed chain without duplicating the fixture test's
/// exact values.
fn verify_model(tree: &KinematicTree<N_FRAMES, f64>) {
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
    // q = 0 rest pose: straight up.
    let hand = link_poses(tree, &Vector::zeros())[TOOL_INDEX];
    let [x, y, z] = *hand.translation().as_array();
    assert!(z > 0.8 && x.hypot(y) < 0.3, "hand at rest: {:?}", [x, y, z]);
}

/// Target position: Lissajous figure around the rest pose, amplitude per axis measured against
/// its own convergence cliff (y ~0.21, x/z ~0.07/0.055) rather than clamped to the tightest one.
#[must_use]
fn lissajous_position(home: [f64; 3], t: f64) -> [f64; 3] {
    [
        home[0] + 0.06 * (TAU * 0.11 * t).sin(),
        home[1] + 0.20 * (TAU * 0.17 * t + 0.4).sin(),
        home[2] + 0.05 * (TAU * 0.07 * t).sin(),
    ]
}

/// Target orientation: rest orientation composed with slerped keyframes, period `CYCLE`.
#[must_use]
fn target_orientation(home: SO3<f64>, t: f64) -> SO3<f64> {
    let keys = keyframes();
    let seg = CYCLE / 4.0;
    let phase = (t % CYCLE) / seg; // 0..4
    let i = phase.floor() as usize % 4;
    let frac = phase - phase.floor();
    let slerped = keys[i].slerp(keys[(i + 1) % 4], frac);
    home * SO3::from_quaternion(slerped)
}

/// Target pose at `t`: Lissajous position, keyframe-composed orientation.
#[must_use]
fn target_pose(home: SE3<f64>, t: f64) -> SE3<f64> {
    SE3::from_parts(
        target_orientation(home.rotation(), t),
        Vector::new(lissajous_position(home.translation().into_array(), t)),
    )
}

/// Target path over one cycle, closed.
#[must_use]
fn target_path(home: SE3<f64>) -> Vec<[f64; 3]> {
    (0..=TARGET_PATH_SEGS)
        .map(|i| {
            let t = CYCLE * i as f64 / TARGET_PATH_SEGS as f64;
            target_pose(home, t).translation().into_array()
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
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading mesh file {path:?}: {e}"));

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
                .map(|[a, b, c]| [a + offset, b + offset, c + offset]),
        );
    }
    (vertices, normals, triangles, colors)
}

/// Startup reachability check: worst position residual over one cycle at 1 ms cadence, post-warmup.
#[must_use]
fn reachability_sweep(
    solver: &InverseKinematics<N_FRAMES, f64>,
    tree: &KinematicTree<N_FRAMES, f64>,
    home_pose: SE3<f64>,
) -> f64 {
    let mut joint_readings = readings(&HOME_POSTURE);
    let steps = (CYCLE * 1000.0) as i64; // 1 ms spacing
    let mut worst = 0.0_f64;
    for n in 1..=steps {
        let t = n as f64 / 1000.0;
        let Ok(report) = solver.solve(tree, TOOL_INDEX, target_pose(home_pose, t), &joint_readings)
        else {
            continue;
        };
        joint_readings = report.joint_positions;
        if n > 200 {
            worst = worst.max(report.position_error);
        }
    }
    worst
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example 3d_arm_ik"
        );
    }

    // Built once: rebuilding per-tick would pollute the timing.
    let tree = arm()?;
    verify_model(&tree);

    let solver = InverseKinematics::<N_FRAMES, f64>::new()
        .with_maximum_iterations(MAXIMUM_ITERATIONS)
        .with_position_tolerance(1e-6)
        .with_orientation_tolerance(1e-6)
        .with_secondary_objective(SecondaryObjective::PreferredPosture(readings(
            &HOME_POSTURE,
        )))
        .with_secondary_gain(SECONDARY_GAIN);

    let home_pose = link_poses(&tree, &readings(&HOME_POSTURE))[TOOL_INDEX];

    let worst_reach = reachability_sweep(&solver, &tree, home_pose);
    eprintln!("reachability sweep: worst position residual over one cycle = {worst_reach:.2e} m");

    let mut rr = RerunSink::live("multicalc-demos/3d-arm-ik")?;

    // Statics: tick 0, forward-fill (see rerun-viz-gotchas).
    rr.set_sequence("tick", 0);
    rr.line_strips3d(
        "world/target_path",
        &[target_path(home_pose)],
        &[CHROME],
        &[0.003],
    )?;
    let (g_o, g_v) = gnomon();
    rr.arrows3d("world/target/gnomon", &g_o, &g_v, &[TARGET])?;
    rr.arrows3d("world/arm/tool/gnomon", &g_o, &g_v, &[HERO])?;

    // Base is a world weld: static pose, logged once.
    let rest_poses = link_poses(&tree, &Vector::zeros());
    rr.transform3d(
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
        rr.mesh3d(&mesh_path(slot), &vertices, &triangles, &normals, &colors)?;
    }

    // Non-singular start pose.
    let mut joint_readings = readings(&HOME_POSTURE);

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

        let target = target_pose(home_pose, t);
        let t0 = Instant::now();
        let result = solver.solve(&tree, TOOL_INDEX, target, &joint_readings);
        let solve_us = t0.elapsed().as_micros() as f64;

        // Budget-out/stalled solves still return the nearest pose (control-loop semantics); only a
        // malformed request errors, and there the arm holds.
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
            for (slot, pose) in poses.iter().enumerate().skip(1).take(N_JOINTS) {
                rr.transform3d(
                    &format!("world/arm/link{slot}"),
                    pose.translation().into_array(),
                    pose.rotation().quaternion().as_array(),
                )?;
            }
            let tool = poses[TOOL_INDEX];
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

            // Skeleton: line through every frame origin.
            let skeleton: Vec<[f64; 3]> = poses
                .iter()
                .map(|pose| pose.translation().into_array())
                .collect();
            rr.line_strips3d("world/arm/skeleton", &[skeleton], &[HERO], &[0.012])?;

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
                "## 3d_arm_ik — Franka Panda, live from its model file\n\
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
