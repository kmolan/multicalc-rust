//! 1 kHz Franka Panda under computed torque, with momentum-observer disturbance rejection and
//! sensorless contact detection (motion + kinematics + control + dynamics showcase).
//!
//! Panda from its MJCF file. `ComputedTorqueController` tracks a jerk-limited joint waypoint
//! cycle from `MotionProfilePlanner`; ABA on a second body is the plant. The plant carries a
//! payload at the hand that the controller's model lacks, and four pushes per cycle act at the
//! hand, `τ_ext = Jᵀ·F`.
//!
//! `MomentumObserver` on the controller's model gives `r ≈ τ_ext − (RNEA_plant − RNEA_model)` from
//! `q`, `q̇` and `τ`. The arm applies `τ = τ_computed − r`, which cancels the payload and the pushes
//! up to the observer lag `1/K`. `r` past per-joint thresholds, set by a push-free startup cycle,
//! flags each push. A second plant without the `− r` term is drawn as a red trail.
//!
//! Disturbance rejection, not compliance: a collaborative arm in contact with a person stops or
//! yields instead of holding its path.
//!
//! Streams to a Rerun viewer; see demos/README.md for WSL setup.
//! Run: cargo run --release -p multicalc-demos --example 3d_arm_torque

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::control::{ComputedTorqueController, JointReference};
use multicalc::dynamics::{ArticulatedBody, MomentumObserver};
use multicalc::error::MotionError;
use multicalc::kinematics::{JacobianFrame, JointKind, KinematicTree, KinematicTreeState};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::motion::{MotionProfilePlanner, ProfileLimits, SynchronizedProfile};
use multicalc::spatial::{SE3, SpatialInertia};
use multicalc_demos::loop_util::{LatencyRing, Pacer};
use multicalc_demos::visualization::franka_panda_mesh::load_body_mesh;
use multicalc_demos::{RerunSink, Rgba, VizSink};
use std::collections::VecDeque;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

// Palette, sRGB with alpha.
const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // tool gnomon
const TARGET: Rgba = [0xc9, 0x85, 0x00, 0xff]; // desired path, reference tool point
const ERROR: Rgba = [0xe6, 0x67, 0x67, 0xff]; // push arrow, contact marker, contact flag
const ERROR_FAINT: Rgba = [0xe6, 0x67, 0x67, 160]; // reference arm trail
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 200]; // tool trail
const CHROME: Rgba = [0x89, 0x87, 0x81, 0xff]; // payload
/// Observer residual per joint: the blue ramp `#86b6ef → #104281`, linear in sRGB, joint 1 lightest.
const RESIDUAL_RAMP: [Rgba; N_JOINTS] = [
    [0x86, 0xb6, 0xef, 0xff],
    [0x72, 0xa3, 0xdd, 0xff],
    [0x5f, 0x8f, 0xca, 0xff],
    [0x4b, 0x7c, 0xb8, 0xff],
    [0x37, 0x69, 0xa6, 0xff],
    [0x24, 0x55, 0x93, 0xff],
    [0x10, 0x42, 0x81, 0xff],
];

const MODEL_FILE: &str = "../third_party/menagerie/franka_emika_panda/panda.xml";
/// Mesh directory (panda.xml `<compiler meshdir="assets"/>`).
const MESH_DIR: &str = "../third_party/menagerie/franka_emika_panda/assets";
/// Tip frame: hand, one weld past the last hinge.
const TIP: &str = "hand";
const N_FRAMES: usize = 9; // link0, seven hinges, hand
const N_JOINTS: usize = 7; // movable joints
const TOOL_INDEX: usize = 8;

/// Joint waypoints, visited in order and closed back to the first; the first is panda.xml's
/// `home` keyframe.
#[allow(clippy::approx_constant)]
const WAYPOINTS: [[f64; N_JOINTS]; WAYPOINT_COUNT] = [
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853],
    [0.9, 0.35, 0.0, -1.95, 0.0, 2.3, 0.1],
    [-0.9, 0.55, 0.35, -1.35, 0.45, 1.9, -1.3],
    [0.0, -0.35, -0.4, -2.25, -0.35, 1.85, 0.45],
];
const WAYPOINT_COUNT: usize = 4;
const SPEED_LIMIT: f64 = 1.0; // rad/s, every joint
const ACCELERATION_LIMIT: f64 = 2.0; // rad/s²
const JERK_LIMIT: f64 = 10.0; // rad/s³
const DWELL: f64 = 1.5; // s held at each waypoint

/// Payload at the hand: a solid sphere, its centre `PAYLOAD_OFFSET` out along the hand z axis.
const PAYLOAD_MASS: f64 = 2.0; // kg
const PAYLOAD_OFFSET: [f64; 3] = [0.0, 0.0, 0.1]; // m, hand frame
const PAYLOAD_RADIUS: f64 = 0.05; // m

/// `ë + 2ζω·ė + ω²·e = 0` on every joint when the model is exact.
const NATURAL_FREQUENCY: f64 = 20.0; // rad/s
const DAMPING_RATIO: f64 = 1.0;
/// Observer bandwidth `K`: `ṙ = K ⊙ (d − r)`.
const OBSERVER_GAIN: f64 = 50.0; // rad/s

/// A push at the hand origin, world axes, over `[start, start + duration)` of cycle time.
struct Push {
    /// Short name for the hud.
    name: &'static str,
    force: [f64; 3], // N
    start: f64,      // s
    duration: f64,   // s
}

/// The pushes of one cycle, in time order and apart from each other.
const PUSHES: [Push; 4] = [
    Push {
        name: "tap at a stop",
        force: [0.0, -40.0, 0.0],
        start: 2.2,
        duration: 0.1,
    },
    Push {
        name: "bump at a stop",
        force: [0.0, 30.0, 0.0],
        start: 6.1,
        duration: 0.5,
    },
    Push {
        name: "bump in motion",
        force: [30.0, 0.0, 0.0],
        start: 8.0,
        duration: 0.5,
    },
    Push {
        name: "lean in motion",
        force: [0.0, 15.0, 0.0],
        start: 11.5,
        duration: 2.0,
    },
];
/// `5/K`: time for `r` to fall back inside the thresholds after a push ends.
const PUSH_SETTLE: f64 = 0.1; // s
/// Per-joint threshold: `margin·band + floor`, band from the push-free startup cycle.
const THRESHOLD_MARGIN: f64 = 1.5;
const THRESHOLD_FLOOR: f64 = 0.5; // N·m

const GRAVITY: f64 = -9.81;
const TICK: f64 = 1e-3; // loop period (s), matching `Pacer`
const GEOMETRY_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats
const TRAIL_MAX: usize = 180; // ~3 s of tool positions at 60 Hz
const PATH_SPACING: f64 = 0.01; // s between desired-path samples
const GNOMON: f64 = 0.18; // tool frame arrow base length
const PUSH_ARROW_SCALE: f64 = 0.01; // m per N

/// Joint readings; zero at the welded base/hand slots.
fn readings(joints: &[f64; N_JOINTS]) -> Vector<N_FRAMES, f64> {
    Vector::from_fn(|slot| {
        if (1..=N_JOINTS).contains(&slot) {
            joints[slot - 1]
        } else {
            0.0
        }
    })
}

/// The nominal Panda from panda.xml, and the plant: the same body with the payload combined into
/// the hand slot.
fn bodies() -> Result<
    (
        ArticulatedBody<N_FRAMES, N_FRAMES, f64>,
        ArticulatedBody<N_FRAMES, N_FRAMES, f64>,
    ),
    Box<dyn Error>,
> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(MODEL_FILE);
    let model = multicalc_robot_model::mjcf::load_path(&path)?;
    let nominal =
        model.articulated_body_to::<N_FRAMES, N_FRAMES>(TIP, Vector::new([0.0, 0.0, GRAVITY]))?;

    let sphere_inertia = 0.4 * PAYLOAD_MASS * PAYLOAD_RADIUS * PAYLOAD_RADIUS;
    let payload = SpatialInertia::new(
        PAYLOAD_MASS,
        Vector::new(PAYLOAD_OFFSET),
        Matrix::from_diagonal([sphere_inertia; 3]),
    )?;
    let inertias: [Option<SpatialInertia<f64>>; N_FRAMES] = core::array::from_fn(|slot| {
        let own = nominal.inertia(slot);
        if slot == TOOL_INDEX {
            Some(own.map_or(payload, |hand| hand.combined(payload)))
        } else {
            own
        }
    });
    let plant = ArticulatedBody::new(*nominal.tree(), &inertias, nominal.gravity())?;
    Ok((nominal, plant))
}

/// Startup check: model structure, and every waypoint inside its joint's travel.
fn verify_model(tree: &KinematicTree<N_FRAMES, N_FRAMES, f64>) {
    assert_eq!(tree.len(), N_FRAMES);
    assert_eq!(tree.joint(0).unwrap().kind(), JointKind::Fixed);
    assert_eq!(tree.joint(TOOL_INDEX).unwrap().kind(), JointKind::Fixed);
    for slot in 1..=N_JOINTS {
        let joint = tree.joint(slot).unwrap();
        assert_eq!(joint.kind(), JointKind::Revolute);
        let (lower, upper) = joint.limits().expect("every hinge has travel");
        for waypoint in &WAYPOINTS {
            let reading = waypoint[slot - 1];
            assert!(
                reading > lower && reading < upper,
                "waypoint {waypoint:?} leaves joint {slot}'s travel"
            );
        }
    }
}

/// The waypoint cycle: one synchronized jerk-limited profile per leg, then a dwell.
struct WaypointCycle {
    /// Leg `i` runs from `WAYPOINTS[i]` to `WAYPOINTS[(i + 1) % WAYPOINT_COUNT]`.
    legs: [SynchronizedProfile<N_JOINTS, f64>; WAYPOINT_COUNT],
    /// Cycle time at which each leg starts; the last entry is the period.
    starts: [f64; WAYPOINT_COUNT + 1],
}

impl WaypointCycle {
    fn plan() -> Result<Self, MotionError> {
        let limits = ProfileLimits::try_new(SPEED_LIMIT, ACCELERATION_LIMIT, Some(JERK_LIMIT))?;
        let planner = MotionProfilePlanner::new(limits);
        let mut legs = Vec::with_capacity(WAYPOINT_COUNT);
        for leg in 0..WAYPOINT_COUNT {
            let from = WAYPOINTS[leg];
            let destination = WAYPOINTS[(leg + 1) % WAYPOINT_COUNT];
            let mut axes = Vec::with_capacity(N_JOINTS);
            for joint in 0..N_JOINTS {
                axes.push(planner.plan(destination[joint] - from[joint])?);
            }
            legs.push(SynchronizedProfile::from_profiles(
                axes.try_into()
                    .unwrap_or_else(|_| unreachable!("seven axes")),
            ));
        }
        let legs: [SynchronizedProfile<N_JOINTS, f64>; WAYPOINT_COUNT] = legs
            .try_into()
            .unwrap_or_else(|_| unreachable!("four legs"));
        let mut starts = [0.0; WAYPOINT_COUNT + 1];
        for leg in 0..WAYPOINT_COUNT {
            starts[leg + 1] = starts[leg] + legs[leg].duration() + DWELL;
        }
        Ok(Self { legs, starts })
    }

    #[must_use]
    fn period(&self) -> f64 {
        self.starts[WAYPOINT_COUNT]
    }

    /// The reference at `time`; the profile clamps past a leg's end, so a dwell holds the
    /// waypoint at rest.
    fn reference_at(&self, time: f64) -> Result<JointReference<N_FRAMES, f64>, MotionError> {
        let cycle_time = time.rem_euclid(self.period());
        let leg = (0..WAYPOINT_COUNT)
            .rev()
            .find(|&leg| cycle_time >= self.starts[leg])
            .unwrap_or(0);
        let state = self.legs[leg].state_at(cycle_time - self.starts[leg])?;
        let from = WAYPOINTS[leg];
        let joint = |values: &Vector<N_JOINTS, f64>, offset: &[f64; N_JOINTS]| {
            Vector::from_fn(|slot| {
                if (1..=N_JOINTS).contains(&slot) {
                    values[slot - 1] + offset[slot - 1]
                } else {
                    0.0
                }
            })
        };
        let still = [0.0; N_JOINTS];
        Ok(JointReference::new(
            joint(&state.position, &from),
            joint(&state.velocity, &still),
            joint(&state.acceleration, &still),
        ))
    }
}

/// The push acting at `cycle_time`, if any.
#[must_use]
fn active_push(cycle_time: f64) -> Option<(usize, &'static Push)> {
    PUSHES
        .iter()
        .enumerate()
        .find(|(_, push)| cycle_time >= push.start && cycle_time < push.start + push.duration)
}

/// `true` inside a push or the `PUSH_SETTLE` after it.
#[must_use]
fn near_push(cycle_time: f64) -> bool {
    PUSHES.iter().any(|push| {
        cycle_time >= push.start && cycle_time < push.start + push.duration + PUSH_SETTLE
    })
}

/// What one tick produced, for the plots and the counters.
struct TickRecord {
    /// Poses the tick started from.
    state: KinematicTreeState<N_FRAMES, f64>,
    /// Desired tool position.
    reference_tool: [f64; 3],
    /// Observer residual `r`.
    residual: Vector<N_FRAMES, f64>,
    /// `‖RNEA_plant(q, q̇, ABA_plant(q, q̇, τ + τ_ext)) − (τ + τ_ext)‖`.
    round_trip_error: f64,
    /// `‖p_tool(q) − p_tool(q_d)‖`.
    tool_error: f64,
    /// Profile, FK, observer and computed torque.
    controller_microseconds: f64,
    /// ABA on the plant body.
    plant_microseconds: f64,
}

/// The two bodies, the controller, the observer and the integrated plant state.
struct ArmTorqueLoop {
    nominal: ArticulatedBody<N_FRAMES, N_FRAMES, f64>,
    plant: ArticulatedBody<N_FRAMES, N_FRAMES, f64>,
    cycle: WaypointCycle,
    controller: ComputedTorqueController<N_FRAMES, f64>,
    observer: MomentumObserver<N_FRAMES, f64>,
    /// Applies `τ = τ_computed − r` when set, `τ = τ_computed` when not.
    compensated: bool,
    joint_positions: Vector<N_FRAMES, f64>,
    joint_velocities: Vector<N_FRAMES, f64>,
    /// `τ` applied over the last tick, as the observer needs it.
    torque: Vector<N_FRAMES, f64>,
}

impl ArmTorqueLoop {
    /// At the first waypoint, at rest.
    fn new(compensated: bool) -> Result<Self, Box<dyn Error>> {
        let (nominal, plant) = bodies()?;
        Ok(Self {
            nominal,
            plant,
            cycle: WaypointCycle::plan()?,
            controller: ComputedTorqueController::from_natural_frequency(
                NATURAL_FREQUENCY,
                DAMPING_RATIO,
            )?,
            observer: MomentumObserver::new(Vector::from_fn(|_| OBSERVER_GAIN), TICK)?,
            compensated,
            joint_positions: readings(&WAYPOINTS[0]),
            joint_velocities: Vector::zeros(),
            torque: Vector::zeros(),
        })
    }

    /// One 1 ms tick at `time`, with `force` (N, world axes) at the hand origin.
    fn step(&mut self, time: f64, force: [f64; 3]) -> Result<TickRecord, Box<dyn Error>> {
        let tree = self.nominal.tree();

        let controller_start = Instant::now();
        let reference = self.cycle.reference_at(time)?;
        let state = tree.forward_kinematics(&self.joint_positions)?;
        let residual =
            self.observer
                .update(&self.nominal, &state, &self.joint_velocities, &self.torque)?;
        let computed = self.controller.torque(
            &self.nominal,
            &state,
            &self.joint_positions,
            &self.joint_velocities,
            &reference,
        )?;
        let torque = if self.compensated {
            computed - residual
        } else {
            computed
        };
        let controller_microseconds = controller_start.elapsed().as_nanos() as f64 / 1000.0;

        // τ_ext = Jᵀ·F, world-aligned Jacobian at the hand origin.
        let external_torque = if force == [0.0; 3] {
            Vector::zeros()
        } else {
            let jacobian = tree.geometric_jacobian(&state, TOOL_INDEX, JacobianFrame::World)?;
            let force = Vector::new(force);
            Vector::from_fn(|slot| {
                jacobian
                    .column(slot)
                    .map_or(0.0, |column| column.linear().dot(force))
            })
        };
        let applied = torque + external_torque;

        let plant_start = Instant::now();
        let acceleration = self
            .plant
            .forward_dynamics(&state, &self.joint_velocities, &applied)?;
        let plant_microseconds = plant_start.elapsed().as_nanos() as f64 / 1000.0;

        // Ground truth, untimed.
        let plant_torque =
            self.plant
                .inverse_dynamics(&state, &self.joint_velocities, &acceleration)?;
        let round_trip_error = (plant_torque - applied).norm();

        let reference_tool = tree
            .forward_kinematics(&reference.position)?
            .pose(TOOL_INDEX)
            .unwrap()
            .translation();
        let tool_error = (state.pose(TOOL_INDEX).unwrap().translation() - reference_tool).norm();

        // Semi-implicit Euler.
        self.joint_velocities += acceleration.scale(TICK);
        self.joint_positions += self.joint_velocities.scale(TICK);
        assert!(
            self.joint_positions.is_finite() && self.joint_velocities.is_finite(),
            "time {time:.3} s: the loop diverged"
        );

        self.torque = torque;
        Ok(TickRecord {
            state,
            reference_tool: reference_tool.into_array(),
            residual,
            round_trip_error,
            tool_error,
            controller_microseconds,
            plant_microseconds,
        })
    }
}

/// `true` when any joint's `|r_j|` exceeds its threshold.
#[must_use]
fn contact(residual: &Vector<N_FRAMES, f64>, thresholds: &[f64; N_JOINTS]) -> bool {
    (1..=N_JOINTS).any(|slot| residual[slot].abs() > thresholds[slot - 1])
}

/// What one push-free cycle reports.
struct Sweep {
    /// Per-joint `max |r_j|`.
    band: [f64; N_JOINTS],
    /// Per-joint `THRESHOLD_MARGIN·band + THRESHOLD_FLOOR`.
    thresholds: [f64; N_JOINTS],
    /// Per push, per joint `|τ_ext_j|` at the reference pose mid-push.
    push_torques: [[f64; N_JOINTS]; PUSHES.len()],
    /// Worst tool error, m.
    peak_tool_error: f64,
    /// Worst round-trip error, N·m.
    peak_round_trip_error: f64,
}

/// Startup sweep: one compensated cycle with no push, on a loop of its own.
fn startup_sweep() -> Result<Sweep, Box<dyn Error>> {
    let mut arm = ArmTorqueLoop::new(true)?;
    let steps = (arm.cycle.period() / TICK).ceil() as i64;
    let mut band = [0.0_f64; N_JOINTS];
    let mut peak_tool_error = 0.0_f64;
    let mut peak_round_trip_error = 0.0_f64;
    for tick in 0..steps {
        let record = arm.step(tick as f64 * TICK, [0.0; 3])?;
        for slot in 1..=N_JOINTS {
            band[slot - 1] = band[slot - 1].max(record.residual[slot].abs());
        }
        peak_tool_error = peak_tool_error.max(record.tool_error);
        peak_round_trip_error = peak_round_trip_error.max(record.round_trip_error);
    }
    let thresholds = band.map(|width| THRESHOLD_MARGIN * width + THRESHOLD_FLOOR);

    let tree = arm.nominal.tree();
    let mut push_torques = [[0.0; N_JOINTS]; PUSHES.len()];
    for (push, torques) in PUSHES.iter().zip(push_torques.iter_mut()) {
        let middle = arm.cycle.reference_at(push.start + 0.5 * push.duration)?;
        let pose = tree.forward_kinematics(&middle.position)?;
        let jacobian = tree.geometric_jacobian(&pose, TOOL_INDEX, JacobianFrame::World)?;
        *torques = core::array::from_fn(|joint| {
            jacobian.column(joint + 1).map_or(0.0, |column| {
                column.linear().dot(Vector::new(push.force)).abs()
            })
        });
    }
    Ok(Sweep {
        band,
        thresholds,
        push_torques,
        peak_tool_error,
        peak_round_trip_error,
    })
}

/// The desired tool path over one cycle, closed.
fn desired_path(arm: &ArmTorqueLoop) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    let period = arm.cycle.period();
    let samples = (period / PATH_SPACING).ceil() as usize;
    let mut path = Vec::with_capacity(samples + 1);
    for sample in 0..=samples {
        let reference = arm
            .cycle
            .reference_at(period * sample as f64 / samples as f64)?;
        let pose = arm
            .nominal
            .tree()
            .forward_kinematics(&reference.position)?
            .pose(TOOL_INDEX)
            .unwrap();
        path.push(pose.translation().into_array());
    }
    Ok(path)
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

/// Appends `point`, dropping the oldest past `TRAIL_MAX`.
fn extend_trail(trail: &mut VecDeque<[f64; 3]>, point: [f64; 3]) {
    if trail.len() == TRAIL_MAX {
        trail.pop_front();
    }
    trail.push_back(point);
}

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example 3d_arm_torque"
        );
    }

    let mut arm = ArmTorqueLoop::new(true)?;
    // Same controller without the `− r` term, drawn as a trail.
    let mut reference_arm = ArmTorqueLoop::new(false)?;
    verify_model(arm.nominal.tree());
    let period = arm.cycle.period();
    for pair in PUSHES.windows(2) {
        assert!(
            pair[0].start + pair[0].duration + PUSH_SETTLE <= pair[1].start,
            "pushes overlap"
        );
    }
    let last = &PUSHES[PUSHES.len() - 1];
    assert!(
        last.start + last.duration + PUSH_SETTLE <= period,
        "the last push runs past the cycle"
    );

    let sweep = startup_sweep()?;
    eprintln!("waypoint cycle: {WAYPOINT_COUNT} legs, period {period:.2} s");
    eprintln!(
        "push-free cycle: worst tool error {:.2} mm, worst round trip {:.1e} N·m",
        sweep.peak_tool_error * 1e3,
        sweep.peak_round_trip_error,
    );
    for joint in 0..N_JOINTS {
        eprintln!(
            "  joint {}: band {:6.2} N·m -> threshold {:6.2} N·m",
            joint + 1,
            sweep.band[joint],
            sweep.thresholds[joint],
        );
    }
    for (push, torques) in PUSHES.iter().zip(&sweep.push_torques) {
        let clearance = (0..N_JOINTS)
            .map(|joint| torques[joint] / sweep.thresholds[joint])
            .fold(0.0_f64, f64::max);
        eprintln!(
            "  {}: {:.0} N at {:.2} s for {:.1} s, {clearance:.1}× the threshold at best",
            push.name,
            Vector::new(push.force).norm(),
            push.start,
            push.duration,
        );
        assert!(
            clearance > 2.0,
            "{} does not clear twice any joint's threshold: raise its force",
            push.name
        );
    }

    let mut sink = RerunSink::live("multicalc-demos/3d-arm-torque")?;

    // Scene furniture, outside time.
    sink.set_static(true);
    let (gnomon_origins, gnomon_vectors) = gnomon();
    sink.arrows3d(
        "world/arm/tool/gnomon",
        &gnomon_origins,
        &gnomon_vectors,
        &[HERO],
    )?;
    sink.points3d_styled(
        "world/arm/tool/payload",
        &[PAYLOAD_OFFSET],
        &[CHROME],
        &[PAYLOAD_RADIUS as f32],
    )?;
    sink.line_strips3d(
        "world/desired_path",
        &[desired_path(&arm)?],
        &[TARGET],
        &[0.004],
    )?;
    let tree = *arm.nominal.tree();
    let base = tree.forward_kinematics(&Vector::zeros())?.pose(0).unwrap();
    sink.transform3d(
        "world/arm/base",
        base.translation().into_array(),
        base.rotation().quaternion().as_array(),
    )?;
    let mesh_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(MESH_DIR);
    for slot in 0..N_FRAMES {
        let path = match slot {
            0 => "world/arm/base/mesh".to_owned(),
            TOOL_INDEX => "world/arm/tool/mesh".to_owned(),
            slot => format!("world/arm/link{slot}/mesh"),
        };
        let mesh = load_body_mesh(&mesh_dir, slot)?;
        sink.mesh3d(
            &path,
            &mesh.vertices,
            &mesh.triangles,
            &mesh.normals,
            &mesh.colors,
        )?;
    }
    sink.set_static(false);

    let residual_names: [String; N_JOINTS] =
        core::array::from_fn(|joint| format!("joint {}: r (N·m)", joint + 1));
    let residual_names: Vec<&str> = residual_names.iter().map(String::as_str).collect();
    sink.series_styles("plots/residual", &RESIDUAL_RAMP, &residual_names, 2.0)?;
    sink.series_style("plots/contact", ERROR, "contact flag", 2.0)?;

    let mut pacer = Pacer::new();
    let mut controller_ring = LatencyRing::new(1024);
    let mut plant_ring = LatencyRing::new(1024);
    let mut tool_error_ring = LatencyRing::new(1024);
    let mut reference_tool_error_ring = LatencyRing::new(1024);
    let mut trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);
    let mut reference_trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);

    let mut pushes: u64 = 0;
    let mut pushes_detected: u64 = 0;
    let mut detected_this_push = false;
    let mut last_detection: Option<(&str, f64)> = None;
    let mut false_contact_ticks: u64 = 0;
    let mut previous_push: Option<usize> = None;

    let mut tick: i64 = 0;
    loop {
        let _ = pacer.wait();
        tick += 1;
        sink.set_sequence("tick", tick);
        let time = tick as f64 * TICK;
        let cycle_time = time.rem_euclid(period);
        let push = active_push(cycle_time);
        let push_index = push.map(|(index, _)| index);
        if push_index.is_some() && push_index != previous_push {
            pushes += 1;
            detected_this_push = false;
        }
        previous_push = push_index;
        let force = push.map_or([0.0; 3], |(_, push)| push.force);

        let record = arm.step(time, force)?;
        let reference_record = reference_arm.step(time, force)?;
        let in_contact = contact(&record.residual, &sweep.thresholds);
        if let Some((_, push)) = push
            && in_contact
            && !detected_this_push
        {
            detected_this_push = true;
            pushes_detected += 1;
            last_detection = Some((push.name, (cycle_time - push.start) * 1e3));
        }
        if in_contact && !near_push(cycle_time) {
            false_contact_ticks += 1;
        }

        if tick > WARMUP_TICKS {
            controller_ring.push(record.controller_microseconds);
            plant_ring.push(record.plant_microseconds);
            tool_error_ring.push(record.tool_error);
            reference_tool_error_ring.push(reference_record.tool_error);
        }

        let residuals: [f64; N_JOINTS] = core::array::from_fn(|joint| record.residual[joint + 1]);
        sink.scalars("plots/residual", &residuals)?;
        sink.scalar("plots/contact", if in_contact { 1.0 } else { 0.0 })?;

        if tick % GEOMETRY_EVERY == 0 {
            let poses: [SE3<f64>; N_FRAMES] =
                core::array::from_fn(|slot| record.state.pose(slot).unwrap());
            for (slot, pose) in poses.iter().enumerate().skip(1).take(N_JOINTS) {
                sink.transform3d(
                    &format!("world/arm/link{slot}"),
                    pose.translation().into_array(),
                    pose.rotation().quaternion().as_array(),
                )?;
            }
            let tool = poses[TOOL_INDEX];
            let tool_position = tool.translation().into_array();
            sink.transform3d(
                "world/arm/tool",
                tool_position,
                tool.rotation().quaternion().as_array(),
            )?;
            sink.points3d_styled(
                "world/reference_tool",
                &[record.reference_tool],
                &[TARGET],
                &[0.015],
            )?;

            // The push arrow ends at the hand; empty while no push acts.
            let (origins, vectors): (Vec<[f64; 3]>, Vec<[f64; 3]>) = match push {
                Some((_, push)) => {
                    let arrow = push.force.map(|component| component * PUSH_ARROW_SCALE);
                    (
                        vec![core::array::from_fn(|axis| {
                            tool_position[axis] - arrow[axis]
                        })],
                        vec![arrow],
                    )
                }
                None => (Vec::new(), Vec::new()),
            };
            sink.arrows3d("world/push", &origins, &vectors, &[ERROR])?;
            let contact_marker: Vec<[f64; 3]> = if in_contact {
                vec![tool_position]
            } else {
                Vec::new()
            };
            sink.points3d_styled("world/contact", &contact_marker, &[ERROR], &[0.08])?;

            extend_trail(&mut trail, tool_position);
            sink.line_strips3d(
                "world/trail",
                &[trail.iter().copied().collect()],
                &[ACCENT],
                &[0.006],
            )?;
            let reference_tool = reference_record
                .state
                .pose(TOOL_INDEX)
                .unwrap()
                .translation()
                .into_array();
            extend_trail(&mut reference_trail, reference_tool);
            sink.line_strips3d(
                "world/reference_arm_trail",
                &[reference_trail.iter().copied().collect()],
                &[ERROR_FAINT],
                &[0.004],
            )?;
        }

        if tick % HUD_EVERY == 0
            && let (Some(controller), Some(plant), Some(tool_error), Some(reference_tool_error)) = (
                controller_ring.summary(),
                plant_ring.summary(),
                tool_error_ring.summary(),
                reference_tool_error_ring.summary(),
            )
        {
            let latest = last_detection.map_or_else(
                || "none yet".to_owned(),
                |(name, milliseconds)| format!("last: {name}, after {milliseconds:.0} ms"),
            );
            let panel = format!(
                "## 3d_arm_torque — Franka Panda under computed torque, live from its model file\n\
                 ### controller tick: median {:.1} µs (p99 {:.1} µs) · plant: ABA {:.1} µs\n\
                 ### {PAYLOAD_MASS:.0} kg payload the model lacks, and {} pushes per cycle, cancelled by τ = τ_computed − r: tool error {:.2} mm · {:.1} mm without the − r term (red trail)\n\
                 ### pushes found from joint data only: {pushes_detected} of {pushes} ({latest}) · {false_contact_ticks} false contact ticks · RNEA ∘ ABA round trip {:.0e} N·m",
                controller.median,
                controller.p99,
                plant.median,
                PUSHES.len(),
                tool_error.median * 1e3,
                reference_tool_error.median * 1e3,
                record.round_trip_error,
            );
            sink.text("hud/stats", &panel)?;
        }
    }
}
