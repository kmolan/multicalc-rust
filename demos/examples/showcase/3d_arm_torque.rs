//! 1 kHz Franka Panda under computed torque, with sensorless contact detection (motion +
//! kinematics + control + dynamics showcase).
//!
//! Panda from its MJCF file. `ComputedTorqueController` tracks a jerk-limited joint waypoint
//! cycle from `MotionProfilePlanner`; ABA on a second body is the plant. The plant carries a
//! payload at the hand that the controller's model lacks, so the tracking error is real.
//!
//! `MomentumObserver` on the controller's model estimates `d = τ_ext − (RNEA_plant − RNEA_model)`
//! from `q`, `q̇` and `τ`. Once per cycle a push at the hand, `τ_ext = Jᵀ·F`, drives `r` past
//! per-joint thresholds set by a push-free startup cycle: collision detection without sensors.
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
const HERO: Rgba = [0x39, 0x87, 0xe5, 0xff]; // arm, tool gnomon, observer residual
const TARGET: Rgba = [0xc9, 0x85, 0x00, 0xff]; // reference tool point, true disturbance
const ERROR: Rgba = [0xe6, 0x67, 0x67, 0xff]; // push arrow, contact marker, error series
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 140]; // tool trail
const CHROME: Rgba = [0x89, 0x87, 0x81, 0xff]; // payload, round-trip series

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

/// Push at the hand origin, world axes, once per cycle during the dwell at `PUSH_WAYPOINT`.
const PUSH_FORCE: [f64; 3] = [0.0, 30.0, 0.0]; // N
const PUSH_WAYPOINT: usize = 2;
const PUSH_DELAY: f64 = 0.5; // s into the dwell
const PUSH_DURATION: f64 = 0.5; // s
/// `5/K`: time for `r` to fall back inside the thresholds after the push ends.
const PUSH_SETTLE: f64 = 0.1; // s
const _: () = assert!(
    PUSH_DELAY + PUSH_DURATION + PUSH_SETTLE <= DWELL,
    "the push and its settle must fit inside the dwell"
);
/// Per-joint threshold: `margin·band + floor`, band from the push-free startup cycle.
const THRESHOLD_MARGIN: f64 = 1.5;
const THRESHOLD_FLOOR: f64 = 0.5; // N·m

const GRAVITY: f64 = -9.81;
const TICK: f64 = 1e-3; // loop period (s), matching `Pacer`
const GEOMETRY_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from timing stats
const TRAIL_MAX: usize = 180; // ~3 s of tool positions at 60 Hz
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

    /// The push window `[start, end)` in cycle time: `PUSH_DELAY` into the dwell at
    /// `PUSH_WAYPOINT`.
    #[must_use]
    fn push_window(&self) -> (f64, f64) {
        let arriving_leg = PUSH_WAYPOINT - 1;
        let start = self.starts[arriving_leg] + self.legs[arriving_leg].duration() + PUSH_DELAY;
        (start, start + PUSH_DURATION)
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

/// What one tick produced, for the plots and the counters.
struct TickRecord {
    /// Poses the tick started from.
    state: KinematicTreeState<N_FRAMES, f64>,
    /// Desired tool position.
    reference_tool: [f64; 3],
    /// `τ_ext` applied this tick.
    external_torque: Vector<N_FRAMES, f64>,
    /// Observer residual `r`.
    residual: Vector<N_FRAMES, f64>,
    /// True `d` over the tick that `r` covers.
    disturbance: Vector<N_FRAMES, f64>,
    /// `‖RNEA_plant(q, q̇, ABA_plant(q, q̇, τ + τ_ext)) − (τ + τ_ext)‖`.
    round_trip_error: f64,
    /// `‖p_tool(q) − p_tool(q_d)‖`.
    tool_error: f64,
    /// Profile, FK, computed torque and observer.
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
    joint_positions: Vector<N_FRAMES, f64>,
    joint_velocities: Vector<N_FRAMES, f64>,
    /// `τ` applied over the last tick, as the observer needs it.
    torque: Vector<N_FRAMES, f64>,
    /// `d` over the last tick.
    disturbance: Vector<N_FRAMES, f64>,
}

impl ArmTorqueLoop {
    /// At the first waypoint, at rest.
    fn new() -> Result<Self, Box<dyn Error>> {
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
            joint_positions: readings(&WAYPOINTS[0]),
            joint_velocities: Vector::zeros(),
            torque: Vector::zeros(),
            disturbance: Vector::zeros(),
        })
    }

    /// One 1 ms tick at `time`; `pushing` applies `PUSH_FORCE` at the hand.
    fn step(&mut self, time: f64, pushing: bool) -> Result<TickRecord, Box<dyn Error>> {
        let tree = self.nominal.tree();

        let controller_start = Instant::now();
        let reference = self.cycle.reference_at(time)?;
        let state = tree.forward_kinematics(&self.joint_positions)?;
        let residual =
            self.observer
                .update(&self.nominal, &state, &self.joint_velocities, &self.torque)?;
        let torque = self.controller.torque(
            &self.nominal,
            &state,
            &self.joint_positions,
            &self.joint_velocities,
            &reference,
        )?;
        let controller_microseconds = controller_start.elapsed().as_nanos() as f64 / 1000.0;

        // τ_ext = Jᵀ·F, world-aligned Jacobian at the hand origin.
        let external_torque = if pushing {
            let jacobian = tree.geometric_jacobian(&state, TOOL_INDEX, JacobianFrame::World)?;
            let force = Vector::new(PUSH_FORCE);
            Vector::from_fn(|slot| {
                jacobian
                    .column(slot)
                    .map_or(0.0, |column| column.linear().dot(force))
            })
        } else {
            Vector::zeros()
        };
        let applied = torque + external_torque;

        let plant_start = Instant::now();
        let acceleration = self
            .plant
            .forward_dynamics(&state, &self.joint_velocities, &applied)?;
        let plant_microseconds = plant_start.elapsed().as_nanos() as f64 / 1000.0;

        // Ground truth, untimed: r estimates τ_ext less what the model misses of the payload.
        let plant_torque =
            self.plant
                .inverse_dynamics(&state, &self.joint_velocities, &acceleration)?;
        let model_torque =
            self.nominal
                .inverse_dynamics(&state, &self.joint_velocities, &acceleration)?;
        let round_trip_error = (plant_torque - applied).norm();
        let disturbance = external_torque - (plant_torque - model_torque);

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

        // r at this tick covers the tick that just ended, so it pairs with the last d.
        let record = TickRecord {
            state,
            reference_tool: reference_tool.into_array(),
            external_torque,
            residual,
            disturbance: self.disturbance,
            round_trip_error,
            tool_error,
            controller_microseconds,
            plant_microseconds,
        };
        self.torque = torque;
        self.disturbance = disturbance;
        Ok(record)
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
    /// Per-joint `|τ_ext_j|` of the push at the `PUSH_WAYPOINT` pose.
    push_torque: [f64; N_JOINTS],
    /// Worst tool error, m.
    peak_tool_error: f64,
    /// Worst round-trip error, N·m.
    peak_round_trip_error: f64,
}

/// Startup sweep: one cycle with no push, on a loop of its own.
fn startup_sweep() -> Result<Sweep, Box<dyn Error>> {
    let mut arm = ArmTorqueLoop::new()?;
    let steps = (arm.cycle.period() / TICK).ceil() as i64;
    let mut band = [0.0_f64; N_JOINTS];
    let mut peak_tool_error = 0.0_f64;
    let mut peak_round_trip_error = 0.0_f64;
    for tick in 0..steps {
        let record = arm.step(tick as f64 * TICK, false)?;
        for slot in 1..=N_JOINTS {
            band[slot - 1] = band[slot - 1].max(record.residual[slot].abs());
        }
        peak_tool_error = peak_tool_error.max(record.tool_error);
        peak_round_trip_error = peak_round_trip_error.max(record.round_trip_error);
    }
    let thresholds = band.map(|width| THRESHOLD_MARGIN * width + THRESHOLD_FLOOR);

    let tree = arm.nominal.tree();
    let push_pose = tree.forward_kinematics(&readings(&WAYPOINTS[PUSH_WAYPOINT]))?;
    let jacobian = tree.geometric_jacobian(&push_pose, TOOL_INDEX, JacobianFrame::World)?;
    let push_torque = core::array::from_fn(|joint| {
        jacobian.column(joint + 1).map_or(0.0, |column| {
            column.linear().dot(Vector::new(PUSH_FORCE)).abs()
        })
    });
    Ok(Sweep {
        band,
        thresholds,
        push_torque,
        peak_tool_error,
        peak_round_trip_error,
    })
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

fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        eprintln!(
            "WARNING: debug build — timing numbers are meaningless. \
             Re-run with: cargo run --release -p multicalc-demos --example 3d_arm_torque"
        );
    }

    let mut arm = ArmTorqueLoop::new()?;
    verify_model(arm.nominal.tree());
    let period = arm.cycle.period();
    let (push_start, push_end) = arm.cycle.push_window();

    let sweep = startup_sweep()?;
    eprintln!(
        "waypoint cycle: {WAYPOINT_COUNT} legs, period {period:.2} s, push {:.0} N at {push_start:.2}..{push_end:.2} s",
        Vector::new(PUSH_FORCE).norm(),
    );
    eprintln!(
        "push-free cycle: worst tool error {:.1} mm, worst round trip {:.1e} N·m",
        sweep.peak_tool_error * 1e3,
        sweep.peak_round_trip_error,
    );
    for joint in 0..N_JOINTS {
        eprintln!(
            "  joint {}: band {:6.2} N·m -> threshold {:6.2} N·m, push gives {:6.2} N·m",
            joint + 1,
            sweep.band[joint],
            sweep.thresholds[joint],
            sweep.push_torque[joint],
        );
    }
    assert!(
        (0..N_JOINTS).any(|joint| sweep.push_torque[joint] > 2.0 * sweep.thresholds[joint]),
        "the push does not clear twice any joint's threshold: raise PUSH_FORCE"
    );

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

    for joint in 1..=N_JOINTS {
        sink.series_styles(
            &format!("plots/residual/joint{joint}"),
            &[HERO, TARGET],
            &[
                &format!("joint {joint}: r, observer (N·m)"),
                &format!("joint {joint}: d, true (N·m)"),
            ],
            2.0,
        )?;
    }
    sink.series_style("plots/contact", ERROR, "contact flag", 2.0)?;
    sink.series_style("plots/tool_error", ERROR, "tool tracking error (mm)", 2.0)?;
    sink.series_style("plots/observer_error", ERROR, "‖r − d‖ (N·m)", 2.0)?;
    sink.series_style("plots/round_trip", CHROME, "‖RNEA(ABA(τ)) − τ‖ (N·m)", 2.0)?;

    let mut pacer = Pacer::new();
    let mut controller_ring = LatencyRing::new(1024);
    let mut plant_ring = LatencyRing::new(1024);
    let mut tool_error_ring = LatencyRing::new(1024);
    let mut trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);

    let mut pushes: u64 = 0;
    let mut pushes_detected: u64 = 0;
    let mut detected_this_push = false;
    let mut last_detection_milliseconds: Option<f64> = None;
    let mut false_contact_ticks: u64 = 0;
    let mut was_pushing = false;

    let mut tick: i64 = 0;
    loop {
        let _ = pacer.wait();
        tick += 1;
        sink.set_sequence("tick", tick);
        let time = tick as f64 * TICK;
        let cycle_time = time.rem_euclid(period);
        let pushing = cycle_time >= push_start && cycle_time < push_end;
        if pushing && !was_pushing {
            pushes += 1;
            detected_this_push = false;
        }
        was_pushing = pushing;

        let record = arm.step(time, pushing)?;
        let in_contact = contact(&record.residual, &sweep.thresholds);
        if in_contact && pushing && !detected_this_push {
            detected_this_push = true;
            pushes_detected += 1;
            last_detection_milliseconds = Some((cycle_time - push_start) * 1e3);
        }
        let push_or_settle = cycle_time >= push_start && cycle_time < push_end + PUSH_SETTLE;
        if in_contact && !push_or_settle {
            false_contact_ticks += 1;
        }

        if tick > WARMUP_TICKS {
            controller_ring.push(record.controller_microseconds);
            plant_ring.push(record.plant_microseconds);
            tool_error_ring.push(record.tool_error);
        }

        for joint in 1..=N_JOINTS {
            sink.scalars(
                &format!("plots/residual/joint{joint}"),
                &[record.residual[joint], record.disturbance[joint]],
            )?;
        }
        sink.scalar("plots/contact", if in_contact { 1.0 } else { 0.0 })?;
        sink.scalar("plots/tool_error", record.tool_error * 1e3)?;
        sink.scalar(
            "plots/observer_error",
            (record.residual - record.disturbance).norm(),
        )?;
        sink.scalar("plots/round_trip", record.round_trip_error)?;

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
                &[0.02],
            )?;

            // The push arrow ends at the hand; empty while no push acts.
            let (origins, vectors): (Vec<[f64; 3]>, Vec<[f64; 3]>) =
                if record.external_torque.norm() > 0.0 {
                    let arrow = PUSH_FORCE.map(|component| component * PUSH_ARROW_SCALE);
                    (
                        vec![core::array::from_fn(|axis| {
                            tool_position[axis] - arrow[axis]
                        })],
                        vec![arrow],
                    )
                } else {
                    (Vec::new(), Vec::new())
                };
            sink.arrows3d("world/push", &origins, &vectors, &[ERROR])?;
            let contact_marker: Vec<[f64; 3]> = if in_contact {
                vec![tool_position]
            } else {
                Vec::new()
            };
            sink.points3d_styled("world/contact", &contact_marker, &[ERROR], &[0.08])?;

            if trail.len() == TRAIL_MAX {
                trail.pop_front();
            }
            trail.push_back(tool_position);
            sink.line_strips3d(
                "world/trail",
                &[trail.iter().copied().collect()],
                &[ACCENT],
                &[0.006],
            )?;
        }

        if tick % HUD_EVERY == 0
            && let (Some(controller), Some(plant), Some(tool_error)) = (
                controller_ring.summary(),
                plant_ring.summary(),
                tool_error_ring.summary(),
            )
        {
            let latency = last_detection_milliseconds.map_or_else(
                || "none yet".to_owned(),
                |milliseconds| format!("{milliseconds:.0} ms"),
            );
            let panel = format!(
                "## 3d_arm_torque — Franka Panda under computed torque, live from its model file\n\
                 ### modules: `motion` · `kinematics` · `control` · `dynamics` — median {:.1} µs per tick (p99 {:.1} µs) · plant: ABA {:.1} µs\n\
                 ### {PAYLOAD_MASS:.0} kg payload the model lacks: tool tracking error {:.1} mm · RNEA ∘ ABA round trip {:.0e} N·m\n\
                 ### collision detection without sensors: {pushes_detected} of {pushes} pushes caught, last after {latency} · {false_contact_ticks} false contact ticks",
                controller.median,
                controller.p99,
                plant.median,
                tool_error.median * 1e3,
                record.round_trip_error,
            );
            sink.text("hud/stats", &panel)?;
        }
    }
}
