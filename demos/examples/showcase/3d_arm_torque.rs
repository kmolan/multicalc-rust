//! 1 kHz Franka Panda under computed torque with momentum-observer disturbance rejection (motion +
//! kinematics + control + dynamics showcase).
//!
//! Panda from its MJCF file. `ComputedTorqueController` tracks a jerk-limited joint waypoint
//! cycle from `MotionProfilePlanner`; ABA on a second body is the plant. The plant carries a
//! payload at the hand that the controller's model lacks, and a near-constant push from a random
//! direction acts at the hand, `τ_ext = Jᵀ·F`.
//!
//! `MomentumObserver` on the controller's model gives `r ≈ τ_ext − (RNEA_plant − RNEA_model)` from
//! `q`, `q̇` and `τ`. The arm applies `τ = τ_computed − r`, which cancels the payload and the push
//! up to the observer lag `1/K`. A second plant without the `− r` term is drawn as a red trail.
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
use multicalc_demos::loop_util::{LatencyRing, Pacer, RECORD_TICKS};
use multicalc_demos::visualization::franka_panda_mesh::load_body_mesh;
use multicalc_demos::{RerunSink, Rgba, VizSink};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, UnitSphere};
use rand_pcg::Pcg32;
use std::collections::VecDeque;
use std::error::Error;
use std::f64::consts::PI;
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

// Palette, sRGB with alpha.
const TARGET: Rgba = [0xc9, 0x85, 0x00, 0xff]; // desired path, waypoints, reference tool point
const ERROR: Rgba = [0xe6, 0x67, 0x67, 0xff]; // push arrow, reference arm trail
const ACCENT: Rgba = [0x90, 0x85, 0xe9, 0xff]; // tool trail
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

/// Pushes at the hand origin: one after another, each from a uniform random direction.
const PUSH_SEED: u64 = 7;
const PUSH_FORCE: Range<f64> = 15.0..35.0; // N
const PUSH_DURATION: Range<f64> = 1.0..2.5; // s
const PUSH_GAP: Range<f64> = 0.1..0.4; // s
/// Raised-cosine ramp at each end of a push.
const PUSH_RAMP: f64 = 0.2; // s
const FIRST_PUSH: f64 = 1.0; // s

const GRAVITY: f64 = -9.81;
const TICK: f64 = 1e-3; // loop period (s), matching `Pacer`
const GEOMETRY_EVERY: i64 = 16; // spatial cadence (~60 Hz)
const HUD_EVERY: i64 = 1000; // text cadence (1 Hz)
const WARMUP_TICKS: i64 = 500; // cold-start ticks excluded from the readouts
const TRAIL_MAX: usize = 900; // one cycle of tool positions at 60 Hz
const PATH_SPACING: f64 = 0.01; // s between desired-path samples
const PUSH_ARROW_SCALE: f64 = 0.012; // m per N
const PUSH_ARROW_RADIUS: f32 = 0.012; // m

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

/// A seeded sequence of pushes, each from a uniform random direction, apart by short gaps.
struct RandomPushes {
    rng: Pcg32,
    /// The current push: `[start, end)` and its full force.
    start: f64,
    end: f64,
    force: [f64; 3],
}

impl RandomPushes {
    fn new() -> Self {
        let mut pushes = Self {
            rng: Pcg32::seed_from_u64(PUSH_SEED),
            start: 0.0,
            end: 0.0,
            force: [0.0; 3],
        };
        pushes.draw(FIRST_PUSH);
        pushes
    }

    /// Draws the next push, starting `start` seconds in.
    fn draw(&mut self, start: f64) {
        let direction: [f64; 3] = UnitSphere.sample(&mut self.rng);
        let magnitude = self.rng.random_range(PUSH_FORCE);
        self.start = start;
        self.end = start + self.rng.random_range(PUSH_DURATION);
        self.force = direction.map(|component| component * magnitude);
    }

    /// The force at `time` (N, world axes); `time` must not decrease between calls.
    ///
    /// Each end ramps on `w = (1 − cos(π·min(1, s/PUSH_RAMP)))/2`, `s` the time to the nearer end.
    fn force_at(&mut self, time: f64) -> [f64; 3] {
        if time >= self.end {
            let gap = self.rng.random_range(PUSH_GAP);
            self.draw(self.end + gap);
        }
        if time < self.start {
            return [0.0; 3];
        }
        let ramp = ((time - self.start).min(self.end - time) / PUSH_RAMP).min(1.0);
        let weight = 0.5 * (1.0 - (PI * ramp).cos());
        self.force.map(|component| component * weight)
    }
}

/// What one tick produced, for the plots and the readouts.
struct TickRecord {
    /// Poses the tick started from.
    state: KinematicTreeState<N_FRAMES, f64>,
    /// Desired tool position.
    reference_tool: [f64; 3],
    /// Observer residual `r`.
    residual: Vector<N_FRAMES, f64>,
    /// `‖p_tool(q) − p_tool(q_d)‖`.
    tool_error: f64,
    /// Profile, FK, observer and computed torque.
    controller_microseconds: f64,
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
        let jacobian = tree.geometric_jacobian(&state, TOOL_INDEX, JacobianFrame::World)?;
        let force = Vector::new(force);
        let external_torque: Vector<N_FRAMES, f64> = Vector::from_fn(|slot| {
            jacobian
                .column(slot)
                .map_or(0.0, |column| column.linear().dot(force))
        });
        let acceleration = self.plant.forward_dynamics(
            &state,
            &self.joint_velocities,
            &(torque + external_torque),
        )?;

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
            tool_error,
            controller_microseconds,
        })
    }
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

/// The tool position at each waypoint.
fn waypoint_tools(
    tree: &KinematicTree<N_FRAMES, N_FRAMES, f64>,
) -> Result<Vec<[f64; 3]>, Box<dyn Error>> {
    WAYPOINTS
        .iter()
        .map(|waypoint| {
            let pose = tree
                .forward_kinematics(&readings(waypoint))?
                .pose(TOOL_INDEX)
                .unwrap();
            Ok(pose.translation().into_array())
        })
        .collect()
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
    let mut pushes = RandomPushes::new();

    let mut sink = RerunSink::from_environment("multicalc-demos/3d-arm-torque")?;

    // Scene furniture, outside time.
    sink.set_static(true);
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
        &[0.003],
    )?;
    sink.points3d_styled(
        "world/waypoints",
        &waypoint_tools(arm.nominal.tree())?,
        &[TARGET],
        &[0.025],
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

    let mut pacer = Pacer::new();
    let mut controller_ring = LatencyRing::new(1000);
    let mut tool_error_ring = LatencyRing::new(1000);
    let mut reference_tool_error_ring = LatencyRing::new(1000);
    let mut trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);
    let mut reference_trail: VecDeque<[f64; 3]> = VecDeque::with_capacity(TRAIL_MAX);

    let mut tick: i64 = 0;
    loop {
        if sink.is_recording() && tick == RECORD_TICKS {
            break;
        }
        let _ = pacer.wait();
        tick += 1;
        sink.set_sequence("tick", tick);
        let time = tick as f64 * TICK;
        let force = pushes.force_at(time);

        let record = arm.step(time, force)?;
        let reference_record = reference_arm.step(time, force)?;

        if tick > WARMUP_TICKS {
            controller_ring.push(record.controller_microseconds);
            tool_error_ring.push(record.tool_error);
            reference_tool_error_ring.push(reference_record.tool_error);
        }

        let residuals: [f64; N_JOINTS] = core::array::from_fn(|joint| record.residual[joint + 1]);
        sink.scalars("plots/residual", &residuals)?;

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

            // The push arrow ends at the hand.
            let arrow = force.map(|component| component * PUSH_ARROW_SCALE);
            let origin: [f64; 3] = core::array::from_fn(|axis| tool_position[axis] - arrow[axis]);
            sink.arrows3d_styled(
                "world/push",
                &[origin],
                &[arrow],
                &[ERROR],
                &[PUSH_ARROW_RADIUS],
            )?;

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
                &[ERROR],
                &[0.004],
            )?;
        }

        if tick % HUD_EVERY == 0
            && let (Some(controller), Some(tool_error), Some(reference_tool_error)) = (
                controller_ring.summary(),
                tool_error_ring.summary(),
                reference_tool_error_ring.summary(),
            )
        {
            let panel = format!(
                "## 3d_arm_torque — random pushes, an unknown {PAYLOAD_MASS:.0} kg payload, and the hand stays on the path\n\
                 ### at worst {:.1} mm off the path · {:.0} mm without the observer (red)\n\
                 ### tick: `SynchronizedProfile` · `KinematicTree` · `MomentumObserver` · `ComputedTorqueController` — median {:.1} µs of the 1 ms tick · no force sensor",
                tool_error.max * 1e3,
                reference_tool_error.max * 1e3,
                controller.median,
            );
            sink.text("hud/stats", &panel)?;
        }
    }
    sink.flush()?;
    Ok(())
}
