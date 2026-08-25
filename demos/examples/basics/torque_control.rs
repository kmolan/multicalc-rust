//! Model-based torque control for a jointed robot: computed torque tracking a moving reference,
//! the closed-loop error dynamics the gains ask for, an impedance yielding to a push, the droop
//! gravity compensation removes from a joint PD, and a position-controlled joint's own servo.
//!
//! Each scenario is checked against the property that defines the law it shows.
//!
//! Run with: `cargo run -p multicalc-demos --example torque_control`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::control::{
    ComputedTorqueController, JointImpedanceController, JointPdController, JointReference,
};
use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::plant::PositionServo;
use multicalc::spatial::{SE3, SO3, SpatialInertia};

const GRAVITY: f64 = -9.81;
const LINK_LENGTH: f64 = 1.0;
const LINK_MASS: f64 = 1.5;
const CENTER_OF_MASS: f64 = 0.5;
const LINK_INERTIA: f64 = 0.02;
const TICK: f64 = 1e-3;

fn main() {
    computed_torque_tracks_a_moving_reference();
    error_dynamics_are_the_gains_you_asked_for();
    impedance_yields_to_a_push();
    gravity_compensation_removes_the_droop();
    position_servo_answers_a_command();
}

/// Two hinges about `+y` on a unit link, each carrying a 1.5 kg arm balancing half a metre out.
///
/// `loaded` chooses whether the joints carry their armature, damping and friction loss.
fn two_link_pendulum(loaded: bool) -> ArticulatedBody<2, 2, f64> {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let link = SE3::from_parts(SO3::identity(), Vector::new([LINK_LENGTH, 0.0, 0.0]));
    let origins = [SE3::identity(), link];
    let armature = [0.05, 0.03];
    let damping = [0.7, 0.4];
    let friction_loss = [0.2, 0.1];

    let mut tree = KinematicTree::<2, 2, f64>::new();
    for index in 0..2 {
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        let joint = if loaded {
            Joint::revolute(axis, origins[index])
                .with_armature(armature[index])
                .with_damping(damping[index])
                .with_friction_loss(friction_loss[index])
        } else {
            Joint::revolute(axis, origins[index])
        };
        tree.push(joint, parent).unwrap();
    }

    let arm = SpatialInertia::new(
        LINK_MASS,
        Vector::new([CENTER_OF_MASS, 0.0, 0.0]),
        Matrix::from_diagonal([LINK_INERTIA, LINK_INERTIA, LINK_INERTIA]),
    )
    .unwrap();
    ArticulatedBody::new(
        tree,
        &[Some(arm), Some(arm)],
        Vector::new([0.0, 0.0, GRAVITY]),
    )
    .unwrap()
}

/// One hinge about `+y`, a 1.5 kg arm balancing half a metre out. No armature or friction.
fn single_pendulum() -> ArticulatedBody<1, 1, f64> {
    let hinge = Joint::revolute(Vector::new([0.0, 1.0, 0.0]), SE3::identity());
    let tree =
        KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World]).unwrap();
    let arm = SpatialInertia::new(
        LINK_MASS,
        Vector::new([CENTER_OF_MASS, 0.0, 0.0]),
        Matrix::from_diagonal([LINK_INERTIA, LINK_INERTIA, LINK_INERTIA]),
    )
    .unwrap();
    ArticulatedBody::new(tree, &[Some(arm)], Vector::new([0.0, 0.0, GRAVITY])).unwrap()
}

/// One tick of the closed loop by explicit Euler on the crate's own forward dynamics.
fn advance<const N: usize>(
    body: &ArticulatedBody<N, N, f64>,
    position: &mut Vector<N, f64>,
    velocity: &mut Vector<N, f64>,
    torque: &Vector<N, f64>,
) {
    let acceleration = body
        .forward_dynamics_at(position, velocity, torque)
        .unwrap();
    *velocity += acceleration.scale(TICK);
    *position += velocity.scale(TICK);
}

/// Computed torque driven by a sinusoidal joint reference, with the reference's own rate and
/// acceleration fed forward.
fn computed_torque_tracks_a_moving_reference() {
    println!("== computed torque, tracking a moving reference ==");
    let body = two_link_pendulum(true);
    let controller = ComputedTorqueController::<2, f64>::from_natural_frequency(30.0, 1.0).unwrap();

    let amplitude = 0.4;
    let frequency = 2.0;
    let reference_at = |time: f64| {
        let phase = frequency * time;
        JointReference::new(
            Vector::new([amplitude * phase.sin(), -amplitude * phase.cos()]),
            Vector::new([
                amplitude * frequency * phase.cos(),
                amplitude * frequency * phase.sin(),
            ]),
            Vector::new([
                -amplitude * frequency * frequency * phase.sin(),
                amplitude * frequency * frequency * phase.cos(),
            ]),
        )
    };

    // Started exactly on the reference, so what is measured is tracking rather than catching up.
    let start = reference_at(0.0);
    let mut position = start.position;
    let mut velocity = start.velocity;
    let mut peak = 0.0_f64;

    let ticks = 2000;
    for tick in 0..ticks {
        let reference = reference_at(tick as f64 * TICK);
        let torque = controller
            .torque_at(&body, &position, &velocity, &reference)
            .unwrap();
        advance(&body, &mut position, &mut velocity, &torque);

        // The error against where the reference has moved to by the end of the tick.
        let arrived = reference_at((tick + 1) as f64 * TICK);
        for joint in 0..2 {
            peak = peak.max((position[joint] - arrived.position[joint]).abs());
        }
    }

    println!("  {:.1} s at {:.0} kHz", ticks as f64 * TICK, 1e-3 / TICK);
    println!("  peak tracking error: {peak:.3e} rad");
    assert!(peak < 1e-3, "peak tracking error {peak}");
    println!();
}

/// The whole point of driving the PD term through `H(q)`: the closed loop settles at the rate the
/// gains name, not at one the arm's inertia decides.
fn error_dynamics_are_the_gains_you_asked_for() {
    println!("== the error dynamics the gains ask for ==");
    let body = single_pendulum();
    let natural_frequency = 10.0;
    let controller =
        ComputedTorqueController::<1, f64>::from_natural_frequency(natural_frequency, 1.0).unwrap();

    let setpoint = 0.5;
    let reference = JointReference::at_rest(Vector::new([setpoint]));
    let mut position = Vector::zeros();
    let mut velocity = Vector::zeros();

    // Critically damped, so the error is `(1 + ωt)·e^(−ωt)` of where it started and first falls
    // under 2 % at ωt ≈ 5.834.
    let band = 0.02 * setpoint;
    let mut settled = f64::NAN;
    for tick in 0..4000 {
        let torque = controller
            .torque_at(&body, &position, &velocity, &reference)
            .unwrap();
        advance(&body, &mut position, &mut velocity, &torque);
        if (position[0] - setpoint).abs() > band {
            settled = (tick + 1) as f64 * TICK;
        }
    }

    let analytic = 5.834 / natural_frequency;
    println!("  ω = {natural_frequency:.0} rad/s, critically damped");
    println!("  settled to 2 %: {settled:.4} s, analytic {analytic:.4} s");
    assert!(
        (settled - analytic).abs() < 0.05 * analytic,
        "settling time {settled} against {analytic}"
    );
    println!();
}

/// An impedance is a spring, so a steady push moves it by exactly `external / stiffness` and holds
/// it there. A tracking controller would push back until the error went away instead.
fn impedance_yields_to_a_push() {
    println!("== an impedance yields to a push ==");
    let body = single_pendulum();
    let stiffness = 30.0;
    let controller =
        JointImpedanceController::new(Vector::new([stiffness]), Vector::new([9.0])).unwrap();

    let setpoint = 0.3;
    let reference = JointReference::at_rest(Vector::new([setpoint]));
    let external = -2.0;
    let mut position = Vector::zeros();
    let mut velocity = Vector::zeros();

    for _ in 0..20000 {
        let torque = controller
            .torque_at(&body, &position, &velocity, &reference)
            .unwrap();
        advance(
            &body,
            &mut position,
            &mut velocity,
            &Vector::new([torque[0] + external]),
        );
    }

    let deflection = position[0] - setpoint;
    let expected = external / stiffness;
    println!("  stiffness {stiffness:.0} N·m/rad, pushed with {external:.1} N·m");
    println!("  deflection: {deflection:.6} rad, expected {expected:.6} rad");
    assert!(
        (deflection - expected).abs() < 1e-4,
        "deflection {deflection} against {expected}"
    );
    println!();
}

/// A PD without gravity cancelled stalls where its spring balances the gravity torque. Cancelling
/// gravity removes that droop, and nothing else about the law changes.
fn gravity_compensation_removes_the_droop() {
    println!("== the droop gravity compensation removes ==");
    let body = single_pendulum();
    let position_gain = 40.0;
    let controller =
        JointPdController::new(Vector::new([position_gain]), Vector::new([12.0])).unwrap();

    let setpoint = 0.6;
    let reference = JointReference::at_rest(Vector::new([setpoint]));
    let settle = |controller: &JointPdController<1, f64>| {
        let mut position = Vector::zeros();
        let mut velocity = Vector::zeros();
        for _ in 0..8000 {
            let torque = controller
                .torque_at(&body, &position, &velocity, &reference)
                .unwrap();
            advance(&body, &mut position, &mut velocity, &torque);
        }
        setpoint - position[0]
    };

    let plain = settle(&controller);
    let compensated = settle(&controller.with_gravity_compensation(true));
    println!("  kp = {position_gain:.0} N·m/rad");
    println!("  steady-state error, gravity left in:   {plain:.3e} rad");
    println!("  steady-state error, gravity cancelled: {compensated:.3e} rad");

    assert!(
        compensated.abs() * 100.0 < plain.abs(),
        "compensated {compensated} against {plain}"
    );
    println!();
}

/// The other side of the same hardware: a joint that takes a commanded position and answers with
/// its own servo, advanced exactly rather than stepped toward.
fn position_servo_answers_a_command() {
    println!("== a position-controlled joint answers a command ==");
    let natural_frequency = 50.0;
    let mut joints = PositionServo::<7, f64>::uniform(natural_frequency, 1.0, TICK).unwrap();
    let commanded = Vector::new([0.4, -0.2, 0.6, 0.1, -0.5, 0.3, 0.0]);

    println!("  7 joints at ω = {natural_frequency:.0} rad/s, critically damped, 1 ms ticks");
    let mut ticks = 0;
    for target in [1, 10, 50, 200] {
        while ticks < target {
            let _ = joints.stepped(commanded);
            ticks += 1;
        }
        let elapsed = ticks as f64 * TICK;
        // Critically damped from rest: `q(t) = q_cmd·(1 − (1 + ωt)·e^(−ωt))`.
        let settled =
            1.0 - (1.0 + natural_frequency * elapsed) * (-natural_frequency * elapsed).exp();
        println!(
            "  after {ticks:3} ticks: joint 0 at {:8.5} rad, closed form {:8.5}",
            joints.positions()[0],
            commanded[0] * settled
        );
        for joint in 0..7 {
            let want = commanded[joint] * settled;
            assert!(
                (joints.positions()[joint] - want).abs() < 1e-12,
                "joint {joint} after {ticks} ticks: {} against {want}",
                joints.positions()[joint]
            );
        }
    }
    println!();
}
