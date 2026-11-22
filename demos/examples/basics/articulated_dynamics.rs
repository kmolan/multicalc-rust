//! Inverse and forward dynamics for a jointed robot: a two-link pendulum held against gravity,
//! the same arm accelerated, what armature and joint friction each add, a torque turned back into
//! the acceleration that produced it, a free swing that holds its energy, and the Menagerie
//! Franka's holding torques read straight from its model file.
//!
//! Run with: `cargo run -p multicalc-demos --example articulated_dynamics`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::ode::Rk4;
use multicalc::spatial::{SE3, SO3, SpatialInertia};

const GRAVITY: f64 = -9.81;
const LINK_LENGTH: f64 = 1.0;
const LINK_MASS: f64 = 1.5;
const CENTER_OF_MASS: f64 = 0.5;
const LINK_INERTIA: f64 = 0.02;
const ARMATURE: [f64; 2] = [0.05, 0.03];
const DAMPING: [f64; 2] = [0.7, 0.4];
const FRICTION_LOSS: [f64; 2] = [0.2, 0.1];
const TICK: f64 = 1e-4;
const TICKS_PER_SECOND: usize = 10_000;

fn main() {
    holding_torque();
    accelerating();
    what_the_rotor_and_friction_add();
    both_directions();
    free_swing_holds_its_energy();
    franka_from_its_model_file();
}

/// Two hinges about `+y` on a unit link, each carrying a 1.5 kg arm balancing half a metre out.
///
/// `loaded` chooses whether the joints carry their armature, damping and friction loss.
fn two_link_pendulum(loaded: bool) -> ArticulatedBody<2, 2, f64> {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let link = SE3::from_parts(SO3::identity(), Vector::new([LINK_LENGTH, 0.0, 0.0]));
    let origins = [SE3::identity(), link];

    let mut tree = KinematicTree::<2, 2, f64>::new();
    for index in 0..2 {
        let (armature, damping, friction_loss) = if loaded {
            (ARMATURE[index], DAMPING[index], FRICTION_LOSS[index])
        } else {
            (0.0, 0.0, 0.0)
        };
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(
            Joint::revolute(axis, origins[index])
                .with_armature(armature)
                .with_damping(damping)
                .with_friction_loss(friction_loss),
            parent,
        )
        .unwrap();
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

/// What each joint holds against gravity with the arm straight out.
fn holding_torque() {
    println!("== holding the arm up ==");
    let body = two_link_pendulum(true);
    let straight_out = Vector::new([0.0, 0.0]);
    let torque = body.gravity_torque_at(&straight_out).unwrap();

    println!("  shoulder: {:8.4} N·m", torque[0]);
    println!("  elbow:    {:8.4} N·m", torque[1]);

    // The shoulder carries both links, the elbow only the far one.
    assert!(torque[0].abs() > torque[1].abs());
    println!();
}

/// The torque an acceleration costs on top of holding still is exactly `H(q)·q̈`.
fn accelerating() {
    println!("== what an acceleration costs ==");
    let body = two_link_pendulum(true);
    let folded = Vector::new([0.3, -0.9]);
    let still = Vector::zeros();
    let wanted = Vector::new([1.0, 0.0]);

    let holding = body.inverse_dynamics_at(&folded, &still, &still).unwrap();
    let driving = body.inverse_dynamics_at(&folded, &still, &wanted).unwrap();
    let inertial = body.joint_space_inertia_at(&folded).unwrap() * wanted;

    println!(
        "  holding still:      [{:8.4}, {:8.4}] N·m",
        holding[0], holding[1]
    );
    println!(
        "  accelerating:       [{:8.4}, {:8.4}] N·m",
        driving[0], driving[1]
    );
    println!(
        "  H(q)·q̈:             [{:8.4}, {:8.4}] N·m",
        inertial[0], inertial[1]
    );
    println!("  recursive Newton-Euler and composite-rigid-body, two readings of one model.");

    for joint in 0..2 {
        assert!((driving[joint] - holding[joint] - inertial[joint]).abs() < 1e-10);
    }
    println!();
}

/// Armature scales the acceleration term; damping and Coulomb friction ride on the rate.
fn what_the_rotor_and_friction_add() {
    println!("== what the rotor and the friction add ==");
    let loaded = two_link_pendulum(true);
    let plain = two_link_pendulum(false);
    let folded = Vector::new([0.3, -0.9]);
    let still = Vector::zeros();
    let wanted = Vector::new([1.0, 0.5]);

    let rotor = loaded
        .inverse_dynamics_at(&folded, &still, &wanted)
        .unwrap()
        - plain.inverse_dynamics_at(&folded, &still, &wanted).unwrap();
    println!(
        "  armature ⊙ q̈:       [{:8.4}, {:8.4}] N·m",
        rotor[0], rotor[1]
    );
    for joint in 0..2 {
        assert!((rotor[joint] - ARMATURE[joint] * wanted[joint]).abs() < 1e-12);
    }

    let moving = Vector::new([1.4, -0.6]);
    let forwards = loaded.bias_torque_at(&folded, &moving).unwrap()
        - plain.bias_torque_at(&folded, &moving).unwrap();
    println!(
        "  moving one way:     [{:8.4}, {:8.4}] N·m",
        forwards[0], forwards[1]
    );

    let backwards = loaded.bias_torque_at(&folded, &-moving).unwrap()
        - plain.bias_torque_at(&folded, &-moving).unwrap();
    println!(
        "  and the other:      [{:8.4}, {:8.4}] N·m",
        backwards[0], backwards[1]
    );
    for joint in 0..2 {
        assert!((forwards[joint] + backwards[joint]).abs() < 1e-12);
    }

    // A standing joint carries no friction loss at all.
    let standing = loaded.bias_torque_at(&folded, &still).unwrap()
        - plain.bias_torque_at(&folded, &still).unwrap();
    println!(
        "  standing still:     [{:8.4}, {:8.4}] N·m",
        standing[0], standing[1]
    );
    for joint in 0..2 {
        assert!(standing[joint].abs() < 1e-12);
    }
    println!();
}

/// The torque that produces an acceleration, put back through the other direction.
fn both_directions() {
    println!("== both directions of one question ==");
    let body = two_link_pendulum(true);
    let folded = Vector::new([0.3, -0.9]);
    let moving = Vector::new([1.1, -0.7]);
    let wanted = Vector::new([0.8, 1.4]);

    let torque = body.inverse_dynamics_at(&folded, &moving, &wanted).unwrap();
    let produced = body.forward_dynamics_at(&folded, &moving, &torque).unwrap();

    println!(
        "  wanted:             [{:8.4}, {:8.4}] rad/s²",
        wanted[0], wanted[1]
    );
    println!(
        "  torque:             [{:8.4}, {:8.4}] N·m",
        torque[0], torque[1]
    );
    println!(
        "  produced:           [{:8.4}, {:8.4}] rad/s²",
        produced[0], produced[1]
    );
    println!("  the articulated-body algorithm is O(n) and never forms H; the same answer is");
    println!("  assemblable from joint_space_inertia plus a Cholesky solve, at O(n³).");

    for joint in 0..2 {
        assert!((produced[joint] - wanted[joint]).abs() < 1e-10);
    }
    println!();
}

/// `[q; q̇]` as the four numbers an integrator carries.
fn swing_rate(body: &ArticulatedBody<2, 2, f64>, state: &Vector<4, f64>) -> Vector<4, f64> {
    let position = Vector::new([state[0], state[1]]);
    let velocity = Vector::new([state[2], state[3]]);
    let acceleration = body
        .forward_dynamics_at(&position, &velocity, &Vector::zeros())
        .unwrap();
    Vector::new([velocity[0], velocity[1], acceleration[0], acceleration[1]])
}

fn total_energy(body: &ArticulatedBody<2, 2, f64>, state: &Vector<4, f64>) -> f64 {
    let solved = body
        .tree()
        .forward_kinematics(&Vector::new([state[0], state[1]]))
        .unwrap();
    let velocity = Vector::new([state[2], state[3]]);
    body.kinetic_energy(&solved, &velocity).unwrap() + body.potential_energy(&solved).unwrap()
}

/// A frictionless swing holds its energy; friction takes it out.
fn free_swing_holds_its_energy() {
    println!("== a free swing ==");
    let start = Vector::new([1.2, -0.5, 0.0, 0.0]);

    let plain = two_link_pendulum(false);
    let before = total_energy(&plain, &start);
    let after = Rk4::integrate(
        &|_time: f64, state: &Vector<4, f64>| swing_rate(&plain, state),
        0.0,
        &start,
        TICK,
        TICKS_PER_SECOND,
        |_time, _state| {},
    );
    let held = total_energy(&plain, &after);
    let drift = (held - before).abs() / before.abs();

    println!("  frictionless, released from rest:");
    println!("    energy at 0 s:    {before:8.6} J");
    println!("    energy at 1 s:    {held:8.6} J");
    println!("    relative drift:   {drift:.3e}");
    assert!(drift < 1e-6);

    let loaded = two_link_pendulum(true);
    let before_loaded = total_energy(&loaded, &start);
    let after_loaded = Rk4::integrate(
        &|_time: f64, state: &Vector<4, f64>| swing_rate(&loaded, state),
        0.0,
        &start,
        TICK,
        TICKS_PER_SECOND,
        |_time, _state| {},
    );
    let left = total_energy(&loaded, &after_loaded);

    println!("  with damping and friction loss:");
    println!("    energy at 0 s:    {before_loaded:8.6} J");
    println!("    energy at 1 s:    {left:8.6} J");
    assert!(left < before_loaded);
    println!();
}

/// The Menagerie Franka Panda, straight from its MJCF file.
fn franka_from_its_model_file() {
    println!("== the Franka Panda, from its model file ==");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../third_party/menagerie/franka_emika_panda/panda.xml");
    let model = multicalc_robot_model::mjcf::load_path(&path).unwrap();
    let panda = model
        .articulated_body::<16, 16>(Vector::new([0.0, 0.0, GRAVITY]))
        .unwrap();

    println!(
        "  {} body slots, {} velocity numbers — a weld takes a slot and leaves its entry zero",
        panda.len(),
        panda.velocity_len()
    );
    let holding = panda.gravity_torque_at(&Vector::zeros()).unwrap();

    // Slot 0 is the welded base, so the seven arm hinges are slots 1 to 7. At the zero
    // configuration the odd ones turn about an axis lined up with gravity — no moment arm — so
    // they hold nothing.
    for joint in 1..8 {
        println!("  joint {joint}: {:9.4} N·m", holding[joint]);
    }

    let mut heaviest = 0.0_f64;
    for joint in 0..panda.velocity_len() {
        assert!(holding[joint].is_finite());
        heaviest = heaviest.max(holding[joint].abs());
    }
    // A horizontal seven-link arm cannot hold itself up for free.
    assert!(heaviest > 1.0);
    println!("  heaviest holding torque: {heaviest:.4} N·m");
}
