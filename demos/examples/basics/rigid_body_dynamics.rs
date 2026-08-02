//! Rigid-body dynamics and rotor mixing: a body tumbling with nothing acting on it, a body
//! dropped from rest, a four-rotor machine holding a hover, the moment those rotors take to catch
//! up to what they are asked for, a roll command split across its rotors, and a command bigger
//! than the rotors can give.
//!
//! Run with: `cargo run -p multicalc-demos --example rigid_body_dynamics`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::dynamics::{RigidBody, free_joint_from_state_vector, state_vector_from_free_joint};
use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::ode::Rk4;
use multicalc::plant::{MultirotorMixer, RotorLag};
use multicalc::spatial::{FreeJointState, SE3, SO3, SpatialInertia, Twist, Wrench};

const GRAVITY_STRENGTH: f64 = 9.81;
const MASS: f64 = 0.8;
const ARM_LENGTH: f64 = 0.15;
const TORQUE_PER_THRUST: f64 = 0.016;
const MINIMUM_THRUST: f64 = 0.0;
const MAXIMUM_THRUST: f64 = 5.0;
const LAG_TIME: f64 = 0.02;
const TICK: f64 = 0.001;

fn main() {
    free_tumble();
    free_fall();
    hover();
    spool_up();
    roll_command();
    over_the_limit();
}

/// A small flying machine, harder to spin about one axis than the others.
fn body(gravity: Vector3D<f64>) -> RigidBody<f64> {
    let balance_point = Vector::new([0.0, 0.0, 0.0]);
    let resistance_to_spinning = Vector::new([0.005, 0.007, 0.009]);
    let inertia =
        SpatialInertia::from_diagonal_inertia(MASS, balance_point, resistance_to_spinning).unwrap();
    RigidBody::new(inertia, gravity).unwrap()
}

fn mixer() -> MultirotorMixer<4, f64> {
    MultirotorMixer::<4, f64>::quadrotor_x(
        ARM_LENGTH,
        TORQUE_PER_THRUST,
        MINIMUM_THRUST,
        MAXIMUM_THRUST,
    )
    .unwrap()
}

fn no_turn() -> Vector3D<f64> {
    Vector::new([0.0, 0.0, 0.0])
}

// ----- a body left to tumble with nothing acting on it -----

fn free_tumble() {
    let no_gravity = Vector::new([0.0, 0.0, 0.0]);
    let machine = body(no_gravity);
    let resistance = Matrix::from_diagonal([0.005, 0.007, 0.009]);

    let starting_turn = Vector::new([7.0, 3.0, 5.0]);
    let start = state_vector_from_free_joint(FreeJointState::new(
        SE3::identity(),
        Twist::new(Vector::new([0.0, 0.0, 0.0]), starting_turn),
    ));

    let step = 1e-4;
    let step_count = 50_000;
    let rate =
        |_time: f64, state: &Vector<13, f64>| machine.state_derivative(state, Wrench::zeros());
    let after = Rk4::integrate(&rate, 0.0, &start, step, step_count, |_time, _state| {});

    let spinning_energy = |turn: Vector3D<f64>| 0.5 * turn.dot(resistance * turn);
    let ending_turn = Vector::new([after[10], after[11], after[12]]);
    let started_with = spinning_energy(starting_turn);
    let ended_with = spinning_energy(ending_turn);
    let energy_drift = (ended_with - started_with).abs() / started_with;

    // Seen from the world the turning momentum holds still, even though the body's own turn rate
    // wanders the whole time.
    let facing = free_joint_from_state_vector(&after)
        .unwrap()
        .pose()
        .rotation();
    let momentum_drift = (facing.act(resistance * ending_turn) - resistance * starting_turn).norm();

    let facing_length =
        (after[3] * after[3] + after[4] * after[4] + after[5] * after[5] + after[6] * after[6])
            .sqrt();

    println!("A body left to tumble for 5 s with nothing acting on it");
    println!(
        "  turn rate started at [{:.3}, {:.3}, {:.3}] rad/s",
        starting_turn[0], starting_turn[1], starting_turn[2]
    );
    println!(
        "  turn rate ended at   [{:.3}, {:.3}, {:.3}] rad/s",
        ending_turn[0], ending_turn[1], ending_turn[2]
    );
    println!(
        "  spinning energy {started_with:.9} J -> {ended_with:.9} J  (drift {energy_drift:.2e})"
    );
    println!("  turning momentum drift {momentum_drift:.2e}");
    println!(
        "  the four orientation numbers are {:.2e} away from unit length",
        (facing_length - 1.0).abs()
    );

    assert!(energy_drift < 1e-9, "a free tumble must hold its energy");
    assert!(
        momentum_drift < 1e-6,
        "a free tumble must hold its turning momentum"
    );
    assert!(
        (facing_length - 1.0).abs() < 1e-6,
        "the orientation numbers must stay close to unit length"
    );
}

// ----- a body dropped from rest -----

fn free_fall() {
    let machine = body(Vector::new([0.0, 0.0, -GRAVITY_STRENGTH]));

    let at_rest = FreeJointState::new(SE3::identity(), Twist::zeros());
    let start = state_vector_from_free_joint(at_rest);
    let rate =
        |_time: f64, state: &Vector<13, f64>| machine.state_derivative(state, Wrench::zeros());

    let step = 1e-3;
    let step_count = 1000;
    let fall_time = 1.0;
    let after = Rk4::integrate(&rate, 0.0, &start, step, step_count, |_time, _state| {});

    let expected_fall = 0.5 * GRAVITY_STRENGTH * fall_time * fall_time;
    let expected_speed = GRAVITY_STRENGTH * fall_time;

    println!("\nA body dropped from rest and left alone for {fall_time} s");
    println!(
        "  it is {:.6} m below where it started (expected {expected_fall:.6})",
        -after[2]
    );
    println!(
        "  heading down at {:.6} m/s (expected {expected_speed:.6})",
        -after[9]
    );

    assert!(
        (after[2] + expected_fall).abs() < 1e-9,
        "it must fall 4.905 m"
    );
    assert!(
        (after[9] + expected_speed).abs() < 1e-9,
        "it must be doing 9.81 m/s"
    );
}

// ----- four rotors carrying the machine's own weight -----

fn hover() {
    let machine = body(Vector::new([0.0, 0.0, -GRAVITY_STRENGTH]));
    let mixer = mixer();

    let weight = MASS * GRAVITY_STRENGTH;
    let commands = mixer.rotor_thrusts(weight, no_turn());
    let thrusts = commands.thrusts();
    let even_share = weight / 4.0;

    println!("\nFour rotors asked to carry a {MASS} kg machine ({weight:.3} N)");
    println!(
        "  thrusts [{:.6}, {:.6}, {:.6}, {:.6}] N",
        thrusts[0], thrusts[1], thrusts[2], thrusts[3]
    );
    println!(
        "  any rotor asked for more than it has? {}",
        commands.saturated()
    );

    for rotor in 0..4 {
        assert!(
            (thrusts[rotor] - even_share).abs() < 1e-12,
            "the push should be shared out evenly"
        );
    }
    assert!(!commands.saturated(), "no rotor should be stretched");

    // Held there for a second, it has not moved.
    let wrench = mixer.wrench(thrusts);
    let start = state_vector_from_free_joint(FreeJointState::new(SE3::identity(), Twist::zeros()));
    let rate = |_time: f64, state: &Vector<13, f64>| machine.state_derivative(state, wrench);
    let after = Rk4::integrate(&rate, 0.0, &start, 1e-3, 1000, |_time, _state| {});

    println!(
        "  after 1 s of holding that, it has moved [{:.2e}, {:.2e}, {:.2e}] m",
        after[0], after[1], after[2]
    );
    for axis in 0..3 {
        assert!(after[axis].abs() < 1e-9, "a hovering machine must stay put");
    }
}

// ----- the moment the rotors take to catch up -----

fn spool_up() {
    let mixer = mixer();
    let mut rotors = RotorLag::<4, f64>::new(LAG_TIME, TICK).unwrap();

    let weight = MASS * GRAVITY_STRENGTH;
    let asked_for = mixer.rotor_thrusts(weight, no_turn()).thrusts();
    let even_share = weight / 4.0;

    println!(
        "\nThe same hover command, but the rotors take {LAG_TIME} s to catch up ({TICK} s ticks)"
    );
    println!("  asked for {even_share:.6} N from each rotor, from a standstill");

    // Follow one rotor for three lag times, reporting where it has got to at each one.
    let ticks_in_one_lag_time = (LAG_TIME / TICK) as usize;
    for lag_time_number in 1..=3 {
        for _ in 0..ticks_in_one_lag_time {
            let _ = rotors.stepped(asked_for);
        }
        let reached = rotors.thrusts()[0];
        println!(
            "  after {:.2} s it is giving {:.6} N — {:.1}% of what it was asked for",
            f64::from(lag_time_number) * LAG_TIME,
            reached,
            100.0 * reached / even_share
        );
    }

    // One lag time closes a little under two thirds of the gap, every time.
    let mut fresh = RotorLag::<4, f64>::new(LAG_TIME, TICK).unwrap();
    for _ in 0..ticks_in_one_lag_time {
        let _ = fresh.stepped(asked_for);
    }
    let closed_fraction = fresh.thrusts()[0] / even_share;
    let closed_in_one_lag_time = 1.0 - (-1.0_f64).exp();
    assert!(
        (closed_fraction - closed_in_one_lag_time).abs() < 1e-12,
        "one lag time should close a little under two thirds of the gap"
    );

    // What the body actually feels on the first tick is a long way short of its own weight, which
    // is why a machine asked to hover from a standstill drops before it climbs.
    let mut from_rest = RotorLag::<4, f64>::new(LAG_TIME, TICK).unwrap();
    let first_tick = from_rest.stepped(asked_for);
    let felt = mixer.wrench(first_tick);
    println!(
        "  on the first tick the body feels {:.6} N, not the {:.6} N it asked for",
        felt.force()[2],
        weight
    );
    assert!(
        felt.force()[2] < weight,
        "the first tick cannot deliver the whole command"
    );

    // Held there, they settle on exactly what was asked for.
    let long_enough_to_settle = 2000;
    for _ in 0..long_enough_to_settle {
        let _ = from_rest.stepped(asked_for);
    }
    println!(
        "  held there, they settle on {:.6} N each",
        from_rest.thrusts()[0]
    );
    for rotor in 0..4 {
        assert!(
            (from_rest.thrusts()[rotor] - even_share).abs() < 1e-12,
            "the rotors should settle on exactly what was asked for"
        );
    }
}

// ----- a roll command, split across the rotors -----

fn roll_command() {
    let mixer = mixer();

    let weight = MASS * GRAVITY_STRENGTH;
    let wanted_turn = Vector::new([0.05, 0.0, 0.0]);
    let commands = mixer.rotor_thrusts(weight, wanted_turn);
    let thrusts = commands.thrusts();
    let total: f64 = (0..4).map(|rotor| thrusts[rotor]).sum();

    println!("\nThe same machine asked to roll, turning about its own x axis at 0.05 N·m");
    println!(
        "  thrusts [{:.6}, {:.6}, {:.6}, {:.6}] N",
        thrusts[0], thrusts[1], thrusts[2], thrusts[3]
    );
    println!("  the two rotors on the +y side push harder, and the total is still {total:.6} N");

    assert!(thrusts[1] > thrusts[0] && thrusts[1] > thrusts[3]);
    assert!(thrusts[2] > thrusts[0] && thrusts[2] > thrusts[3]);
    assert!(
        (total - weight).abs() < 1e-12,
        "rolling should not change the total push"
    );

    // Reading the thrusts back gives exactly the push and turn that was asked for.
    let produced = mixer.wrench(thrusts);
    println!(
        "  reading those back gives a push of {:.6} N and a turn of [{:.6}, {:.6}, {:.6}] N·m",
        produced.force()[2],
        produced.torque()[0],
        produced.torque()[1],
        produced.torque()[2]
    );
    assert!((produced.force()[2] - weight).abs() < 1e-12);
    assert!((produced.torque() - wanted_turn).norm() < 1e-12);

    // A level, still body given that wrench starts rolling and nothing else.
    let machine = body(Vector::new([0.0, 0.0, -GRAVITY_STRENGTH]));
    let motion = machine.accelerations(SO3::identity(), no_turn(), produced);
    println!(
        "  a level, still body given that starts turning at [{:.3}, {:.3}, {:.3}] rad/s²",
        motion.angular()[0],
        motion.angular()[1],
        motion.angular()[2]
    );
    assert!(motion.angular()[0] > 0.0, "it should start rolling");
}

// ----- more than the rotors can give -----

fn over_the_limit() {
    let beyond_reach = 30.0;
    let commands = mixer().rotor_thrusts(beyond_reach, no_turn());
    let thrusts = commands.thrusts();

    println!("\nThe same machine asked for {beyond_reach} N, far more than four 5 N rotors have");
    println!(
        "  thrusts [{:.6}, {:.6}, {:.6}, {:.6}] N",
        thrusts[0], thrusts[1], thrusts[2], thrusts[3]
    );
    println!(
        "  any rotor asked for more than it has? {}",
        commands.saturated()
    );
    println!(
        "  so the body does not get the push that was asked for — an outer loop has to notice"
    );

    assert!(
        commands.saturated(),
        "the rotors must report being maxed out"
    );
    for rotor in 0..4 {
        assert!(
            (thrusts[rotor] - MAXIMUM_THRUST).abs() < 1e-12,
            "every rotor should sit at its limit"
        );
    }
}
