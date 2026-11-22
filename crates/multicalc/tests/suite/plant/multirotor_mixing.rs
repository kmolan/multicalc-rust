//! Mixing tests: holding still, each single turn on its own, the round trip out to the rotors
//! and back, what happens when a rotor is asked for more than it has, and the layouts that are
//! refused.

use multicalc::dynamics::RigidBody;
use multicalc::error::PlantError;
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::plant::{MultirotorMixer, RotorSpin};
use multicalc::scalar::Numeric;
use multicalc::spatial::{SO3, SpatialInertia};

const ARM_LENGTH: f64 = 0.15;
const TORQUE_PER_THRUST: f64 = 0.016;
const MINIMUM_THRUST: f64 = 0.0;
const MAXIMUM_THRUST: f64 = 5.0;

/// The four-rotor X machine every test below shares.
fn mixer() -> MultirotorMixer<4, f64> {
    MultirotorMixer::<4, f64>::quadrotor_x(
        ARM_LENGTH,
        TORQUE_PER_THRUST,
        MINIMUM_THRUST,
        MAXIMUM_THRUST,
    )
    .unwrap()
}

/// What a 0.8 kg machine has to push with to carry itself.
const HOVER_THRUST: f64 = 7.848;

fn no_turn<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ZERO, T::ZERO, T::ZERO])
}

#[test]
fn holding_still_shares_the_push_out_evenly() {
    let commands = mixer().rotor_thrusts(HOVER_THRUST, no_turn());

    let even_share = HOVER_THRUST / 4.0;
    for rotor in 0..4 {
        assert!((commands.thrusts()[rotor] - even_share).abs() < 1e-12);
    }
    assert!(!commands.saturated());
}

#[test]
fn a_turn_about_x_lifts_one_side() {
    let commands = mixer().rotor_thrusts(HOVER_THRUST, Vector::new([0.05, 0.0, 0.0]));
    let thrusts = commands.thrusts();

    // Rotors 1 and 2 are the two sitting at a positive y, so they are the ones that push harder.
    assert!(thrusts[1] > thrusts[0]);
    assert!(thrusts[1] > thrusts[3]);
    assert!(thrusts[2] > thrusts[0]);
    assert!(thrusts[2] > thrusts[3]);

    let total: f64 = (0..4).map(|rotor| thrusts[rotor]).sum();
    assert!((total - HOVER_THRUST).abs() < 1e-12);
}

#[test]
fn a_turn_about_z_splits_the_diagonals() {
    let commands = mixer().rotor_thrusts(HOVER_THRUST, Vector::new([0.0, 0.0, 0.02]));
    let thrusts = commands.thrusts();

    // Rotors 0 and 1 are the clockwise diagonal, and a turn about z is made by leaning on them.
    assert!(thrusts[0] > thrusts[2]);
    assert!(thrusts[0] > thrusts[3]);
    assert!(thrusts[1] > thrusts[2]);
    assert!(thrusts[1] > thrusts[3]);

    let total: f64 = (0..4).map(|rotor| thrusts[rotor]).sum();
    assert!((total - HOVER_THRUST).abs() < 1e-12);
}

#[test]
fn going_out_to_the_rotors_and_back_lands_where_it_started() {
    let mixer = mixer();
    let commands = [
        (HOVER_THRUST, no_turn()),
        (HOVER_THRUST, Vector::new([0.05, 0.0, 0.0])),
        (HOVER_THRUST, Vector::new([0.0, -0.04, 0.0])),
        (6.0, Vector::new([0.01, 0.02, 0.015])),
    ];

    for (collective_thrust, torque) in commands {
        let shared_out = mixer.rotor_thrusts(collective_thrust, torque);
        assert!(!shared_out.saturated());

        let produced = mixer.wrench(shared_out.thrusts());
        assert!(produced.force()[0].abs() < 1e-12);
        assert!(produced.force()[1].abs() < 1e-12);
        assert!((produced.force()[2] - collective_thrust).abs() < 1e-12);
        assert!((produced.torque() - torque).norm() < 1e-12);
    }
}

#[test]
fn the_layout_matches_the_arm_length_by_hand() {
    let allocation = mixer().allocation();
    let half = ARM_LENGTH / 2.0_f64.sqrt();

    let expected = [
        [1.0, 1.0, 1.0, 1.0],
        [-half, half, half, -half],
        [-half, half, -half, half],
        [
            TORQUE_PER_THRUST,
            TORQUE_PER_THRUST,
            -TORQUE_PER_THRUST,
            -TORQUE_PER_THRUST,
        ],
    ];

    for row in 0..4 {
        for rotor in 0..4 {
            assert!((allocation[(row, rotor)] - expected[row][rotor]).abs() < 1e-12);
        }
    }
}

#[test]
fn asking_for_more_than_a_rotor_has_clamps_and_says_so() {
    let mixer = mixer();

    let too_much = mixer.rotor_thrusts(30.0, no_turn());
    assert!(too_much.saturated());
    for rotor in 0..4 {
        assert!((too_much.thrusts()[rotor] - MAXIMUM_THRUST).abs() < 1e-12);
    }

    let below_nothing = mixer.rotor_thrusts(-1.0, no_turn());
    assert!(below_nothing.saturated());
    for rotor in 0..4 {
        assert!((below_nothing.thrusts()[rotor] - MINIMUM_THRUST).abs() < 1e-12);
    }
}

#[test]
fn a_six_rotor_ring_also_works() {
    let radius = 0.2;
    let positions: [Vector3D<f64>; 6] = core::array::from_fn(|rotor| {
        let angle = rotor as f64 * core::f64::consts::FRAC_PI_3;
        Vector::new([radius * angle.cos(), radius * angle.sin(), 0.0])
    });
    let spins: [RotorSpin; 6] = core::array::from_fn(|rotor| {
        if rotor % 2 == 0 {
            RotorSpin::Clockwise
        } else {
            RotorSpin::CounterClockwise
        }
    });
    let mixer = MultirotorMixer::<6, f64>::new(
        positions,
        spins,
        TORQUE_PER_THRUST,
        MINIMUM_THRUST,
        MAXIMUM_THRUST,
    )
    .unwrap();

    let commands = [
        (HOVER_THRUST, no_turn()),
        (HOVER_THRUST, Vector::new([0.05, 0.0, 0.0])),
    ];
    for (collective_thrust, torque) in commands {
        let shared_out = mixer.rotor_thrusts(collective_thrust, torque);
        assert!(!shared_out.saturated());

        let produced = mixer.wrench(shared_out.thrusts());
        assert!((produced.force()[2] - collective_thrust).abs() < 1e-10);
        assert!((produced.torque() - torque).norm() < 1e-10);
    }
}

#[test]
fn layouts_that_cannot_produce_every_turn_are_refused() {
    // Three rotors in a ring: one short of being able to set every push and turn on its own.
    let radius = 0.2;
    let positions: [Vector3D<f64>; 3] = core::array::from_fn(|rotor| {
        let angle = rotor as f64 * core::f64::consts::TAU / 3.0;
        Vector::new([radius * angle.cos(), radius * angle.sin(), 0.0])
    });
    let spins = [
        RotorSpin::Clockwise,
        RotorSpin::CounterClockwise,
        RotorSpin::Clockwise,
    ];
    assert_eq!(
        MultirotorMixer::<3, f64>::new(
            positions,
            spins,
            TORQUE_PER_THRUST,
            MINIMUM_THRUST,
            MAXIMUM_THRUST
        ),
        Err(PlantError::RotorLayoutNotIndependent)
    );

    // Four rotors all along the x axis: nothing can tip the body about x.
    let in_a_line = [
        Vector::new([0.1, 0.0, 0.0]),
        Vector::new([0.2, 0.0, 0.0]),
        Vector::new([-0.1, 0.0, 0.0]),
        Vector::new([-0.2, 0.0, 0.0]),
    ];
    let alternating = [
        RotorSpin::Clockwise,
        RotorSpin::CounterClockwise,
        RotorSpin::Clockwise,
        RotorSpin::CounterClockwise,
    ];
    assert_eq!(
        MultirotorMixer::<4, f64>::new(
            in_a_line,
            alternating,
            TORQUE_PER_THRUST,
            MINIMUM_THRUST,
            MAXIMUM_THRUST
        ),
        Err(PlantError::RotorLayoutNotIndependent)
    );

    assert_eq!(
        MultirotorMixer::<4, f64>::quadrotor_x(0.0, 0.016, 0.0, 5.0),
        Err(PlantError::NonPositiveArmLength)
    );
    assert_eq!(
        MultirotorMixer::<4, f64>::quadrotor_x(0.15, 0.0, 0.0, 5.0),
        Err(PlantError::NonPositiveTorqueRatio)
    );
    assert_eq!(
        MultirotorMixer::<4, f64>::quadrotor_x(0.15, 0.016, 5.0, 5.0),
        Err(PlantError::InvalidThrustLimits)
    );
    assert_eq!(
        MultirotorMixer::<4, f64>::quadrotor_x(f64::NAN, 0.016, 0.0, 5.0),
        Err(PlantError::NonFinite)
    );
}

#[test]
fn single_precision_round_trips_too() {
    let mixer = MultirotorMixer::<4, f32>::quadrotor_x(0.15, 0.016, 0.0, 5.0).unwrap();

    let collective_thrust = 7.848_f32;
    let shared_out = mixer.rotor_thrusts(collective_thrust, no_turn());
    assert!(!shared_out.saturated());

    let produced = mixer.wrench(shared_out.thrusts());
    assert!((produced.force()[2] - collective_thrust).abs() < 1e-4);
    assert!(produced.torque().norm() < 1e-4);
}

#[test]
fn a_mixer_drives_a_body() {
    let mass = 0.8;
    let gravity_strength = 9.81;
    let inertia = SpatialInertia::from_diagonal_inertia(
        mass,
        Vector::new([0.0, 0.0, 0.0]),
        Vector::new([0.005, 0.005, 0.009]),
    )
    .unwrap();
    let body = RigidBody::new(inertia, Vector::new([0.0, 0.0, -gravity_strength])).unwrap();

    // Asked for exactly what it takes to carry its own weight, the machine holds still.
    let weight = mass * gravity_strength;
    let commands = mixer().rotor_thrusts(weight, no_turn());
    let produced = mixer().wrench(commands.thrusts());

    let motion = body.accelerations(SO3::identity(), no_turn(), produced);
    assert!(motion.linear().norm() < 1e-12);
    assert!(motion.angular().norm() < 1e-12);
}
