//! Mixing tests: holding still, each single turn on its own, the round trip out to the rotors
//! and back, what happens when a rotor is asked for more than it has, and the layouts that are
//! refused.

use multicalc::dynamics::RigidBody;
use multicalc::error::PlantError;
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::plant::{MultirotorMixer, RotorSpin};
use multicalc::scalar::Numeric;
use multicalc::spatial::{SO3, SpatialInertia};
use proptest::prelude::*;

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

fn six_rotor_ring() -> MultirotorMixer<6, f64> {
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
    MultirotorMixer::<6, f64>::new(
        positions,
        spins,
        TORQUE_PER_THRUST,
        MINIMUM_THRUST,
        MAXIMUM_THRUST,
    )
    .unwrap()
}

#[test]
fn a_six_rotor_ring_also_works() {
    let mixer = six_rotor_ring();

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

const ALTERNATING: [RotorSpin; 4] = [
    RotorSpin::Clockwise,
    RotorSpin::CounterClockwise,
    RotorSpin::Clockwise,
    RotorSpin::CounterClockwise,
];

fn layout<T: Numeric>(
    planar_positions: [[f64; 2]; 4],
    torque_per_thrust: f64,
) -> Result<MultirotorMixer<4, T>, PlantError> {
    let positions: [Vector3D<T>; 4] = core::array::from_fn(|rotor| {
        let [x, y] = planar_positions[rotor];
        Vector::new([T::from_f64(x), T::from_f64(y), T::ZERO])
    });
    MultirotorMixer::<4, T>::new(
        positions,
        ALTERNATING,
        T::from_f64(torque_per_thrust),
        T::ZERO,
        T::from_f64(5.0),
    )
}

fn x_corners(half_span: f64) -> [[f64; 2]; 4] {
    [
        [half_span, half_span],
        [-half_span, half_span],
        [-half_span, -half_span],
        [half_span, -half_span],
    ]
}

#[test]
fn rank_check_accepts_the_same_layouts_at_both_precisions() {
    for half_span in [0.15, 15.0, 150.0, 1500.0] {
        assert!(
            layout::<f32>(x_corners(half_span), 0.016).is_ok(),
            "f32 half span {half_span}"
        );
        assert!(
            layout::<f64>(x_corners(half_span), 0.016).is_ok(),
            "f64 half span {half_span}"
        );
    }
    for torque_per_thrust in [1e-4, 1e-5, 1e-6] {
        assert!(
            layout::<f32>(x_corners(0.15), torque_per_thrust).is_ok(),
            "f32 ratio {torque_per_thrust:e}"
        );
        assert!(
            layout::<f64>(x_corners(0.15), torque_per_thrust).is_ok(),
            "f64 ratio {torque_per_thrust:e}"
        );
    }
}

#[test]
fn rank_check_rejects_collinear_rotors_at_both_precisions() {
    // All four rotors on the line y = 0.7·x.
    let tilted_line = [[0.1, 0.07], [0.37, 0.259], [-0.13, -0.091], [-0.29, -0.203]];
    let axis_line = [[0.1, 0.0], [0.2, 0.0], [-0.1, 0.0], [-0.2, 0.0]];
    for planar_positions in [tilted_line, axis_line] {
        assert_eq!(
            layout::<f32>(planar_positions, 0.016).err(),
            Some(PlantError::RotorLayoutNotIndependent)
        );
        assert_eq!(
            layout::<f64>(planar_positions, 0.016).err(),
            Some(PlantError::RotorLayoutNotIndependent)
        );
    }
}

// ----- Desaturation -----

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual {actual}, expected {expected}"
    );
}

fn largest_and_smallest<const ROTOR_COUNT: usize>(thrusts: Vector<ROTOR_COUNT, f64>) -> (f64, f64) {
    let largest = thrusts
        .as_array()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let smallest = thrusts
        .as_array()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    (largest, smallest)
}

#[test]
fn the_collective_gives_way_near_the_top() {
    let mixer = mixer();
    let commands = mixer.rotor_thrusts(19.0, Vector::new([0.2, 0.0, 0.0]));
    let shortfall = commands.shortfall();

    assert!(commands.saturated());
    assert!(shortfall[0] > 0.0);
    assert_eq!([shortfall[1], shortfall[2], shortfall[3]], [0.0; 3]);

    let produced = mixer.wrench(commands.thrusts());
    assert_close(produced.torque()[0], 0.2);
    assert_close(produced.torque()[1], 0.0);
    assert_close(produced.torque()[2], 0.0);
    assert_close(produced.force()[2], 19.0 - shortfall[0]);

    for thrust in commands.thrusts().as_array() {
        assert!((MINIMUM_THRUST..=MAXIMUM_THRUST).contains(thrust));
    }
    let (largest, _) = largest_and_smallest(commands.thrusts());
    assert_close(largest, MAXIMUM_THRUST);
}

#[test]
fn the_collective_rises_at_low_throttle() {
    let mixer = mixer();
    let commands = mixer.rotor_thrusts(0.5, Vector::new([0.2, 0.0, 0.0]));
    let shortfall = commands.shortfall();

    assert!(shortfall[0] < 0.0);
    assert_eq!([shortfall[1], shortfall[2], shortfall[3]], [0.0; 3]);

    let produced = mixer.wrench(commands.thrusts());
    assert_close(produced.torque()[0], 0.2);
    assert_close(produced.torque()[1], 0.0);
    assert_close(produced.torque()[2], 0.0);

    let (_, smallest) = largest_and_smallest(commands.thrusts());
    assert_close(smallest, MINIMUM_THRUST);
}

#[test]
fn yaw_gives_way_before_roll() {
    let mixer = mixer();
    let commands = mixer.rotor_thrusts(10.0, Vector::new([0.1, 0.0, 0.2]));
    let shortfall = commands.shortfall();

    assert_eq!([shortfall[0], shortfall[1], shortfall[2]], [0.0; 3]);
    assert!(shortfall[3] > 0.0 && shortfall[3] < 0.2);

    let produced = mixer.wrench(commands.thrusts());
    assert_close(produced.torque()[0], 0.1);
    assert_close(produced.torque()[1], 0.0);
    assert_close(produced.torque()[2], 0.2 - shortfall[3]);

    let (largest, smallest) = largest_and_smallest(commands.thrusts());
    assert_close(largest - smallest, MAXIMUM_THRUST - MINIMUM_THRUST);
}

#[test]
fn roll_and_pitch_scale_together_once_yaw_is_gone() {
    let mixer = mixer();

    let commands = mixer.rotor_thrusts(10.0, Vector::new([2.0, 0.0, 0.1]));
    let shortfall = commands.shortfall();
    assert_eq!(shortfall[3], 0.1);
    assert!(shortfall[0].abs() < 1e-12);
    let produced = mixer.wrench(commands.thrusts());
    assert_close(produced.torque()[2], 0.0);
    assert_close(produced.torque()[0], 2.0 - shortfall[1]);
    assert!(produced.torque()[0] > 0.0 && produced.torque()[0] < 2.0);
    for (thrust, expected) in commands
        .thrusts()
        .as_array()
        .iter()
        .zip([0.0, 5.0, 5.0, 0.0])
    {
        assert_close(*thrust, expected);
    }

    // One factor scales roll and pitch, so the torque direction stays the same.
    let diagonal = mixer.rotor_thrusts(10.0, Vector::new([1.5, 1.0, 0.0]));
    let produced = mixer.wrench(diagonal.thrusts());
    assert_close(produced.torque()[1] / produced.torque()[0], 1.0 / 1.5);
}

#[test]
fn a_command_that_fits_is_unchanged() {
    let mixer = mixer();
    let torque = Vector::new([0.1, -0.05, 0.02]);
    let commands = mixer.rotor_thrusts(8.0, torque);

    assert!(!commands.saturated());
    assert_eq!(commands.shortfall(), Vector::zeros());

    let produced = mixer.wrench(commands.thrusts());
    assert_close(produced.force()[2], 8.0);
    for axis in 0..3 {
        assert_close(produced.torque()[axis], torque[axis]);
    }
}

#[test]
fn a_command_that_is_not_finite_comes_back_not_finite() {
    let mixer = mixer();
    for commands in [
        mixer.rotor_thrusts(f64::NAN, no_turn()),
        mixer.rotor_thrusts(HOVER_THRUST, Vector::new([f64::INFINITY, 0.0, 0.0])),
    ] {
        assert!(!commands.thrusts().is_finite());
        assert!(commands.saturated());
    }
}

fn check_priorities<const ROTOR_COUNT: usize>(
    mixer: MultirotorMixer<ROTOR_COUNT, f64>,
    collective_thrust: f64,
    torque: Vector3D<f64>,
) -> Result<(), TestCaseError> {
    let commands = mixer.rotor_thrusts(collective_thrust, torque);
    let shortfall = commands.shortfall();
    for thrust in commands.thrusts().as_array() {
        prop_assert!((MINIMUM_THRUST..=MAXIMUM_THRUST).contains(thrust));
    }
    let produced = mixer.wrench(commands.thrusts());
    let produced = [
        produced.force()[2],
        produced.torque()[0],
        produced.torque()[1],
        produced.torque()[2],
    ];
    let wanted = [collective_thrust, torque[0], torque[1], torque[2]];
    for component in 0..4 {
        let target = wanted[component] - shortfall[component];
        prop_assert!((produced[component] - target).abs() <= 1e-12 * (1.0 + target.abs()));
    }
    for axis in 0..3 {
        let given_up = shortfall[axis + 1];
        prop_assert!(given_up * torque[axis] >= 0.0);
        prop_assert!(given_up.abs() <= torque[axis].abs());
    }
    if shortfall[1] != 0.0 || shortfall[2] != 0.0 {
        prop_assert_eq!(shortfall[3], torque[2]);
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn priorities_hold_for_random_commands(
        collective_thrust in -5.0f64..30.0,
        roll in -1.5f64..1.5,
        pitch in -1.5f64..1.5,
        yaw in -0.3f64..0.3,
    ) {
        let torque = Vector::new([roll, pitch, yaw]);
        check_priorities(mixer(), collective_thrust, torque)?;
        check_priorities(six_rotor_ring(), collective_thrust, torque)?;
        // With `ALTERNATING` spins, two collective shares are negative.
        let asymmetric = layout::<f64>(
            [[0.25, -0.1], [-0.1, 0.12], [0.2, 0.15], [-0.12, -0.2]],
            TORQUE_PER_THRUST,
        )
        .unwrap();
        check_priorities(asymmetric, collective_thrust, torque)?;
    }
}
