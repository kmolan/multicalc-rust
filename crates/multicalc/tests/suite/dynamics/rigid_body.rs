//! Rigid-body tests: gravity alone, a spinning body resisting a turn, a body that does not balance
//! on its own origin, the quantities a torque-free tumble has to hold on to, and the round trip
//! through the thirteen numbers an integrator carries.

use core::f64::consts::FRAC_PI_2;

use multicalc::dynamics::{RigidBody, free_joint_from_state_vector, state_vector_from_free_joint};
use multicalc::error::DynamicsError;
use multicalc::linear_algebra::{Matrix, Vector, Vector3D};
use multicalc::ode::Rk4;
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::{FreeJointState, SE3, SO3, SpatialInertia, Twist, Wrench};

/// A body that balances on its own origin, so its origin and balance point move together.
fn balanced_body<T: Numeric>(mass: f64, resistance: [f64; 3], gravity: [f64; 3]) -> RigidBody<T> {
    let balance_point = Vector::new([T::ZERO, T::ZERO, T::ZERO]);
    let resistance_to_spinning = Vector::new(resistance.map(T::from_f64));
    let inertia = SpatialInertia::from_diagonal_inertia(
        T::from_f64(mass),
        balance_point,
        resistance_to_spinning,
    )
    .unwrap();
    RigidBody::new(inertia, Vector::new(gravity.map(T::from_f64))).unwrap()
}

fn zeros<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ZERO, T::ZERO, T::ZERO])
}

fn earth_gravity() -> [f64; 3] {
    [0.0, 0.0, -9.81]
}

fn no_gravity() -> [f64; 3] {
    [0.0, 0.0, 0.0]
}

#[test]
fn gravity_alone_accelerates_and_does_not_turn() {
    let body = balanced_body::<f64>(2.0, [1.0, 1.0, 1.0], earth_gravity());
    let level = SO3::identity();
    let not_turning = zeros();

    let motion = body.accelerations(level, not_turning, Wrench::zeros());

    assert!((motion.linear() - Vector::new([0.0, 0.0, -9.81])).norm() < 1e-12);
    assert!(motion.angular().norm() < 1e-12);
}

#[test]
fn a_push_along_body_z_lifts_a_tipped_body_sideways() {
    let body = balanced_body::<f64>(2.0, [1.0, 1.0, 1.0], earth_gravity());
    // Tipped a quarter turn about x, so what was the body's up axis now points along the world's
    // -y axis.
    let tipped = SO3::exp(Vector::new([FRAC_PI_2, 0.0, 0.0]));
    let not_turning = zeros();
    let push_along_body_up = Wrench::new(Vector::new([0.0, 0.0, 2.0]), zeros());

    let motion = body.accelerations(tipped, not_turning, push_along_body_up);

    assert!(motion.linear()[1] < 0.0);
    assert!(motion.linear()[0].abs() < 1e-12);
}

#[test]
fn a_turn_about_x_spins_the_body_up_about_x() {
    let body = balanced_body::<f64>(0.8, [0.005, 0.007, 0.009], no_gravity());
    let level = SO3::identity();
    let not_turning = zeros();
    let turn_about_x = Wrench::new(zeros(), Vector::new([0.01, 0.0, 0.0]));

    let motion = body.accelerations(level, not_turning, turn_about_x);

    assert!((motion.angular() - Vector::new([0.01 / 0.005, 0.0, 0.0])).norm() < 1e-12);
}

#[test]
fn a_spinning_body_resists_having_its_axis_moved() {
    let resistance = [0.005, 0.007, 0.009];
    let body = balanced_body::<f64>(0.8, resistance, no_gravity());
    let level = SO3::identity();
    let turning = Vector::new([1.0, 2.0, 3.0]);

    let motion = body.accelerations(level, turning, Wrench::zeros());

    // Worked out again here from the same numbers: -I⁻¹·(ω × (I·ω)).
    let rotational_inertia = Matrix::from_diagonal(resistance);
    let expected = (rotational_inertia.inverse().unwrap()
        * turning.cross(rotational_inertia * turning))
    .scale(-1.0);

    assert!((motion.angular() - expected).norm() < 1e-12);
    // With this term dropped the answer would be zero, so the test would not hold.
    assert!(motion.angular().norm() > 1e-3);
}

#[test]
fn a_body_that_does_not_balance_on_its_origin_swings_its_origin() {
    let balance_point = Vector::new([0.1, 0.0, 0.0]);
    let resistance_to_spinning = Vector::new([0.01, 0.02, 0.03]);
    let inertia =
        SpatialInertia::from_diagonal_inertia(1.0, balance_point, resistance_to_spinning).unwrap();
    let body = RigidBody::new(inertia, zeros()).unwrap();

    let level = SO3::identity();
    let turning = Vector::new([0.0, 0.0, 5.0]);
    let motion = body.accelerations(level, turning, Wrench::zeros());

    // The origin is swung outward: -(ω × (ω × c)) = [2.5, 0, 0].
    let expected = turning.cross(turning.cross(balance_point)).scale(-1.0);
    assert!((motion.linear() - expected).norm() < 1e-12);
    assert!((motion.linear() - Vector::new([2.5, 0.0, 0.0])).norm() < 1e-12);

    // The same body balancing on its own origin has nothing to swing.
    let centered = balanced_body::<f64>(1.0, [0.01, 0.02, 0.03], no_gravity());
    let still_centered = centered.accelerations(level, turning, Wrench::zeros());
    assert!(still_centered.linear().norm() < 1e-12);
}

#[test]
fn a_free_tumble_holds_its_energy_and_its_turning_momentum() {
    let resistance = [0.005, 0.007, 0.009];
    let body = balanced_body::<f64>(0.8, resistance, no_gravity());
    let rotational_inertia = Matrix::from_diagonal(resistance);

    let starting_turn = Vector::new([7.0, 3.0, 5.0]);
    let start = state_vector_from_free_joint(FreeJointState::new(
        SE3::identity(),
        Twist::new(zeros(), starting_turn),
    ));

    let step = 1e-4;
    let step_count = 50_000;
    let rate = |_time: f64, state: &Vector<13, f64>| body.state_derivative(state, Wrench::zeros());
    let after = Rk4::integrate(&rate, 0.0, &start, step, step_count, |_time, _state| {});

    // The spinning energy: ½·ωᵀ·I·ω.
    let spinning_energy = |turn: Vector3D<f64>| 0.5 * turn.dot(rotational_inertia * turn);
    let started_with = spinning_energy(starting_turn);
    let ended_with = spinning_energy(Vector::new([after[10], after[11], after[12]]));
    assert!((ended_with - started_with).abs() / started_with < 1e-9);

    // The turning momentum seen from the world, R·(I·ω), holds still even though the body's own
    // turn rate does not.
    let facing = free_joint_from_state_vector(&after)
        .unwrap()
        .pose()
        .rotation();
    let started_momentum = rotational_inertia * starting_turn;
    let ended_momentum =
        facing.act(rotational_inertia * Vector::new([after[10], after[11], after[12]]));
    assert!((ended_momentum - started_momentum).norm() < 1e-6);

    // The four orientation numbers drift, but only slightly.
    let facing_length =
        (after[3] * after[3] + after[4] * after[4] + after[5] * after[5] + after[6] * after[6])
            .sqrt();
    assert!((facing_length - 1.0).abs() < 1e-6);
}

#[test]
fn the_thirteen_numbers_round_trip() {
    let pose = SE3::from_parts(
        SO3::exp(Vector::new([0.3, -0.2, 0.7])),
        Vector::new([1.0, -2.0, 3.0]),
    );
    let velocity = Twist::new(Vector::new([0.5, 0.1, -0.2]), Vector::new([0.3, -0.4, 0.5]));
    let start = FreeJointState::new(pose, velocity);

    let packed = state_vector_from_free_joint(start);
    let unpacked = free_joint_from_state_vector(&packed).unwrap();

    let started_place = start.generalized_position();
    let ended_place = unpacked.generalized_position();
    for index in 0..7 {
        assert!((started_place[index] - ended_place[index]).abs() < 1e-15);
    }

    let started_motion = start.generalized_velocity();
    let ended_motion = unpacked.generalized_velocity();
    for index in 0..6 {
        assert!((started_motion[index] - ended_motion[index]).abs() < 1e-15);
    }
}

#[test]
fn a_state_with_no_direction_has_no_derivative() {
    let body = balanced_body::<f64>(0.8, [0.005, 0.007, 0.009], earth_gravity());
    let no_direction = Vector::new([
        1.0, -2.0, 3.0, // where it is
        0.0, 0.0, 0.0, 0.0, // which way it faces, naming no direction at all
        0.5, 0.1, -0.2, // how fast it is moving
        0.3, -0.4, 0.5, // how fast it is turning
    ]);

    let derivative = body.state_derivative(&no_direction, Wrench::zeros());

    assert!(derivative.norm() < 1e-15);
}

#[test]
fn a_body_with_a_flat_inertia_is_refused() {
    // Symmetric with a positive diagonal, so `SpatialInertia` accepts it, but it is not positive
    // definite so there is no way to invert it into an acceleration.
    let flat = Matrix::new([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let inertia = SpatialInertia::new(1.0, zeros(), flat).unwrap();
    assert_eq!(
        RigidBody::new(inertia, Vector::new(earth_gravity())),
        Err(DynamicsError::NonPositiveInertia)
    );

    let sound =
        SpatialInertia::from_diagonal_inertia(1.0, zeros(), Vector::new([1.0, 1.0, 1.0])).unwrap();
    assert_eq!(
        RigidBody::new(sound, Vector::new([0.0, 0.0, f64::NAN])),
        Err(DynamicsError::NonFinite)
    );
}

#[test]
fn single_precision_holds_the_same_identities() {
    let resistance = [0.005_f32, 0.007, 0.009];
    let body = balanced_body::<f32>(0.8, [0.005, 0.007, 0.009], no_gravity());
    let level = SO3::<f32>::identity();
    let turning = Vector::new([1.0_f32, 2.0, 3.0]);

    let motion = body.accelerations(level, turning, Wrench::zeros());

    let rotational_inertia = Matrix::from_diagonal(resistance);
    let expected = (rotational_inertia.inverse().unwrap()
        * turning.cross(rotational_inertia * turning))
    .scale(-1.0);

    assert!((motion.angular() - expected).norm() < 1e-4);
}

#[test]
fn the_derivative_can_be_differentiated() {
    let resistance = [0.005, 0.007, 0.009];
    let applied_turn_x = 0.01;

    // How the turning part's x component responds to the applied turn's x component.
    let body = balanced_body::<Dual<f64>>(0.8, resistance, no_gravity());
    let seeded = Wrench::new(
        zeros(),
        Vector::new([
            Dual::variable(applied_turn_x),
            Dual::constant(0.0),
            Dual::constant(0.0),
        ]),
    );
    let differentiated = body
        .accelerations(SO3::identity(), zeros(), seeded)
        .angular()[0]
        .deriv;

    // The same quantity, measured by nudging the applied turn at f64.
    let plain = balanced_body::<f64>(0.8, resistance, no_gravity());
    let turning_x = |applied: f64| {
        plain
            .accelerations(
                SO3::identity(),
                zeros(),
                Wrench::new(zeros(), Vector::new([applied, 0.0, 0.0])),
            )
            .angular()[0]
    };
    let nudge = 1e-6;
    let by_hand =
        (turning_x(applied_turn_x + nudge) - turning_x(applied_turn_x - nudge)) / (2.0 * nudge);

    assert!((differentiated - by_hand).abs() < 1e-6);
}

// How far apart two orientations are, in radians.
fn angle_between(a: SO3<f64>, b: SO3<f64>) -> f64 {
    (a.inverse() * b).log().norm()
}

// A steady body-axes push and turn, the same one the tumbling tests below share.
fn steady_push() -> Wrench<f64> {
    Wrench::new(
        Vector::new([0.0, 0.0, 8.0]),
        Vector::new([0.02, -0.01, 0.005]),
    )
}

#[test]
fn stepped_free_fall_matches_the_closed_form() {
    // Gravity never changes, and a half-way step carries an unchanging acceleration exactly, so
    // this has to land on the closed form and not merely near it.
    let body = balanced_body::<f64>(0.8, [0.005, 0.007, 0.009], earth_gravity());
    let mut state = FreeJointState::new(SE3::identity(), Twist::zeros());
    let step = 1e-3;
    for _ in 0..1000 {
        state = body.stepped(state, Wrench::zeros(), step);
    }

    let fall_time = 1.0;
    let expected_fall = 0.5 * 9.81 * fall_time * fall_time;
    assert!((state.pose().translation()[2] + expected_fall).abs() < 1e-12);
    assert!((state.velocity().linear()[2] + 9.81 * fall_time).abs() < 1e-12);
}

#[test]
fn stepped_holds_a_steady_spin() {
    // Equal resistance about every axis means nothing eats into the spin, so a body left alone
    // keeps turning at the rate it started with and simply piles up the turn.
    let body = balanced_body::<f64>(0.8, [0.006, 0.006, 0.006], no_gravity());
    let spinning = Twist::new(zeros(), Vector::new([0.0, 0.0, 3.0]));
    let mut state = FreeJointState::new(SE3::identity(), spinning);
    let step = 1e-3;
    for _ in 0..1000 {
        state = body.stepped(state, Wrench::zeros(), step);
    }

    assert!((state.velocity().angular() - Vector::new([0.0, 0.0, 3.0])).norm() < 1e-12);
    assert!((state.pose().rotation().log() - Vector::new([0.0, 0.0, 3.0])).norm() < 1e-9);
}

#[test]
fn stepped_keeps_the_orientation_a_true_rotation() {
    // Twenty seconds of hard tumbling under a steady push: the direction the body faces is
    // composed on rather than integrated, so it never leaves unit length to be scaled back.
    let body = balanced_body::<f64>(0.8, [0.005, 0.007, 0.009], earth_gravity());
    let tumbling = Twist::new(zeros(), Vector::new([7.0, 3.0, 5.0]));
    let mut state = FreeJointState::new(SE3::identity(), tumbling);
    let step = 1e-3;
    for _ in 0..20_000 {
        state = body.stepped(state, steady_push(), step);
    }

    let facing = state.pose().rotation().quaternion();
    assert!((facing.norm() - 1.0).abs() < 1e-12);
}

#[test]
fn stepped_converges_second_order() {
    // Halving the tick should cut the endpoint error by about four, in where the body ends up as
    // well as in which way it faces.
    let body = balanced_body::<f64>(0.8, [0.005, 0.007, 0.009], earth_gravity());
    let tumbling = Twist::new(zeros(), Vector::new([2.0, -1.0, 3.0]));
    let start = FreeJointState::new(SE3::identity(), tumbling);
    let final_time = 0.5;

    let after = |steps: usize| {
        let step = final_time / steps as f64;
        let mut state = start;
        for _ in 0..steps {
            state = body.stepped(state, steady_push(), step);
        }
        state
    };

    let reference = after(500_000);
    let endpoint_error = |steps: usize| {
        let state = after(steps);
        angle_between(state.pose().rotation(), reference.pose().rotation())
            + (state.pose().translation() - reference.pose().translation()).norm()
    };
    let ratio = endpoint_error(500) / endpoint_error(1000);
    assert!((3.2..=4.8).contains(&ratio), "convergence ratio {ratio}");
}

#[test]
fn stepped_agrees_with_the_thirteen_number_path() {
    // The same run two ways: the state carried whole with the direction on the manifold, and the
    // thirteen loose numbers handed to RK4. A stray half, a composition from the wrong side, or a
    // world-for-body frame slip shows up here even though each path is self-consistent on its own.
    let body = balanced_body::<f64>(0.8, [0.005, 0.007, 0.009], earth_gravity());
    let tumbling = Twist::new(zeros(), Vector::new([2.0, -1.0, 3.0]));
    let start = FreeJointState::new(SE3::identity(), tumbling);
    let step = 1e-4;
    let steps = 200;

    let mut stepped = start;
    for _ in 0..steps {
        stepped = body.stepped(stepped, steady_push(), step);
    }

    let start_numbers = state_vector_from_free_joint(start);
    let rate = |_time: f64, state: &Vector<13, f64>| body.state_derivative(state, steady_push());
    let by_rk4 = Rk4::integrate(&rate, 0.0, &start_numbers, step, steps, |_, _| {});
    let by_rk4 = free_joint_from_state_vector(&by_rk4).unwrap();

    assert!(
        (stepped.pose().translation() - by_rk4.pose().translation()).norm() < 1e-6,
        "the two paths put the body in different places"
    );
    assert!(
        angle_between(stepped.pose().rotation(), by_rk4.pose().rotation()) < 1e-6,
        "the two paths point the body in different directions"
    );
}
