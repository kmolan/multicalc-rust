//! Momentum-observer tests: the exact discrete law on an arm held still, a free and a driven swing
//! against an RK4 plant, `reset`, `f32`, and the refusals.

use multicalc::dynamics::{ArticulatedBody, MomentumObserver};
use multicalc::error::DynamicsError;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::ode::Rk4;
use multicalc::scalar::Numeric;
use multicalc::spatial::{SE3, SO3, SpatialInertia};

const OBSERVER_GAIN: f64 = 50.0;
const TIMESTEP: f64 = 1e-3;
const PLANT_SUBSTEPS: usize = 10;
const EXTERNAL_TORQUE: [f64; 2] = [0.8, -0.5];

/// Two revolute joints about `+y`, a unit link between them, both links 1.5 kg balancing 0.5 m out.
fn double_pendulum_with<T: Numeric>(
    armature: [f64; 2],
    damping: [f64; 2],
    friction_loss: [f64; 2],
) -> ArticulatedBody<2, 2, T> {
    let axis = Vector::new([T::ZERO, T::ONE, T::ZERO]);
    let link = SE3::from_parts(SO3::identity(), Vector::new([T::ONE, T::ZERO, T::ZERO]));
    let origins = [SE3::identity(), link];

    let mut tree = KinematicTree::<2, 2, T>::new();
    for index in 0..2 {
        let joint = Joint::revolute(axis, origins[index])
            .with_armature(T::from_f64(armature[index]))
            .with_damping(T::from_f64(damping[index]))
            .with_friction_loss(T::from_f64(friction_loss[index]));
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(joint, parent).unwrap();
    }

    let inertia = SpatialInertia::new(
        T::from_f64(1.5),
        Vector::new([T::HALF, T::ZERO, T::ZERO]),
        Matrix::from_diagonal([T::from_f64(0.02), T::from_f64(0.02), T::from_f64(0.02)]),
    )
    .unwrap();
    let gravity = Vector::new([T::ZERO, T::ZERO, T::from_f64(-9.81)]);
    ArticulatedBody::new(tree, &[Some(inertia), Some(inertia)], gravity).unwrap()
}

/// Armature and viscous damping, no Coulomb friction: smooth, so the RK4 plant stays fourth order.
fn viscous_double_pendulum<T: Numeric>() -> ArticulatedBody<2, 2, T> {
    double_pendulum_with([0.05, 0.03], [0.7, 0.4], [0.0, 0.0])
}

fn observer<T: Numeric>() -> MomentumObserver<2, T> {
    MomentumObserver::new(
        Vector::new([T::from_f64(OBSERVER_GAIN); 2]),
        T::from_f64(TIMESTEP),
    )
    .unwrap()
}

/// `1 − (1 + K·dt)^−k`: the share of a step in `τ_ext` that `r` shows after `k` ticks.
fn step_response(tick: usize) -> f64 {
    1.0 - (1.0 + OBSERVER_GAIN * TIMESTEP).powi(-(tick as i32))
}

/// The commanded torque over the tick starting at `time`.
fn drive_torque<T: Numeric>(time: f64) -> Vector<2, T> {
    Vector::new([
        T::from_f64(2.0 * (3.0 * time).sin()),
        T::from_f64((2.0 * time).cos()),
    ])
}

/// Holds `configuration` still with `τ = G(q) − τ_ext`, so `ṗ = 0` and only the discrete law is
/// left. Returns the worst `|r_k − τ_ext·(1 − (1 + K·dt)^−k)|` over `tick_count` ticks.
fn held_still_worst_error<T: Numeric>(tick_count: usize, to_f64: fn(T) -> f64) -> f64 {
    let body = viscous_double_pendulum::<T>();
    let configuration = Vector::new([T::from_f64(0.3), T::from_f64(0.9)]);
    let state = body.tree().forward_kinematics(&configuration).unwrap();
    let gravity_torque = body.gravity_torque(&state).unwrap();
    let torque = Vector::new([
        gravity_torque[0] - T::from_f64(EXTERNAL_TORQUE[0]),
        gravity_torque[1] - T::from_f64(EXTERNAL_TORQUE[1]),
    ]);

    let mut observer = observer::<T>();
    let latched = observer
        .update(&body, &state, &Vector::zeros(), &torque)
        .unwrap();
    assert_eq!(latched, Vector::zeros());

    let mut worst = 0.0_f64;
    for tick in 1..=tick_count {
        let residual = observer
            .update(&body, &state, &Vector::zeros(), &torque)
            .unwrap();
        for joint in 0..2 {
            let want = EXTERNAL_TORQUE[joint] * step_response(tick);
            worst = worst.max((to_f64(residual[joint]) - want).abs());
        }
    }
    worst
}

/// Swings `body` from `[1.2, −0.5]` at rest for two seconds. The plant gets `τ + τ_ext` held over
/// each tick through ten RK4 substeps; the observer gets `τ` alone. Returns the worst
/// `|r_k − τ_ext| − |τ_ext|·(1 + K·dt)^−k` over every tick.
fn swing_worst_excess<T: Numeric>(
    body: &ArticulatedBody<2, 2, T>,
    driven: bool,
    external_torque: [f64; 2],
    to_f64: fn(T) -> f64,
) -> f64 {
    let mut observer = observer::<T>();
    let mut plant = Vector::new([T::from_f64(1.2), T::from_f64(-0.5), T::ZERO, T::ZERO]);
    let latched = observer
        .update_at(
            body,
            &Vector::new([plant[0], plant[1]]),
            &Vector::new([plant[2], plant[3]]),
            &Vector::zeros(),
        )
        .unwrap();
    assert_eq!(latched, Vector::zeros());

    let tick_count = 2000;
    let substep = T::from_f64(TIMESTEP / PLANT_SUBSTEPS as f64);
    let mut worst = 0.0_f64;
    for tick in 0..tick_count {
        let torque = if driven {
            drive_torque::<T>(tick as f64 * TIMESTEP)
        } else {
            Vector::zeros()
        };
        let applied = Vector::new([
            torque[0] + T::from_f64(external_torque[0]),
            torque[1] + T::from_f64(external_torque[1]),
        ]);
        let rate = |_, plant: &Vector<4, T>| {
            let acceleration = body
                .forward_dynamics_at(
                    &Vector::new([plant[0], plant[1]]),
                    &Vector::new([plant[2], plant[3]]),
                    &applied,
                )
                .unwrap();
            Vector::new([plant[2], plant[3], acceleration[0], acceleration[1]])
        };
        for _ in 0..PLANT_SUBSTEPS {
            plant = Rk4::step(&rate, T::ZERO, &plant, substep);
        }

        let residual = observer
            .update_at(
                body,
                &Vector::new([plant[0], plant[1]]),
                &Vector::new([plant[2], plant[3]]),
                &torque,
            )
            .unwrap();
        for joint in 0..2 {
            let envelope = external_torque[joint].abs() * (1.0 - step_response(tick + 1));
            let error = (to_f64(residual[joint]) - external_torque[joint]).abs();
            worst = worst.max(error - envelope);
        }
    }
    worst
}

#[test]
fn held_still_residual_follows_the_discrete_step_response() {
    let worst = held_still_worst_error::<f64>(500, |value| value);
    assert!(
        worst < 1e-12,
        "worst |r − τ_ext·(1 − (1 + K·dt)^−k)| = {worst}"
    );
}

#[test]
fn free_swing_residual_stays_near_zero() {
    let body = viscous_double_pendulum::<f64>();
    let worst = swing_worst_excess(&body, false, [0.0, 0.0], |value| value);
    assert!(worst < 1e-4, "worst |r| = {worst}");
}

#[test]
fn driven_swing_residual_stays_near_zero() {
    let body = viscous_double_pendulum::<f64>();
    let worst = swing_worst_excess(&body, true, [0.0, 0.0], |value| value);
    assert!(worst < 1e-4, "worst |r| = {worst}");
}

#[test]
fn driven_swing_with_a_push_follows_the_step_response() {
    let body = viscous_double_pendulum::<f64>();
    let worst = swing_worst_excess(&body, true, EXTERNAL_TORQUE, |value| value);
    assert!(
        worst < 1e-4,
        "worst excess over the step response = {worst}"
    );
}

#[test]
fn f32_observer_holds_both_identities() {
    let epsilon = f64::from(f32::EPSILON);
    let held = held_still_worst_error::<f32>(500, f64::from);
    assert!(held < 64.0 * epsilon, "held still: {held}");

    // f32 rounding in `p` reaches `r` through `K/(1 + K·dt)`; 2¹⁴·ε covers it at `|p| ≈ 5`.
    let body = viscous_double_pendulum::<f32>();
    let swing = swing_worst_excess(&body, true, EXTERNAL_TORQUE, f64::from);
    assert!(
        swing < 16_384.0 * epsilon,
        "driven swing with a push: {swing}"
    );
}

#[test]
fn reset_restarts_like_a_fresh_observer() {
    let body = viscous_double_pendulum::<f64>();
    let configuration = Vector::new([0.4, -0.2]);
    let velocity = Vector::new([1.0, -0.5]);
    let torque = Vector::new([3.0, 1.0]);

    let mut used = observer::<f64>();
    for _ in 0..50 {
        let _ = used
            .update_at(&body, &configuration, &velocity, &torque)
            .unwrap();
    }
    assert_ne!(used.residual(), Vector::zeros());

    used.reset();
    let fresh = observer::<f64>();
    assert_eq!(used, fresh);

    // A moving start latches with no transient.
    let latched = used
        .update_at(&body, &configuration, &velocity, &torque)
        .unwrap();
    assert_eq!(latched, Vector::zeros());
}

#[test]
fn failed_update_leaves_the_observer_unchanged() {
    let body = viscous_double_pendulum::<f64>();
    let configuration = Vector::new([0.4, -0.2]);
    let velocity = Vector::new([1.0, -0.5]);
    let torque = Vector::new([3.0, 1.0]);

    // A failed first update does not latch.
    let mut observer = observer::<f64>();
    let unstarted = observer;
    assert_eq!(
        observer.update_at(
            &body,
            &configuration,
            &velocity,
            &Vector::new([f64::NAN, 1.0])
        ),
        Err(DynamicsError::NonFinite)
    );
    assert_eq!(observer, unstarted);

    for _ in 0..10 {
        let _ = observer
            .update_at(&body, &configuration, &velocity, &torque)
            .unwrap();
    }
    let before = observer;
    assert_eq!(
        observer.update_at(
            &body,
            &configuration,
            &velocity,
            &Vector::new([f64::INFINITY, 1.0])
        ),
        Err(DynamicsError::NonFinite)
    );
    assert_eq!(
        observer.update_at(
            &body,
            &configuration,
            &Vector::new([f64::NAN, 0.0]),
            &torque
        ),
        Err(DynamicsError::NonFinite)
    );

    let one_joint = KinematicTree::<2, 2, f64>::try_from_joints(
        &[Joint::revolute(
            Vector::new([0.0, 1.0, 0.0]),
            SE3::identity(),
        )],
        &[JointParent::World],
    )
    .unwrap();
    let other_state = one_joint.forward_kinematics(&Vector::zeros()).unwrap();
    assert_eq!(
        observer.update(&body, &other_state, &velocity, &torque),
        Err(DynamicsError::StateShapeMismatch)
    );
    assert_eq!(observer, before);
}

#[test]
fn slots_past_the_model_stay_zero() {
    let hinge = Joint::revolute(Vector::new([0.0, 1.0, 0.0]), SE3::<f64>::identity());
    let tree =
        KinematicTree::<1, 3, f64>::try_from_joints(&[hinge], &[JointParent::World]).unwrap();
    let link = SpatialInertia::new(
        2.0,
        Vector::new([0.5, 0.0, 0.0]),
        Matrix::from_diagonal([0.01, 0.01, 0.01]),
    )
    .unwrap();
    let body = ArticulatedBody::new(tree, &[Some(link)], Vector::new([0.0, 0.0, -9.81])).unwrap();

    let mut observer = MomentumObserver::new(Vector::new([50.0; 3]), TIMESTEP).unwrap();
    let still = Vector::zeros();
    let gravity_torque = body.gravity_torque_at(&still).unwrap();
    let torque = Vector::new([gravity_torque[0] - 0.8, 5.0, -5.0]);
    let mut residual = Vector::zeros();
    for _ in 0..100 {
        residual = observer.update_at(&body, &still, &still, &torque).unwrap();
    }
    assert!(residual[0] > 0.5);
    assert_eq!(residual[1], 0.0);
    assert_eq!(residual[2], 0.0);
}

#[test]
fn constructor_checks_gains_and_timestep() {
    let observer = observer::<f64>();
    assert_eq!(observer.gains(), Vector::new([OBSERVER_GAIN; 2]));
    assert_eq!(observer.timestep(), TIMESTEP);
    assert_eq!(observer.residual(), Vector::zeros());

    let new = MomentumObserver::<2, f64>::new;
    assert_eq!(
        new(Vector::new([50.0, 0.0]), TIMESTEP),
        Err(DynamicsError::NonPositiveGain)
    );
    assert_eq!(
        new(Vector::new([50.0, -1.0]), TIMESTEP),
        Err(DynamicsError::NonPositiveGain)
    );
    assert_eq!(
        new(Vector::new([50.0, f64::NAN]), TIMESTEP),
        Err(DynamicsError::NonFinite)
    );
    assert_eq!(
        new(Vector::new([50.0, 50.0]), 0.0),
        Err(DynamicsError::NonPositiveTimestep)
    );
    assert_eq!(
        new(Vector::new([50.0, 50.0]), -TIMESTEP),
        Err(DynamicsError::NonPositiveTimestep)
    );
    assert_eq!(
        new(Vector::new([50.0, 50.0]), f64::INFINITY),
        Err(DynamicsError::NonFinite)
    );
}
