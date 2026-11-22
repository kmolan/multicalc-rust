//! Computed-torque tests: the exactly-linear error dynamics that define the law, the Coulomb
//! feedforward's desired-rate evaluation, the wrap a continuous joint's error needs, the `Dual`
//! derivative, and the constructor's refusals.

use core::f64::consts::PI;

use multicalc::control::{ComputedTorqueController, JointReference};
use multicalc::dynamics::ArticulatedBody;
use multicalc::error::ControlError;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::{SE3, SO3, SpatialInertia};

const GRAVITY: f64 = -9.81;

/// A link of `mass` balancing 0.5 m out along `x`, isotropic about its own balance point.
fn link_inertia<T: Numeric>(mass: f64) -> SpatialInertia<T> {
    SpatialInertia::new(
        T::from_f64(mass),
        Vector::new([T::HALF, T::ZERO, T::ZERO]),
        Matrix::from_diagonal([T::from_f64(0.01), T::from_f64(0.01), T::from_f64(0.01)]),
    )
    .unwrap()
}

fn earth_gravity<T: Numeric>() -> Vector<3, T> {
    Vector::new([T::ZERO, T::ZERO, T::from_f64(GRAVITY)])
}

/// One revolute joint about `+y` at the origin, a 2 kg link balancing 0.5 m out along `x`.
fn single_pendulum<T: Numeric>() -> ArticulatedBody<1, 1, T> {
    let axis = Vector::new([T::ZERO, T::ONE, T::ZERO]);
    let hinge = Joint::revolute(axis, SE3::identity());
    let tree = KinematicTree::<1, 1, T>::try_from_joints(&[hinge], &[JointParent::World]).unwrap();
    ArticulatedBody::new(tree, &[Some(link_inertia(2.0))], earth_gravity()).unwrap()
}

/// Two revolute joints about `+y`, a unit link between them, both links 2 kg balancing 0.5 m out.
fn double_pendulum<T: Numeric>(friction_loss: f64) -> ArticulatedBody<2, 2, T> {
    let axis = Vector::new([T::ZERO, T::ONE, T::ZERO]);
    let link = SE3::from_parts(SO3::identity(), Vector::new([T::ONE, T::ZERO, T::ZERO]));
    let origins = [SE3::identity(), link];
    let armature = [0.05, 0.03];
    let damping = [0.7, 0.4];

    let mut tree = KinematicTree::<2, 2, T>::new();
    for index in 0..2 {
        let joint = Joint::revolute(axis, origins[index])
            .with_armature(T::from_f64(armature[index]))
            .with_damping(T::from_f64(damping[index]))
            .with_friction_loss(T::from_f64(friction_loss));
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(joint, parent).unwrap();
    }

    let inertia = link_inertia(2.0);
    ArticulatedBody::new(tree, &[Some(inertia), Some(inertia)], earth_gravity()).unwrap()
}

/// One continuous joint about `+y`, so the position error wraps.
fn continuous_hinge() -> ArticulatedBody<1, 1, f64> {
    let hinge = Joint::continuous(Vector::new([0.0, 1.0, 0.0]), SE3::identity());
    let tree =
        KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World]).unwrap();
    ArticulatedBody::new(tree, &[Some(link_inertia(2.0))], earth_gravity()).unwrap()
}

#[test]
fn error_dynamics_are_exactly_linear() {
    let body = double_pendulum::<f64>(0.0);
    let position_gains = Vector::new([100.0, 64.0]);
    let velocity_gains = Vector::new([20.0, 16.0]);
    let controller = ComputedTorqueController::new(position_gains, velocity_gains).unwrap();

    let position = Vector::new([0.4, -0.9]);
    let velocity = Vector::new([0.7, -0.2]);
    let reference = JointReference::new(
        Vector::new([-0.3, 0.5]),
        Vector::new([-0.6, 0.25]),
        Vector::new([1.1, -0.8]),
    );

    let torque = controller
        .torque_at(&body, &position, &velocity, &reference)
        .unwrap();
    let acceleration = body
        .forward_dynamics_at(&position, &velocity, &torque)
        .unwrap();

    // `ë + kd⊙ė + kp⊙e = 0`, rearranged.
    for joint in 0..2 {
        let want = reference.acceleration[joint]
            + velocity_gains[joint] * (reference.velocity[joint] - velocity[joint])
            + position_gains[joint] * (reference.position[joint] - position[joint]);
        assert!(
            (acceleration[joint] - want).abs() < 1e-10,
            "joint {joint}: {} vs {want}",
            acceleration[joint]
        );
    }
}

#[test]
fn coulomb_feedforward_uses_the_desired_rate() {
    let friction_loss = 0.3;
    let body = double_pendulum::<f64>(friction_loss);
    let controller =
        ComputedTorqueController::new(Vector::new([100.0, 64.0]), Vector::new([20.0, 16.0]))
            .unwrap();

    let position = Vector::new([0.4, -0.9]);
    let velocity = Vector::new([0.5, -0.5]);
    let state = body.tree().forward_kinematics(&position).unwrap();

    let sign = |rate: f64| if rate > 0.0 { 1.0 } else { -1.0 };
    for desired_velocity in [Vector::new([-0.4, 0.4]), velocity] {
        let reference =
            JointReference::new(desired_velocity, desired_velocity, Vector::new([1.1, -0.8]));
        let acceleration =
            controller.reference_acceleration(&body, &position, &velocity, &reference);
        let torque = controller
            .torque(&body, &state, &position, &velocity, &reference)
            .unwrap();
        let model = body
            .inverse_dynamics(&state, &velocity, &acceleration)
            .unwrap();

        for joint in 0..2 {
            let want = friction_loss * (sign(desired_velocity[joint]) - sign(velocity[joint]));
            assert!(
                (torque[joint] - model[joint] - want).abs() < 1e-12,
                "joint {joint}: {} vs {want}",
                torque[joint] - model[joint]
            );
        }
    }
}

#[test]
fn standing_joint_carries_no_coulomb_feedforward() {
    let body = double_pendulum::<f64>(0.3);
    let controller =
        ComputedTorqueController::new(Vector::new([100.0, 64.0]), Vector::new([20.0, 16.0]))
            .unwrap();

    let position = Vector::new([0.4, -0.9]);
    let velocity = Vector::zeros();
    let reference = JointReference::at_rest(Vector::new([-0.3, 0.5]));
    let state = body.tree().forward_kinematics(&position).unwrap();

    let acceleration = controller.reference_acceleration(&body, &position, &velocity, &reference);
    let torque = controller
        .torque(&body, &state, &position, &velocity, &reference)
        .unwrap();
    let model = body
        .inverse_dynamics(&state, &velocity, &acceleration)
        .unwrap();

    for joint in 0..2 {
        assert_eq!(torque[joint], model[joint], "joint {joint}");
    }
}

#[test]
fn continuous_joints_wrap_the_position_error() {
    let body = continuous_hinge();
    // Unit position gain and no rate feedback, so the reference acceleration *is* the error.
    let controller = ComputedTorqueController::new(Vector::new([1.0]), Vector::new([0.0])).unwrap();
    let reference = JointReference::at_rest(Vector::new([-3.0]));

    let error =
        controller.reference_acceleration(&body, &Vector::new([3.0]), &Vector::zeros(), &reference);
    assert!((error[0] - (2.0 * PI - 6.0)).abs() < 1e-12, "{}", error[0]);
}

#[test]
fn torque_differentiates_under_dual() {
    let dual_body = single_pendulum::<Dual<f64>>();
    let plain = single_pendulum::<f64>();
    let dual_controller = ComputedTorqueController::<1, Dual<f64>>::from_natural_frequency(
        Dual::constant(10.0),
        Dual::constant(1.0),
    )
    .unwrap();
    let controller = ComputedTorqueController::<1, f64>::from_natural_frequency(10.0, 1.0).unwrap();

    let base_position = 0.4;
    let velocity = 0.7;
    let setpoint = -0.2;

    let by_position = dual_controller
        .torque_at(
            &dual_body,
            &Vector::new([Dual::variable(base_position)]),
            &Vector::new([Dual::constant(velocity)]),
            &JointReference::at_rest(Vector::new([Dual::constant(setpoint)])),
        )
        .unwrap()[0]
        .deriv;

    let torque_at = |position: f64| {
        controller
            .torque_at(
                &plain,
                &Vector::new([position]),
                &Vector::new([velocity]),
                &JointReference::at_rest(Vector::new([setpoint])),
            )
            .unwrap()[0]
    };
    let step = 1e-6;
    let difference =
        (torque_at(base_position + step) - torque_at(base_position - step)) / (2.0 * step);
    assert!(
        (by_position - difference).abs() <= 1e-7 * difference.abs().max(1.0),
        "{by_position} vs {difference}"
    );
}

#[test]
fn gains_are_validated() {
    assert_eq!(
        ComputedTorqueController::new(Vector::new([f64::NAN]), Vector::new([1.0])),
        Err(ControlError::NonFinite)
    );
    assert_eq!(
        ComputedTorqueController::new(Vector::new([1.0]), Vector::new([-1.0])),
        Err(ControlError::NegativeGain)
    );
    assert_eq!(
        ComputedTorqueController::<1, f64>::from_natural_frequency(0.0, 1.0),
        Err(ControlError::NonPositiveGain)
    );
    assert_eq!(
        ComputedTorqueController::<1, f64>::from_natural_frequency(10.0, -0.1),
        Err(ControlError::NegativeGain)
    );
}
