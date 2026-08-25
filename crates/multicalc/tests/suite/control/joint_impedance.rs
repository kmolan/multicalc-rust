//! Joint-impedance tests: that the arm keeps its natural inertia under pure bias compensation,
//! that a free axis is a legitimate gain, and the constructor's refusals.

use multicalc::control::{JointImpedanceController, JointReference};
use multicalc::dynamics::ArticulatedBody;
use multicalc::error::ControlError;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::Numeric;
use multicalc::spatial::{SE3, SO3, SpatialInertia};

const GRAVITY: f64 = -9.81;

/// Two revolute joints about `+y`, a unit link between them, both links 2 kg balancing 0.5 m out,
/// carrying armature, viscous damping and Coulomb friction.
fn double_pendulum() -> ArticulatedBody<2, 2, f64> {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let link = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
    let origins = [SE3::identity(), link];
    let armature = [0.05, 0.03];
    let damping = [0.7, 0.4];
    let friction_loss = [0.2, 0.1];

    let mut tree = KinematicTree::<2, 2, f64>::new();
    for index in 0..2 {
        let joint = Joint::revolute(axis, origins[index])
            .with_armature(armature[index])
            .with_damping(damping[index])
            .with_friction_loss(friction_loss[index]);
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(joint, parent).unwrap();
    }

    let inertia = SpatialInertia::new(
        2.0,
        Vector::new([f64::HALF, 0.0, 0.0]),
        Matrix::from_diagonal([0.01, 0.01, 0.01]),
    )
    .unwrap();
    ArticulatedBody::new(
        tree,
        &[Some(inertia), Some(inertia)],
        Vector::new([0.0, 0.0, GRAVITY]),
    )
    .unwrap()
}

#[test]
fn impedance_keeps_the_natural_inertia() {
    let body = double_pendulum();
    // No spring and no damper: pure bias compensation, so nothing is left to accelerate the arm.
    let controller = JointImpedanceController::new(Vector::zeros(), Vector::zeros()).unwrap();

    for (position, velocity) in [
        (Vector::new([0.4, -0.9]), Vector::new([0.7, -0.2])),
        (Vector::new([-1.2, 2.1]), Vector::new([-0.3, 0.9])),
        (Vector::new([0.0, 0.0]), Vector::new([0.0, 0.0])),
    ] {
        // The reference rate has to match the measured one, or the Coulomb feedforward moves the
        // friction term off the rate the model cancelled it at.
        let reference = JointReference::new(position, velocity, Vector::zeros());
        let torque = controller
            .torque_at(&body, &position, &velocity, &reference)
            .unwrap();
        let acceleration = body
            .forward_dynamics_at(&position, &velocity, &torque)
            .unwrap();
        for joint in 0..2 {
            assert!(
                acceleration[joint].abs() < 1e-10,
                "joint {joint} at {position:?}: {}",
                acceleration[joint]
            );
        }
    }
}

#[test]
fn zero_stiffness_is_accepted() {
    assert!(
        JointImpedanceController::new(Vector::<2, f64>::zeros(), Vector::new([1.0, 1.0])).is_ok()
    );
    assert_eq!(
        JointImpedanceController::new(Vector::new([-1.0, 0.0]), Vector::new([1.0, 1.0])),
        Err(ControlError::NegativeGain)
    );
}

#[test]
fn gains_are_validated() {
    assert_eq!(
        JointImpedanceController::new(Vector::new([f64::NAN]), Vector::new([1.0])),
        Err(ControlError::NonFinite)
    );
    assert_eq!(
        JointImpedanceController::new(Vector::new([1.0]), Vector::new([-1.0])),
        Err(ControlError::NegativeGain)
    );
}
