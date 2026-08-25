//! Joint-PD tests: the droop gravity compensation removes and the droop it leaves behind, and the
//! constructor's refusals.

use multicalc::control::{JointPdController, JointReference};
use multicalc::dynamics::ArticulatedBody;
use multicalc::error::ControlError;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::spatial::{SE3, SpatialInertia};

const GRAVITY: f64 = -9.81;

/// One revolute joint about `+y` at the origin, a 2 kg link balancing 0.5 m out along `x`.
fn single_pendulum() -> ArticulatedBody<1, 1, f64> {
    let hinge = Joint::revolute(Vector::new([0.0, 1.0, 0.0]), SE3::identity());
    let tree =
        KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World]).unwrap();
    let inertia = SpatialInertia::new(
        2.0,
        Vector::new([0.5, 0.0, 0.0]),
        Matrix::from_diagonal([0.01, 0.01, 0.01]),
    )
    .unwrap();
    ArticulatedBody::new(tree, &[Some(inertia)], Vector::new([0.0, 0.0, GRAVITY])).unwrap()
}

/// 4 s of the closed loop at 1 ms, from rest at zero, by explicit Euler on the crate's own forward
/// dynamics.
fn settle(body: &ArticulatedBody<1, 1, f64>, controller: &JointPdController<1, f64>) -> f64 {
    let reference = JointReference::at_rest(Vector::new([0.6]));
    let timestep = 0.001;
    let mut position = Vector::zeros();
    let mut velocity = Vector::zeros();
    for _ in 0..4000 {
        let torque = controller
            .torque_at(body, &position, &velocity, &reference)
            .unwrap();
        let acceleration = body
            .forward_dynamics_at(&position, &velocity, &torque)
            .unwrap();
        velocity += acceleration.scale(timestep);
        position += velocity.scale(timestep);
    }
    position[0]
}

#[test]
fn gravity_compensation_removes_the_droop() {
    let body = single_pendulum();
    let position_gain = 40.0;
    let controller =
        JointPdController::new(Vector::new([position_gain]), Vector::new([12.0])).unwrap();
    assert!(!controller.compensates_gravity());

    let compensated = settle(&body, &controller.with_gravity_compensation(true));
    assert!((compensated - 0.6).abs() < 1e-4, "{compensated}");

    // Without compensation the spring stalls where it balances gravity: `kp⊙e = G(q)`.
    let sagged = settle(&body, &controller);
    let gravity = body.gravity_torque_at(&Vector::new([sagged])).unwrap();
    let droop = 0.6 - sagged;
    assert!(droop.abs() > 1e-3, "{droop}");
    assert!(
        (droop - gravity[0] / position_gain).abs() < 1e-3,
        "{droop} vs {}",
        gravity[0] / position_gain
    );
}

#[test]
fn gains_are_validated() {
    assert_eq!(
        JointPdController::new(Vector::new([f64::INFINITY]), Vector::new([1.0])),
        Err(ControlError::NonFinite)
    );
    assert_eq!(
        JointPdController::new(Vector::new([1.0]), Vector::new([-1.0])),
        Err(ControlError::NegativeGain)
    );
}
