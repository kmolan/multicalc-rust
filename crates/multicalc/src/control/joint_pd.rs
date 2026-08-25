//! High-stiffness joint-space PD, the torque-side view of position-controlled hardware.
#![deny(clippy::indexing_slicing)]

use crate::control::motion_reference::{JointReference, joint_position_error};
use crate::dynamics::ArticulatedBody;
use crate::error::ControlError;
use crate::kinematics::KinematicTreeState;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// Proportional-derivative control in joint space, optionally with gravity cancelled.
///
/// `τ = kp⊙e + kd⊙ė`, plus `G(q)` when gravity compensation is on.
///
/// No Coriolis or friction feedforward: this is the law to reach for when the model is not trusted,
/// and at high stiffness the feedback dominates those terms anyway. It is the torque-side
/// counterpart of [`PositionServo`](crate::plant::PositionServo) — the same closed loop, seen from
/// the torque side rather than the position-command side.
///
/// A discrete high-gain PD has a sample-rate-bounded stability limit: gains must be chosen for the
/// loop rate, and no gain is stable at every rate.
///
/// The model is still an argument with gravity compensation off, because the position error needs
/// the tree to wrap a `Continuous` joint.
///
/// ```
/// use multicalc::control::{JointPdController, JointReference};
/// use multicalc::dynamics::ArticulatedBody;
/// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
/// use multicalc::linear_algebra::{Matrix, Vector};
/// use multicalc::spatial::{SE3, SpatialInertia};
///
/// let hinge = Joint::revolute(Vector::new([0.0, 1.0, 0.0]), SE3::<f64>::identity());
/// let tree = KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World])
///     .unwrap();
/// let link = SpatialInertia::new(
///     2.0,
///     Vector::new([0.5, 0.0, 0.0]),
///     Matrix::from_diagonal([0.01, 0.01, 0.01]),
/// )
/// .unwrap();
/// let body =
///     ArticulatedBody::new(tree, &[Some(link)], Vector::new([0.0, 0.0, -9.81])).unwrap();
///
/// let position_gain = 400.0;
/// let controller = JointPdController::new(Vector::new([position_gain]), Vector::new([40.0]))
///     .unwrap();
/// let reference = JointReference::at_rest(Vector::new([0.6]));
///
/// let settle = |controller: &JointPdController<1, f64>| {
///     let timestep = 0.001;
///     let mut position = Vector::new([0.0]);
///     let mut velocity = Vector::zeros();
///     for _ in 0..2000 {
///         let torque = controller
///             .torque_at(&body, &position, &velocity, &reference)
///             .unwrap();
///         let acceleration = body.forward_dynamics_at(&position, &velocity, &torque).unwrap();
///         velocity = velocity + acceleration.scale(timestep);
///         position = position + velocity.scale(timestep);
///     }
///     position
/// };
///
/// // With gravity cancelled it lands on the setpoint.
/// let compensated = settle(&controller.with_gravity_compensation(true));
/// assert!((compensated[0] - 0.6).abs() < 1e-3);
///
/// // Without, it holds where the spring balances gravity: `kp⊙e = G(q)`.
/// let sagged = settle(&controller);
/// let gravity = body.gravity_torque_at(&sagged).unwrap();
/// assert!((0.6 - sagged[0]).abs() > 1e-3);
/// assert!((0.6 - sagged[0] - gravity[0] / position_gain).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointPdController<const MAX_CONFIG: usize, T: Numeric = f64> {
    position_gains: Vector<MAX_CONFIG, T>,
    velocity_gains: Vector<MAX_CONFIG, T>,
    compensate_gravity: bool,
}

impl<const MAX_CONFIG: usize, T: Numeric> JointPdController<MAX_CONFIG, T> {
    /// Builds the controller from per-joint position and velocity gains, with gravity compensation
    /// off.
    ///
    /// Returns [`ControlError::NonFinite`] if any gain is not finite, or
    /// [`ControlError::NegativeGain`] if any gain is negative.
    pub fn new(
        position_gains: Vector<MAX_CONFIG, T>,
        velocity_gains: Vector<MAX_CONFIG, T>,
    ) -> Result<Self, ControlError> {
        if !position_gains.is_finite() || !velocity_gains.is_finite() {
            return Err(ControlError::NonFinite);
        }
        for slot in 0..MAX_CONFIG {
            let position_gain = position_gains.get(slot).copied().unwrap_or(T::ZERO);
            let velocity_gain = velocity_gains.get(slot).copied().unwrap_or(T::ZERO);
            if position_gain < T::ZERO || velocity_gain < T::ZERO {
                return Err(ControlError::NegativeGain);
            }
        }
        Ok(Self {
            position_gains,
            velocity_gains,
            compensate_gravity: false,
        })
    }

    /// Turns gravity cancellation on or off. Off by default.
    #[inline]
    #[must_use]
    pub fn with_gravity_compensation(mut self, compensate: bool) -> Self {
        self.compensate_gravity = compensate;
        self
    }

    /// The per-joint position gains `kp`.
    #[inline]
    pub fn position_gains(&self) -> Vector<MAX_CONFIG, T> {
        self.position_gains
    }

    /// The per-joint velocity gains `kd`.
    #[inline]
    pub fn velocity_gains(&self) -> Vector<MAX_CONFIG, T> {
        self.velocity_gains
    }

    /// Whether the gravity term is cancelled.
    #[inline]
    #[must_use]
    pub fn compensates_gravity(&self) -> bool {
        self.compensate_gravity
    }

    /// Joint torque for already-solved poses.
    ///
    /// `τ = kp⊙e + kd⊙(q̇_d − q̇)`, plus `G(q)` with compensation on.
    ///
    /// Errors: [`ControlError::Dynamics`] carrying whatever [`ArticulatedBody::gravity_torque`]
    /// reports, only reachable with compensation on.
    pub fn torque<const MAX_JOINTS: usize>(
        &self,
        body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        measured_position: &Vector<MAX_CONFIG, T>,
        measured_velocity: &Vector<MAX_CONFIG, T>,
        reference: &JointReference<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, ControlError> {
        let error = joint_position_error(body, &reference.position, measured_position);
        let feedback = Vector::from_fn(|slot| {
            let rate_error = reference.velocity.get(slot).copied().unwrap_or(T::ZERO)
                - measured_velocity.get(slot).copied().unwrap_or(T::ZERO);
            self.position_gains.get(slot).copied().unwrap_or(T::ZERO)
                * error.get(slot).copied().unwrap_or(T::ZERO)
                + self.velocity_gains.get(slot).copied().unwrap_or(T::ZERO) * rate_error
        });
        if self.compensate_gravity {
            Ok(feedback + body.gravity_torque(state)?)
        } else {
            Ok(feedback)
        }
    }

    /// Joint torque for configuration `measured_position`.
    ///
    /// Solves the world poses first, then hands them to [`torque`](JointPdController::torque).
    ///
    /// Errors: as [`forward_kinematics`](crate::kinematics::KinematicTree::forward_kinematics), via
    /// [`ControlError::Kinematics`], and as [`torque`](JointPdController::torque).
    pub fn torque_at<const MAX_JOINTS: usize>(
        &self,
        body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
        measured_position: &Vector<MAX_CONFIG, T>,
        measured_velocity: &Vector<MAX_CONFIG, T>,
        reference: &JointReference<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, ControlError> {
        let state = body.tree().forward_kinematics(measured_position)?;
        self.torque(
            body,
            &state,
            measured_position,
            measured_velocity,
            reference,
        )
    }
}
