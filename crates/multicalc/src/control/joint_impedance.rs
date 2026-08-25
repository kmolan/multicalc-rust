//! Joint-space impedance: a spring-damper about a reference, on the model's own inertia.
#![deny(clippy::indexing_slicing)]

use crate::control::motion_reference::{
    JointReference, coulomb_feedforward_correction, joint_position_error,
};
use crate::dynamics::ArticulatedBody;
use crate::error::ControlError;
use crate::kinematics::KinematicTreeState;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// A spring-damper in joint space, on top of full bias compensation.
///
/// `τ = k⊙e + d⊙ė + C(q,q̇)·q̇ + G(q) + damping⊙q̇ + friction_loss⊙sign(q̇_d)`.
///
/// Gravity, Coriolis and the passive terms are cancelled, leaving the spring-damper the gains
/// describe on the model's **natural inertia** — that is what makes this compliant rather than
/// stiff. No acceleration feedforward, which would make it a tracking law; for tracking use
/// [`ComputedTorqueController`](crate::control::ComputedTorqueController), whose PD term is driven
/// through `H(q)` for exactly linear error dynamics.
///
/// Zero stiffness on an axis is meaningful — the joint is free along it — so gains are required to
/// be non-negative rather than strictly positive.
///
/// ```
/// use multicalc::control::{JointImpedanceController, JointReference};
/// use multicalc::dynamics::ArticulatedBody;
/// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
/// use multicalc::linear_algebra::{Matrix, Vector};
/// use multicalc::spatial::{SE3, SO3, SpatialInertia};
///
/// let axis = Vector::new([0.0, 1.0, 0.0]);
/// let link_offset = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
/// let tree = KinematicTree::<2, 2, f64>::try_from_joints(
///     &[
///         Joint::revolute(axis, SE3::identity()),
///         Joint::revolute(axis, link_offset),
///     ],
///     &[JointParent::World, JointParent::Joint(0)],
/// )
/// .unwrap();
/// let link = SpatialInertia::new(
///     2.0,
///     Vector::new([0.5, 0.0, 0.0]),
///     Matrix::from_diagonal([0.01, 0.01, 0.01]),
/// )
/// .unwrap();
/// let body = ArticulatedBody::new(
///     tree,
///     &[Some(link), Some(link)],
///     Vector::new([0.0, 0.0, -9.81]),
/// )
/// .unwrap();
///
/// // Soft springs, damped near critical on each joint's own inertia. With the bias cancelled the
/// // arm settles on the reference rather than sagging under gravity.
/// let controller =
///     JointImpedanceController::new(Vector::new([20.0, 20.0]), Vector::new([20.0, 6.5])).unwrap();
/// let reference = JointReference::at_rest(Vector::new([0.4, -0.3]));
///
/// let timestep = 0.001;
/// let mut position = Vector::zeros();
/// let mut velocity = Vector::zeros();
/// for _ in 0..8000 {
///     let torque = controller
///         .torque_at(&body, &position, &velocity, &reference)
///         .unwrap();
///     let acceleration = body.forward_dynamics_at(&position, &velocity, &torque).unwrap();
///     velocity = velocity + acceleration.scale(timestep);
///     position = position + velocity.scale(timestep);
/// }
/// assert!((position[0] - reference.position[0]).abs() < 1e-4);
/// assert!((position[1] - reference.position[1]).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointImpedanceController<const MAX_CONFIG: usize, T: Numeric = f64> {
    stiffness: Vector<MAX_CONFIG, T>,
    damping: Vector<MAX_CONFIG, T>,
}

impl<const MAX_CONFIG: usize, T: Numeric> JointImpedanceController<MAX_CONFIG, T> {
    /// Builds the controller from per-joint stiffness and damping.
    ///
    /// Returns [`ControlError::NonFinite`] if any gain is not finite, or
    /// [`ControlError::NegativeGain`] if any gain is negative.
    pub fn new(
        stiffness: Vector<MAX_CONFIG, T>,
        damping: Vector<MAX_CONFIG, T>,
    ) -> Result<Self, ControlError> {
        if !stiffness.is_finite() || !damping.is_finite() {
            return Err(ControlError::NonFinite);
        }
        for slot in 0..MAX_CONFIG {
            let spring = stiffness.get(slot).copied().unwrap_or(T::ZERO);
            let damper = damping.get(slot).copied().unwrap_or(T::ZERO);
            if spring < T::ZERO || damper < T::ZERO {
                return Err(ControlError::NegativeGain);
            }
        }
        Ok(Self { stiffness, damping })
    }

    /// The per-joint stiffness.
    #[inline]
    pub fn stiffness(&self) -> Vector<MAX_CONFIG, T> {
        self.stiffness
    }

    /// The per-joint damping.
    #[inline]
    pub fn damping(&self) -> Vector<MAX_CONFIG, T> {
        self.damping
    }

    /// Joint torque for already-solved poses.
    ///
    /// `τ = k⊙e + d⊙(q̇_d − q̇) + C(q,q̇)·q̇ + G(q) + damping⊙q̇ + friction_loss⊙sign(q̇_d)`.
    ///
    /// Coulomb friction is fed forward at the desired rate; viscous damping stays at the measured
    /// rate, where it is linear and cancels the model exactly.
    ///
    /// Errors: [`ControlError::Dynamics`] carrying whatever [`ArticulatedBody::bias_torque`]
    /// reports.
    pub fn torque<const MAX_JOINTS: usize>(
        &self,
        body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        measured_position: &Vector<MAX_CONFIG, T>,
        measured_velocity: &Vector<MAX_CONFIG, T>,
        reference: &JointReference<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, ControlError> {
        let error = joint_position_error(body, &reference.position, measured_position);
        let bias = body.bias_torque(state, measured_velocity)?;
        let correction =
            coulomb_feedforward_correction(body, &reference.velocity, measured_velocity);
        let feedback = Vector::from_fn(|slot| {
            let rate_error = reference.velocity.get(slot).copied().unwrap_or(T::ZERO)
                - measured_velocity.get(slot).copied().unwrap_or(T::ZERO);
            self.stiffness.get(slot).copied().unwrap_or(T::ZERO)
                * error.get(slot).copied().unwrap_or(T::ZERO)
                + self.damping.get(slot).copied().unwrap_or(T::ZERO) * rate_error
        });
        Ok(feedback + bias + correction)
    }

    /// Joint torque for configuration `measured_position`.
    ///
    /// Solves the world poses first, then hands them to
    /// [`torque`](JointImpedanceController::torque).
    ///
    /// Errors: as [`forward_kinematics`](crate::kinematics::KinematicTree::forward_kinematics), via
    /// [`ControlError::Kinematics`], and as [`torque`](JointImpedanceController::torque).
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
