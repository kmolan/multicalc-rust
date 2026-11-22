//! Computed-torque control: feedback linearization through the joint-space inertia.
#![deny(clippy::indexing_slicing)]

use crate::control::motion_reference::{
    JointReference, coulomb_feedforward_correction, joint_position_error,
};
use crate::dynamics::ArticulatedBody;
use crate::error::ControlError;
use crate::kinematics::KinematicTreeState;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// Model-based joint tracking control: the PD term is driven through the joint-space inertia, so
/// the tracking error obeys `ë + kd⊙ė + kp⊙e = 0` exactly when the model is exact.
///
/// One RNEA pass per tick: [`ArticulatedBody::inverse_dynamics`] evaluates
/// `H(q)·q̈ + C(q,q̇)·q̇ + G(q) + armature⊙q̈ + damping⊙q̇ + friction_loss⊙sign(q̇)`, so evaluating it
/// at the reference acceleration is the whole law. A second O(n) pass moves the Coulomb term to the
/// desired rate.
///
/// Gains are held per joint, not as matrices: the law is diagonal in joint space by construction.
///
/// Every operation is generic over [`Numeric`](crate::Numeric), so wrapping one `torque` in a
/// [`Dual`](crate::Dual) differentiates the whole control law exactly.
///
/// ```
/// use multicalc::control::{ComputedTorqueController, JointReference};
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
/// // ω = 10 rad/s, critically damped.
/// let controller = ComputedTorqueController::<1, f64>::from_natural_frequency(10.0, 1.0)
///     .unwrap();
/// let reference = JointReference::at_rest(Vector::new([0.6]));
///
/// let timestep = 0.001;
/// let mut position = Vector::new([0.0]);
/// let mut velocity = Vector::zeros();
/// for _ in 0..2000 {
///     let torque = controller
///         .torque_at(&body, &position, &velocity, &reference)
///         .unwrap();
///     let acceleration = body.forward_dynamics_at(&position, &velocity, &torque).unwrap();
///     velocity = velocity + acceleration.scale(timestep);
///     position = position + velocity.scale(timestep);
/// }
/// assert!((position[0] - 0.6).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComputedTorqueController<const MAX_CONFIG: usize, T: Numeric = f64> {
    position_gains: Vector<MAX_CONFIG, T>,
    velocity_gains: Vector<MAX_CONFIG, T>,
}

impl<const MAX_CONFIG: usize, T: Numeric> ComputedTorqueController<MAX_CONFIG, T> {
    /// Builds the controller from per-joint position and velocity gains.
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
        })
    }

    /// Uniform gains from a closed-loop natural frequency and damping ratio:
    /// `kp = ω²`, `kd = 2·ζ·ω`, giving `ë + 2ζω·ė + ω²·e = 0` on every joint.
    ///
    /// A discrete loop has a sample-rate-bounded stability limit; pick `ω` well below the loop rate.
    ///
    /// Returns [`ControlError::NonFinite`] if either argument is not finite,
    /// [`ControlError::NonPositiveGain`] if `natural_frequency` is not strictly positive, or
    /// [`ControlError::NegativeGain`] if `damping_ratio` is negative.
    pub fn from_natural_frequency(
        natural_frequency: T,
        damping_ratio: T,
    ) -> Result<Self, ControlError> {
        if !natural_frequency.is_finite() || !damping_ratio.is_finite() {
            return Err(ControlError::NonFinite);
        }
        if natural_frequency <= T::ZERO {
            return Err(ControlError::NonPositiveGain);
        }
        if damping_ratio < T::ZERO {
            return Err(ControlError::NegativeGain);
        }
        Ok(Self {
            position_gains: Vector::from_fn(|_| natural_frequency * natural_frequency),
            velocity_gains: Vector::from_fn(|_| T::TWO * damping_ratio * natural_frequency),
        })
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

    /// The acceleration the feedback asks for: `q̈_d + kd⊙(q̇_d − q̇) + kp⊙e`.
    ///
    /// Exposed because it is the quantity a stability argument is made about, and because a caller
    /// running its own recursion can feed it straight to [`ArticulatedBody::inverse_dynamics`].
    pub fn reference_acceleration<const MAX_JOINTS: usize>(
        &self,
        body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
        measured_position: &Vector<MAX_CONFIG, T>,
        measured_velocity: &Vector<MAX_CONFIG, T>,
        reference: &JointReference<MAX_CONFIG, T>,
    ) -> Vector<MAX_CONFIG, T> {
        let error = joint_position_error(body, &reference.position, measured_position);
        Vector::from_fn(|slot| {
            let acceleration = reference.acceleration.get(slot).copied().unwrap_or(T::ZERO);
            let rate_error = reference.velocity.get(slot).copied().unwrap_or(T::ZERO)
                - measured_velocity.get(slot).copied().unwrap_or(T::ZERO);
            let position_error = error.get(slot).copied().unwrap_or(T::ZERO);
            acceleration
                + self.velocity_gains.get(slot).copied().unwrap_or(T::ZERO) * rate_error
                + self.position_gains.get(slot).copied().unwrap_or(T::ZERO) * position_error
        })
    }

    /// Joint torque for already-solved poses.
    ///
    /// `τ = H(q)·a_ref + C(q,q̇)·q̇ + G(q) + armature⊙a_ref + damping⊙q̇ + friction_loss⊙sign(q̇_d)`,
    /// with `a_ref = q̈_d + kd⊙(q̇_d − q̇) + kp⊙e`.
    ///
    /// Coulomb friction is fed forward at the *desired* rate, so it cannot flip sign on measurement
    /// noise about zero; viscous damping stays at the measured rate, where it is linear and cancels
    /// the model exactly. Neither term is differentiable where the desired rate crosses zero: under
    /// [`Dual`](crate::Dual) the Coulomb term contributes nothing to `∂τ/∂q̇_d` there.
    ///
    /// Errors: [`ControlError::Dynamics`] carrying whatever
    /// [`ArticulatedBody::inverse_dynamics`] reports.
    pub fn torque<const MAX_JOINTS: usize>(
        &self,
        body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        measured_position: &Vector<MAX_CONFIG, T>,
        measured_velocity: &Vector<MAX_CONFIG, T>,
        reference: &JointReference<MAX_CONFIG, T>,
    ) -> Result<Vector<MAX_CONFIG, T>, ControlError> {
        let acceleration =
            self.reference_acceleration(body, measured_position, measured_velocity, reference);
        let torque = body.inverse_dynamics(state, measured_velocity, &acceleration)?;
        let correction =
            coulomb_feedforward_correction(body, &reference.velocity, measured_velocity);
        Ok(torque + correction)
    }

    /// Joint torque for configuration `measured_position`.
    ///
    /// Solves the world poses first, then hands them to [`torque`](ComputedTorqueController::torque).
    ///
    /// Errors: as [`forward_kinematics`](crate::kinematics::KinematicTree::forward_kinematics), via
    /// [`ControlError::Kinematics`], and as [`torque`](ComputedTorqueController::torque).
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
