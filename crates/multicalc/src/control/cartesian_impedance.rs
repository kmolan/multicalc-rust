//! Cartesian impedance: a spring-damper at the tool, applied through `Jᵀ`.
#![deny(clippy::indexing_slicing)]

use crate::control::motion_reference::{CartesianReference, joint_position_error};
use crate::dynamics::ArticulatedBody;
use crate::error::{ControlError, KinematicsError};
use crate::kinematics::{JacobianFrame, KinematicTreeState};
use crate::linear_algebra::{Vector, Vector6D};
use crate::scalar::Numeric;
use crate::spatial::Twist;

/// A six-axis spring-damper at a tool frame, mapped to joint torque by the Jacobian transpose.
///
/// `J` maps joint rates to a tool twist, so `Jᵀ` maps a tool wrench to joint torques. No inverse
/// kinematics and no inversion of `J`: near a singularity the law loses the ability to push along
/// some direction rather than producing large joint motions.
///
/// ```text
/// e       = (X⁻¹·X_d).log()                      tool frame, [v; ω]
/// e_twist = twist_d − J·q̇
/// τ       = Jᵀ·(k⊙e + d⊙e_twist) + C(q,q̇)·q̇ + G(q) + damping⊙q̇ + friction_loss⊙sign(q̇)
/// ```
///
/// The apparent inertia at the tool is the model's own `Λ = (J·H⁻¹·Jᵀ)⁻¹`; reshaping it needs a
/// wrist force sensor, which this crate has no notion of.
///
/// [`JacobianFrame::Body`] — the default — makes the stiffness axes tool-fixed, what a contact task
/// wants: soft along the approach direction, stiff laterally. [`JacobianFrame::World`] makes them
/// base-fixed. Both Jacobians are taken at the end-effector origin and differ by a rotation, so the
/// frame choice rotates the error twist and selects the matching Jacobian, nothing more.
///
/// ```
/// use multicalc::control::{CartesianImpedanceController, CartesianReference};
/// use multicalc::dynamics::ArticulatedBody;
/// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
/// use multicalc::linear_algebra::{Matrix, Vector};
/// use multicalc::spatial::{SE3, SO3, SpatialInertia};
///
/// let axis = Vector::new([0.0, 1.0, 0.0]);
/// let link_offset = SE3::from_parts(SO3::<f64>::identity(), Vector::new([0.3, 0.0, 0.0]));
/// let tree = KinematicTree::<3, 3, f64>::try_from_joints(
///     &[
///         Joint::revolute(axis, SE3::identity()),
///         Joint::revolute(axis, link_offset),
///         Joint::revolute(axis, link_offset),
///     ],
///     &[
///         JointParent::World,
///         JointParent::Joint(0),
///         JointParent::Joint(1),
///     ],
/// )
/// .unwrap();
/// let link = SpatialInertia::new(
///     1.0,
///     Vector::new([0.15, 0.0, 0.0]),
///     Matrix::from_diagonal([0.01, 0.01, 0.01]),
/// )
/// .unwrap();
/// let body = ArticulatedBody::new(
///     tree,
///     &[Some(link), Some(link), Some(link)],
///     Vector::new([0.0, 0.0, -9.81]),
/// )
/// .unwrap();
///
/// // Isotropic: 800 N/m in translation, 40 N·m/rad in rotation, damped at `2·sqrt(k)`.
/// let stiffness = Vector::new([800.0, 800.0, 800.0, 40.0, 40.0, 40.0]);
/// let damping = stiffness.map(|entry: f64| 2.0 * entry.sqrt());
/// let controller = CartesianImpedanceController::<3, f64>::new(stiffness, damping, 2).unwrap();
///
/// // A target 5 mm along world x of where the tool starts.
/// let start = Vector::new([0.2, -0.4, 0.3]);
/// let state = body.tree().forward_kinematics(&start).unwrap();
/// let here = state.pose(2).unwrap();
/// let target = SE3::from_parts(
///     here.rotation(),
///     here.translation() + Vector::new([0.005, 0.0, 0.0]),
/// );
/// let reference = CartesianReference::at_rest(target);
///
/// let timestep = 0.001;
/// let mut position = start;
/// let mut velocity = Vector::zeros();
/// for _ in 0..3000 {
///     let torque = controller
///         .torque_at(&body, &position, &velocity, &reference)
///         .unwrap();
///     let acceleration = body.forward_dynamics_at(&position, &velocity, &torque).unwrap();
///     velocity = velocity + acceleration.scale(timestep);
///     position = position + velocity.scale(timestep);
/// }
///
/// let landed = body.tree().forward_kinematics(&position).unwrap();
/// let gap = landed.pose(2).unwrap().translation() - target.translation();
/// assert!(gap.norm() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianImpedanceController<const MAX_CONFIG: usize, T: Numeric = f64> {
    stiffness: Vector6D<T>,
    damping: Vector6D<T>,
    tool_index: usize,
    frame: JacobianFrame,
    posture: Option<NullSpacePosture<MAX_CONFIG, T>>,
}

/// The null-space posture term's configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
struct NullSpacePosture<const MAX_CONFIG: usize, T: Numeric> {
    position: Vector<MAX_CONFIG, T>,
    position_gain: T,
    velocity_gain: T,
    joint_weights: Vector<MAX_CONFIG, T>,
    pseudo_inverse_damping: T,
}

impl<const MAX_CONFIG: usize, T: Numeric> CartesianImpedanceController<MAX_CONFIG, T> {
    /// Builds the controller from six-axis stiffness and damping (`[v; ω]` order: three
    /// translational then three rotational) and the tool slot they act at.
    ///
    /// Frame defaults to [`JacobianFrame::Body`] and no null-space posture term is configured.
    /// `tool_index` is not range-checked here — the tree is not present. An out-of-range slot
    /// surfaces at the first `torque` call as
    /// [`ControlError::Kinematics`] carrying
    /// [`KinematicsError::ToolIndexOutOfRange`](crate::error::KinematicsError::ToolIndexOutOfRange).
    ///
    /// Returns [`ControlError::NonFinite`] if any entry is not finite, or
    /// [`ControlError::NegativeGain`] if any entry is negative.
    pub fn new(
        stiffness: Vector6D<T>,
        damping: Vector6D<T>,
        tool_index: usize,
    ) -> Result<Self, ControlError> {
        if !stiffness.is_finite() || !damping.is_finite() {
            return Err(ControlError::NonFinite);
        }
        for axis in 0..6 {
            let spring = stiffness.get(axis).copied().unwrap_or(T::ZERO);
            let damper = damping.get(axis).copied().unwrap_or(T::ZERO);
            if spring < T::ZERO || damper < T::ZERO {
                return Err(ControlError::NegativeGain);
            }
        }
        Ok(Self {
            stiffness,
            damping,
            tool_index,
            frame: JacobianFrame::Body,
            posture: None,
        })
    }

    /// Chooses whether the stiffness axes are tool-fixed ([`JacobianFrame::Body`], the default) or
    /// base-fixed ([`JacobianFrame::World`]).
    #[inline]
    #[must_use]
    pub fn with_frame(mut self, frame: JacobianFrame) -> Self {
        self.frame = frame;
        self
    }

    /// Adds a null-space posture term pulling the configuration toward `posture` without
    /// disturbing the tool.
    ///
    /// Costs a damped pseudo-inverse per tick: a 6×n factorization, not a matrix product.
    ///
    /// Returns [`ControlError::NonFinite`] if any argument is not finite, or
    /// [`ControlError::NegativeGain`] if a gain, a joint weight or `pseudo_inverse_damping` is
    /// negative.
    pub fn with_null_space_posture(
        mut self,
        posture: Vector<MAX_CONFIG, T>,
        position_gain: T,
        velocity_gain: T,
        joint_weights: Vector<MAX_CONFIG, T>,
        pseudo_inverse_damping: T,
    ) -> Result<Self, ControlError> {
        if !posture.is_finite()
            || !joint_weights.is_finite()
            || !position_gain.is_finite()
            || !velocity_gain.is_finite()
            || !pseudo_inverse_damping.is_finite()
        {
            return Err(ControlError::NonFinite);
        }
        if position_gain < T::ZERO || velocity_gain < T::ZERO || pseudo_inverse_damping < T::ZERO {
            return Err(ControlError::NegativeGain);
        }
        for slot in 0..MAX_CONFIG {
            if joint_weights.get(slot).copied().unwrap_or(T::ZERO) < T::ZERO {
                return Err(ControlError::NegativeGain);
            }
        }
        self.posture = Some(NullSpacePosture {
            position: posture,
            position_gain,
            velocity_gain,
            joint_weights,
            pseudo_inverse_damping,
        });
        Ok(self)
    }

    /// The six-axis stiffness, `[v; ω]` order.
    #[inline]
    pub fn stiffness(&self) -> Vector6D<T> {
        self.stiffness
    }

    /// The six-axis damping, `[v; ω]` order.
    #[inline]
    pub fn damping(&self) -> Vector6D<T> {
        self.damping
    }

    /// The tool slot the spring-damper acts at.
    #[inline]
    #[must_use]
    pub fn tool_index(&self) -> usize {
        self.tool_index
    }

    /// The frame the stiffness axes are fixed in.
    #[inline]
    #[must_use]
    pub fn frame(&self) -> JacobianFrame {
        self.frame
    }

    /// The tool pose error `(X⁻¹·X_d).log()`, in the controller's configured frame.
    ///
    /// Errors: [`ControlError::Kinematics`] carrying
    /// [`KinematicsError::ToolIndexOutOfRange`](crate::error::KinematicsError::ToolIndexOutOfRange)
    /// if the tool slot is past the model's joint count.
    pub fn pose_error<const MAX_JOINTS: usize>(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        reference: &CartesianReference<T>,
    ) -> Result<Twist<T>, ControlError> {
        let pose = state.pose(self.tool_index).ok_or(ControlError::Kinematics(
            KinematicsError::ToolIndexOutOfRange,
        ))?;
        let error = Twist::from_vector(pose.inverse().compose(reference.pose).log());
        Ok(match self.frame {
            JacobianFrame::Body => error,
            JacobianFrame::World => {
                let rotation = pose.rotation();
                Twist::new(rotation.act(error.linear()), rotation.act(error.angular()))
            }
        })
    }

    /// Joint torque for already-solved poses.
    ///
    /// `measured_position` feeds only the null-space posture term; with no posture configured it is
    /// unused.
    ///
    /// Errors: [`ControlError::Kinematics`] from the tool slot or the Jacobian, and
    /// [`ControlError::Dynamics`] from the bias term.
    pub fn torque<const MAX_JOINTS: usize>(
        &self,
        body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        measured_position: &Vector<MAX_CONFIG, T>,
        measured_velocity: &Vector<MAX_CONFIG, T>,
        reference: &CartesianReference<T>,
    ) -> Result<Vector<MAX_CONFIG, T>, ControlError> {
        let error = self.pose_error(state, reference)?.to_vector();
        let jacobian = body
            .tree()
            .geometric_jacobian(state, self.tool_index, self.frame)?;
        let twist_error =
            reference.twist.to_vector() - jacobian.tool_twist(measured_velocity).to_vector();
        let wrench = Vector6D::from_fn(|axis| {
            self.stiffness.get(axis).copied().unwrap_or(T::ZERO)
                * error.get(axis).copied().unwrap_or(T::ZERO)
                + self.damping.get(axis).copied().unwrap_or(T::ZERO)
                    * twist_error.get(axis).copied().unwrap_or(T::ZERO)
        });

        let mut torque =
            jacobian.matrix().transpose() * wrench + body.bias_torque(state, measured_velocity)?;

        if let Some(posture) = &self.posture {
            let projector = jacobian
                .null_space_projector(&posture.joint_weights, posture.pseudo_inverse_damping)?;
            let posture_error = joint_position_error(body, &posture.position, measured_position);
            let desired = Vector::from_fn(|slot| {
                posture.position_gain * posture_error.get(slot).copied().unwrap_or(T::ZERO)
                    - posture.velocity_gain
                        * measured_velocity.get(slot).copied().unwrap_or(T::ZERO)
            });
            torque += projector * desired;
        }

        Ok(torque)
    }

    /// Joint torque for configuration `measured_position`.
    ///
    /// Solves the world poses first, then hands them to
    /// [`torque`](CartesianImpedanceController::torque).
    ///
    /// Errors: as [`forward_kinematics`](crate::kinematics::KinematicTree::forward_kinematics), via
    /// [`ControlError::Kinematics`], and as [`torque`](CartesianImpedanceController::torque).
    pub fn torque_at<const MAX_JOINTS: usize>(
        &self,
        body: &ArticulatedBody<MAX_JOINTS, MAX_CONFIG, T>,
        measured_position: &Vector<MAX_CONFIG, T>,
        measured_velocity: &Vector<MAX_CONFIG, T>,
        reference: &CartesianReference<T>,
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
