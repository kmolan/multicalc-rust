//! Working out the joint readings that put a chosen frame where you want it.
#![deny(clippy::indexing_slicing)]

use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// Why the solver stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InverseKinematicsTermination {
    /// The frame reached the target, inside both tolerances.
    Converged,
    /// Steps became too small to make progress while the target was still out of reach — the arm
    /// is against a limit, at a pose where it has lost a direction of motion, or the target is
    /// out of reach altogether.
    Stalled,
    /// The iteration budget ran out with the target still out of reach.
    IterationBudget,
}

/// The outcome of a solve: the readings found, how far off they left the frame, and why the solver
/// stopped.
///
/// A solve that did not converge still carries the best readings it found, so a control loop can
/// command the nearest pose it managed and decide for itself what to do about the shortfall.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct InverseKinematicsReport<const MAX_JOINTS: usize, T: Numeric = f64> {
    /// The joint readings the solver settled on.
    pub joint_positions: Vector<MAX_JOINTS, T>,
    /// How far the frame's origin is from the target, in metres.
    pub position_error: T,
    /// How far the frame is from being aimed at the target, in radians.
    pub orientation_error: T,
    /// How many passes the solver made.
    pub iterations: usize,
    /// Why the solver stopped.
    pub termination: InverseKinematicsTermination,
}

/// What the arm should do with the freedom a task leaves it.
///
/// An arm with more joints than the task needs can move without disturbing the frame it is holding.
/// This says what to do with that spare freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecondaryObjective<const MAX_JOINTS: usize, T: Numeric = f64> {
    /// Leave the spare freedom unused.
    None,
    /// Drift toward a chosen set of readings.
    PreferredPosture(Vector<MAX_JOINTS, T>),
    /// Drift toward the middle of each joint's travel.
    JointLimitMargin,
}

/// A solver that works out the joint readings putting a chosen frame at a target pose.
///
/// It steps from a starting guess toward the target, each step asking the Jacobian which joint
/// motions move the frame the right way. Near a pose where the arm has lost a direction of motion
/// the step is held back, so the readings stay sane instead of shooting off. Readings are kept
/// inside each joint's travel, and no single step is allowed to be large.
///
/// A solver is nothing but its settings, so building one is cheap — make one per call site and
/// configure it in place if that reads better.
///
/// ```
/// use multicalc::kinematics::{InverseKinematics, SecondaryObjective};
///
/// let _solver = InverseKinematics::<7, f64>::new()
///     .with_position_tolerance(1e-4)
///     .with_maximum_iterations(50)
///     .with_secondary_objective(SecondaryObjective::JointLimitMargin);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct InverseKinematics<const MAX_JOINTS: usize, T: Numeric = f64> {
    position_tolerance: T,
    orientation_tolerance: T,
    maximum_iterations: usize,
    singular_value_threshold: T,
    maximum_damping: T,
    maximum_step_norm: T,
    minimum_step_norm: T,
    respect_limits: bool,
    joint_weights: Vector<MAX_JOINTS, T>,
    secondary_objective: SecondaryObjective<MAX_JOINTS, T>,
    secondary_gain: T,
}

impl<const MAX_JOINTS: usize, T: Numeric> Default for InverseKinematics<MAX_JOINTS, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_JOINTS: usize, T: Numeric> InverseKinematics<MAX_JOINTS, T> {
    /// A solver with default settings: tolerances of 1e-6 m and 1e-6 rad, a budget of 100 passes,
    /// holding back the step once the smallest singular value drops below 1e-3 and holding it back
    /// by 1e-2 where a direction of motion is gone entirely, steps capped at 0.2, joint limits
    /// respected, every joint weighted equally, and no secondary objective.
    #[must_use]
    pub fn new() -> Self {
        Self {
            position_tolerance: T::from_f64(1e-6),
            orientation_tolerance: T::from_f64(1e-6),
            maximum_iterations: 100,
            singular_value_threshold: T::from_f64(1e-3),
            maximum_damping: T::from_f64(1e-2),
            maximum_step_norm: T::from_f64(0.2),
            minimum_step_norm: T::from_f64(1e-12),
            respect_limits: true,
            joint_weights: Vector::from_fn(|_| T::ONE),
            secondary_objective: SecondaryObjective::None,
            secondary_gain: T::from_f64(0.1),
        }
    }

    /// Sets how close the frame's origin must get, in metres.
    #[must_use]
    pub fn with_position_tolerance(mut self, position_tolerance: T) -> Self {
        self.position_tolerance = position_tolerance;
        self
    }

    /// Sets how closely the frame must be aimed, in radians.
    #[must_use]
    pub fn with_orientation_tolerance(mut self, orientation_tolerance: T) -> Self {
        self.orientation_tolerance = orientation_tolerance;
        self
    }

    /// Sets how many passes the solver may make before giving up.
    #[must_use]
    pub fn with_maximum_iterations(mut self, maximum_iterations: usize) -> Self {
        self.maximum_iterations = maximum_iterations;
        self
    }

    /// Sets how near to losing a direction of motion the arm has to be before steps are held back.
    #[must_use]
    pub fn with_singular_value_threshold(mut self, singular_value_threshold: T) -> Self {
        self.singular_value_threshold = singular_value_threshold;
        self
    }

    /// Sets how hard steps are held back where a direction of motion is gone entirely.
    #[must_use]
    pub fn with_maximum_damping(mut self, maximum_damping: T) -> Self {
        self.maximum_damping = maximum_damping;
        self
    }

    /// Sets the longest one step is allowed to be.
    #[must_use]
    pub fn with_maximum_step_norm(mut self, maximum_step_norm: T) -> Self {
        self.maximum_step_norm = maximum_step_norm;
        self
    }

    /// Sets the step length below which the solve is called stalled.
    #[must_use]
    pub fn with_minimum_step_norm(mut self, minimum_step_norm: T) -> Self {
        self.minimum_step_norm = minimum_step_norm;
        self
    }

    /// Sets whether every reading is kept inside its joint's travel.
    #[must_use]
    pub fn with_respect_limits(mut self, respect_limits: bool) -> Self {
        self.respect_limits = respect_limits;
        self
    }

    /// Sets how costly each joint is to move, one weight per joint.
    #[must_use]
    pub fn with_joint_weights(mut self, joint_weights: Vector<MAX_JOINTS, T>) -> Self {
        self.joint_weights = joint_weights;
        self
    }

    /// Sets what to do with the freedom the task leaves the arm.
    #[must_use]
    pub fn with_secondary_objective(
        mut self,
        secondary_objective: SecondaryObjective<MAX_JOINTS, T>,
    ) -> Self {
        self.secondary_objective = secondary_objective;
        self
    }

    /// Sets how hard the secondary objective pulls.
    #[must_use]
    pub fn with_secondary_gain(mut self, secondary_gain: T) -> Self {
        self.secondary_gain = secondary_gain;
        self
    }
}
