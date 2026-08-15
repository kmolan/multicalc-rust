//! Damped least-squares inverse kinematics with joint-limit clamping and null-space resolution.
#![deny(clippy::indexing_slicing)]

use crate::error::KinematicsError;
use crate::kinematics::joint::JointKind;
use crate::kinematics::kinematic_jacobian::{JacobianFrame, KinematicJacobian};
use crate::kinematics::kinematic_tree::KinematicTree;
use crate::linear_algebra::{Matrix, Vector};
use crate::ode::ExponentialMap;
use crate::scalar::Numeric;
use crate::spatial::{FreeJointState, SE3, Twist};

/// Why the solver stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InverseKinematicsTermination {
    /// End-effector pose is within both tolerances of the target.
    Converged,
    /// Steps fell below the minimum norm with the target unmet: saturated against a joint limit,
    /// at a singular configuration, or the target is outside the workspace.
    Stalled,
    /// Iteration budget exhausted with the target unmet.
    IterationBudget,
}

/// Solve result: the configuration reached, its residual task error, and the termination reason.
///
/// A non-converged solve still carries its best configuration, so a servo loop can command the
/// nearest pose achieved and arbitrate on the residual itself.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct InverseKinematicsReport<const MAX_CONFIG: usize, T: Numeric = f64> {
    /// Configuration the solver settled on.
    pub joint_positions: Vector<MAX_CONFIG, T>,
    /// Translational residual, metres.
    pub position_error: T,
    /// Rotational residual, radians.
    pub orientation_error: T,
    /// Iterations performed.
    pub iterations: usize,
    /// Termination reason.
    pub termination: InverseKinematicsTermination,
}

/// Secondary objective for a redundant chain, resolved in the null space of the task.
///
/// Applies only to the DOF the task leaves unconstrained, so it never perturbs the end-effector
/// pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecondaryObjective<const MAX_CONFIG: usize, T: Numeric = f64> {
    /// Leave the redundancy unresolved.
    None,
    /// Bias toward a reference configuration.
    PreferredPosture(Vector<MAX_CONFIG, T>),
    /// Bias toward the midpoint of each joint's range, maximizing limit margin.
    JointLimitMargin,
}

/// Damped least-squares inverse-kinematics solver, configured by a builder.
///
/// Iterates `Δq = J⁺_λ · e` on the body-frame pose error, with λ ramped in as the configuration
/// approaches a singularity so joint rates stay bounded where the chain loses a task-space DOF.
/// Steps are norm-limited and clamped to joint limits each iteration.
///
/// The solver holds nothing but its gains, so construction is free.
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
pub struct InverseKinematics<const MAX_CONFIG: usize, T: Numeric = f64> {
    position_tolerance: T,
    orientation_tolerance: T,
    maximum_iterations: usize,
    singular_value_threshold: T,
    maximum_damping: T,
    maximum_step_norm: T,
    minimum_step_norm: T,
    respect_limits: bool,
    joint_weights: Vector<MAX_CONFIG, T>,
    secondary_objective: SecondaryObjective<MAX_CONFIG, T>,
    secondary_gain: T,
}

impl<const MAX_CONFIG: usize, T: Numeric> Default for InverseKinematics<MAX_CONFIG, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_CONFIG: usize, T: Numeric> InverseKinematics<MAX_CONFIG, T> {
    /// Default gains: 1e-6 m and 1e-6 rad tolerances, 100-iteration budget, damping ramping in
    /// below σ_min = 1e-3 and reaching λ = 1e-2 at a full singularity, step norm capped at 0.2,
    /// joint limits enforced, unit joint weights, no secondary objective.
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

    /// Sets the translational convergence tolerance, metres.
    #[must_use]
    pub fn with_position_tolerance(mut self, position_tolerance: T) -> Self {
        self.position_tolerance = position_tolerance;
        self
    }

    /// Sets the rotational convergence tolerance, radians.
    #[must_use]
    pub fn with_orientation_tolerance(mut self, orientation_tolerance: T) -> Self {
        self.orientation_tolerance = orientation_tolerance;
        self
    }

    /// Sets the iteration budget.
    #[must_use]
    pub fn with_maximum_iterations(mut self, maximum_iterations: usize) -> Self {
        self.maximum_iterations = maximum_iterations;
        self
    }

    /// Sets the σ_min below which damping ramps in.
    #[must_use]
    pub fn with_singular_value_threshold(mut self, singular_value_threshold: T) -> Self {
        self.singular_value_threshold = singular_value_threshold;
        self
    }

    /// Sets λ at a full singularity.
    #[must_use]
    pub fn with_maximum_damping(mut self, maximum_damping: T) -> Self {
        self.maximum_damping = maximum_damping;
        self
    }

    /// Sets the per-iteration step-norm cap.
    #[must_use]
    pub fn with_maximum_step_norm(mut self, maximum_step_norm: T) -> Self {
        self.maximum_step_norm = maximum_step_norm;
        self
    }

    /// Sets the step norm below which the solve is declared stalled.
    #[must_use]
    pub fn with_minimum_step_norm(mut self, minimum_step_norm: T) -> Self {
        self.minimum_step_norm = minimum_step_norm;
        self
    }

    /// Sets whether joint limits are enforced.
    #[must_use]
    pub fn with_respect_limits(mut self, respect_limits: bool) -> Self {
        self.respect_limits = respect_limits;
        self
    }

    /// Sets the diagonal of `W`, one weight per joint.
    #[must_use]
    pub fn with_joint_weights(mut self, joint_weights: Vector<MAX_CONFIG, T>) -> Self {
        self.joint_weights = joint_weights;
        self
    }

    /// Sets the null-space secondary objective.
    #[must_use]
    pub fn with_secondary_objective(
        mut self,
        secondary_objective: SecondaryObjective<MAX_CONFIG, T>,
    ) -> Self {
        self.secondary_objective = secondary_objective;
        self
    }

    /// Sets the secondary-objective gain.
    #[must_use]
    pub fn with_secondary_gain(mut self, secondary_gain: T) -> Self {
        self.secondary_gain = secondary_gain;
        self
    }

    /// Solves for the configuration placing frame `tool_index` at `target`, seeded at `seed`.
    ///
    /// Non-convergence is not an error: the report carries the termination reason and the best
    /// configuration found, since the nearest achieved pose is still actionable in a servo loop.
    ///
    /// Errors: [`ToolIndexOutOfRange`](KinematicsError::ToolIndexOutOfRange) for an unknown frame,
    /// [`NonPositiveWeight`](KinematicsError::NonPositiveWeight) for a non-positive joint weight,
    /// [`NonFinite`](KinematicsError::NonFinite) for a non-finite target or seed, or
    /// [`Linalg`](KinematicsError::Linalg) if a decomposition fails.
    ///
    /// ```
    /// use multicalc::kinematics::{
    ///     InverseKinematics, InverseKinematicsTermination, Joint, JointParent, KinematicTree,
    /// };
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
    ///
    /// // Planar 2R arm, end-effector frame 1 m past the elbow.
    /// let tree = KinematicTree::<3, 3, f64>::try_from_joints(
    ///     &[
    ///         Joint::revolute(z, SE3::identity()),
    ///         Joint::revolute(z, link),
    ///         Joint::fixed(link),
    ///     ],
    ///     &[
    ///         JointParent::World,
    ///         JointParent::Joint(0),
    ///         JointParent::Joint(1),
    ///     ],
    /// )
    /// .unwrap();
    ///
    /// // Inside the workspace: reached with the elbow at a right angle.
    /// let target = SE3::from_parts(SO3::identity(), Vector::new([1.0, 1.0, 0.0]));
    /// let solver = InverseKinematics::<3, f64>::new();
    /// let report = solver
    ///     .solve(&tree, 2, target, &Vector::new([0.3, 0.3, 0.0]))
    ///     .unwrap();
    ///
    /// assert_eq!(report.termination, InverseKinematicsTermination::Converged);
    /// assert!(report.position_error < 1e-6);
    /// ```
    pub fn solve<const MAX_JOINTS: usize>(
        &self,
        tree: &KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
        tool_index: usize,
        target: SE3<T>,
        seed: &Vector<MAX_CONFIG, T>,
    ) -> Result<InverseKinematicsReport<MAX_CONFIG, T>, KinematicsError> {
        if tool_index >= tree.len() {
            return Err(KinematicsError::ToolIndexOutOfRange);
        }
        if !seed.is_finite()
            || !target.translation().is_finite()
            || target
                .rotation()
                .quaternion()
                .as_array()
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(KinematicsError::NonFinite);
        }
        for index in 0..tree.velocity_len() {
            let weight = self
                .joint_weights
                .get(index)
                .ok_or(KinematicsError::NonPositiveWeight)?;
            if *weight <= T::ZERO {
                return Err(KinematicsError::NonPositiveWeight);
            }
        }

        let mut configuration = *seed;
        self.hold_inside_limits(tree, &mut configuration);

        let mut position_error = T::ZERO;
        let mut orientation_error = T::ZERO;
        let mut iterations = 0;

        for _ in 0..self.maximum_iterations {
            iterations += 1;
            let state = tree.forward_kinematics(&configuration)?;
            let current = state
                .pose(tool_index)
                .ok_or(KinematicsError::ToolIndexOutOfRange)?;

            // Body-frame pose error: log(T_current⁻¹ · T_target). The Jacobian below is taken in
            // the same frame so error and columns agree.
            let task_error = (current.inverse() * target).log();
            (position_error, orientation_error) = split_error(task_error);

            if position_error <= self.position_tolerance
                && orientation_error <= self.orientation_tolerance
            {
                return Ok(InverseKinematicsReport {
                    joint_positions: configuration,
                    position_error,
                    orientation_error,
                    iterations,
                    termination: InverseKinematicsTermination::Converged,
                });
            }

            let jacobian = tree.geometric_jacobian(&state, tool_index, JacobianFrame::Body)?;
            let bias = secondary_bias(
                &self.secondary_objective,
                tree,
                &configuration,
                self.secondary_gain,
            );
            let step = limited_step(
                joint_step(
                    &jacobian,
                    task_error,
                    bias,
                    &self.joint_weights,
                    self.singular_value_threshold,
                    self.maximum_damping,
                )?,
                tree,
                self.maximum_step_norm,
            );

            let mut next = configuration;
            for index in 0..tree.len() {
                let Some(joint) = tree.joint(index) else {
                    continue;
                };
                let Some(co) = tree.config_offset(index) else {
                    continue;
                };
                let Some(vo) = tree.velocity_offset(index) else {
                    continue;
                };

                if joint.kind() == JointKind::Floating {
                    let mut place = [T::ZERO; 7];
                    for (slot, value) in place.iter_mut().enumerate() {
                        *value = configuration.get(co + slot).copied().unwrap_or(T::ZERO);
                    }
                    let Some(current_free) =
                        FreeJointState::from_generalized_vectors(place, [T::ZERO; 6])
                    else {
                        continue;
                    };
                    let d_linear = Vector::<3, T>::from_fn(|row| {
                        step.get(vo + row).copied().unwrap_or(T::ZERO)
                    });
                    let d_angular = Vector::<3, T>::from_fn(|row| {
                        step.get(vo + 3 + row).copied().unwrap_or(T::ZERO)
                    });

                    // d_linear is already a world-frame slide (matching MuJoCo's own free-joint
                    // convention and the Jacobian's translational columns, which are the constant
                    // world basis, not carried by the joint's own orientation), so it adds to a
                    // world position directly.
                    let new_position = current_free.pose().translation() + d_linear;
                    let new_orientation = ExponentialMap::attitude_step(
                        current_free.pose().rotation(),
                        d_angular,
                        T::ONE,
                    );
                    let new_free = FreeJointState::new(
                        SE3::from_parts(new_orientation, new_position),
                        Twist::zeros(),
                    );
                    for (slot, value) in new_free.generalized_position().iter().enumerate() {
                        if let Some(dst) = next.get_mut(co + slot) {
                            *dst = *value;
                        }
                    }
                } else if let (Some(dst), Some(change)) = (next.get_mut(co), step.get(vo)) {
                    *dst += *change;
                }
            }
            self.hold_inside_limits(tree, &mut next);

            // Measured on the realized step, so motion clipped by a limit counts as no progress.
            if (next - configuration).norm() <= self.minimum_step_norm {
                return Ok(InverseKinematicsReport {
                    joint_positions: configuration,
                    position_error,
                    orientation_error,
                    iterations,
                    termination: InverseKinematicsTermination::Stalled,
                });
            }
            configuration = next;
        }

        // Re-evaluate at the final configuration so the residuals match the returned q.
        if iterations > 0 {
            let state = tree.forward_kinematics(&configuration)?;
            let current = state
                .pose(tool_index)
                .ok_or(KinematicsError::ToolIndexOutOfRange)?;
            (position_error, orientation_error) = split_error((current.inverse() * target).log());
        }

        Ok(InverseKinematicsReport {
            joint_positions: configuration,
            position_error,
            orientation_error,
            iterations,
            termination: InverseKinematicsTermination::IterationBudget,
        })
    }

    /// Clamps the configuration into the joint limits, when limit enforcement is on.
    fn hold_inside_limits<const MAX_JOINTS: usize>(
        &self,
        tree: &KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
        configuration: &mut Vector<MAX_CONFIG, T>,
    ) {
        if !self.respect_limits {
            return;
        }
        for index in 0..tree.len() {
            let Some(joint) = tree.joint(index) else {
                continue;
            };
            if joint.kind() == JointKind::Floating {
                continue;
            }
            let Some((lower, upper)) = joint.limits() else {
                continue;
            };
            let Some(co) = tree.config_offset(index) else {
                continue;
            };
            if let Some(slot) = configuration.get_mut(co) {
                *slot = slot.max(lower).min(upper);
            }
        }
    }
}

/// Splits a `[v; ω]` task error into its translational and rotational norms.
fn split_error<T: Numeric>(task_error: Vector<6, T>) -> (T, T) {
    let part = |offset: usize| {
        Vector::<3, T>::from_fn(|row| task_error.get(offset + row).copied().unwrap_or(T::ZERO))
            .norm()
    };
    (part(0), part(3))
}

/// One iteration's joint displacement: the damped least-squares task step plus the null-space
/// term.
///
/// λ² ramps in as σ_min falls below the threshold, bounding the joint rates a near-degenerate
/// task direction would otherwise demand.
fn joint_step<const MAX_CONFIG: usize, T: Numeric>(
    jacobian: &KinematicJacobian<MAX_CONFIG, T>,
    task_error: Vector<6, T>,
    secondary_bias: Vector<MAX_CONFIG, T>,
    joint_weights: &Vector<MAX_CONFIG, T>,
    singular_value_threshold: T,
    maximum_damping: T,
) -> Result<Vector<MAX_CONFIG, T>, KinematicsError> {
    let entries = jacobian.matrix();
    let weighted_transpose = jacobian.weighted_transpose(joint_weights)?;

    // Going through the Gram matrix keeps the decomposition a fixed 6×6 whatever MAX_CONFIG is:
    // `svd` needs rows ≥ cols, which any redundant arm's wide J violates. Also correct for an
    // under-actuated chain, and cheap enough for a servo rate.
    let base = entries * weighted_transpose;

    // σ(J W⁻¹ Jᵀ) = σ(J)² up to the weighting; values come back descending, so index 5 is σ_min.
    let smallest = base
        .svd()?
        .singular_values()
        .get(5)
        .copied()
        .unwrap_or(T::ZERO)
        .max(T::ZERO)
        .sqrt();
    let damping_squared = if smallest >= singular_value_threshold {
        T::ZERO
    } else {
        let ratio = smallest / singular_value_threshold;
        maximum_damping * maximum_damping * (T::ONE - ratio * ratio)
    };

    let damped = base + Matrix::<6, 6, T>::identity().scale(damping_squared);
    let damped_svd = damped.svd()?;

    // One decomposition serves both terms, and the null-space projector is applied rather than
    // formed, so nothing MAX_CONFIG × MAX_CONFIG is ever allocated.
    let task_step = weighted_transpose * damped_svd.solve(task_error);
    let projected = weighted_transpose * damped_svd.solve(entries * secondary_bias);
    let null_step = secondary_bias - projected;

    Ok(task_step + null_step)
}

/// The secondary objective's desired joint-space velocity, before null-space projection. Zero at
/// fixed and floating joints (a floating joint has no scalar limit or preferred-posture reading in
/// the sense these two objectives mean) and at slots past the tree's joint count.
fn secondary_bias<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric>(
    objective: &SecondaryObjective<MAX_CONFIG, T>,
    tree: &KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
    configuration: &Vector<MAX_CONFIG, T>,
    gain: T,
) -> Vector<MAX_CONFIG, T> {
    let mut bias = Vector::<MAX_CONFIG, T>::zeros();
    if matches!(objective, SecondaryObjective::None) {
        return bias;
    }

    for index in 0..tree.len() {
        let Some(joint) = tree.joint(index) else {
            continue;
        };
        if joint.kind() == JointKind::Fixed || joint.kind() == JointKind::Floating {
            continue;
        }
        let Some(co) = tree.config_offset(index) else {
            continue;
        };
        let Some(vo) = tree.velocity_offset(index) else {
            continue;
        };
        let Some(reading) = configuration.get(co).copied() else {
            continue;
        };

        let wanted = match objective {
            SecondaryObjective::None => continue,
            SecondaryObjective::PreferredPosture(reference) => match reference.get(co).copied() {
                // Unbounded joint: shortest-arc error, not the raw difference.
                Some(preferred) if joint.kind() == JointKind::Continuous => {
                    gain * (preferred - reading).wrap_to_pi()
                }
                Some(preferred) => gain * (preferred - reading),
                None => continue,
            },
            SecondaryObjective::JointLimitMargin => match joint.limits() {
                Some((lower, upper)) if upper > lower => {
                    let midpoint = (lower + upper) * T::HALF;
                    // Normalized by range, so a short-travel joint is not driven as hard as a
                    // full-revolution one.
                    gain * (midpoint - reading) / (upper - lower)
                }
                _ => continue,
            },
        };

        if let Some(slot) = bias.get_mut(vo) {
            *slot = wanted;
        }
    }
    bias
}

/// Masks the step to actuated joints and scales it back to the norm cap.
///
/// The mask matters: a fixed joint's Jacobian column is zero, so the null-space term is otherwise
/// free to put a nonzero value in a slot that commands nothing.
fn limited_step<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric>(
    step: Vector<MAX_CONFIG, T>,
    tree: &KinematicTree<MAX_JOINTS, MAX_CONFIG, T>,
    maximum_step_norm: T,
) -> Vector<MAX_CONFIG, T> {
    let mut masked = Vector::<MAX_CONFIG, T>::zeros();
    for index in 0..tree.len() {
        let Some(joint) = tree.joint(index) else {
            continue;
        };
        if joint.kind() == JointKind::Fixed {
            continue;
        }
        let Some(vo) = tree.velocity_offset(index) else {
            continue;
        };
        let width = if joint.kind() == JointKind::Floating {
            6
        } else {
            1
        };
        for offset in 0..width {
            if let (Some(dst), Some(src)) = (masked.get_mut(vo + offset), step.get(vo + offset)) {
                *dst = *src;
            }
        }
    }

    let norm = masked.norm();
    if norm > maximum_step_norm && norm > T::ZERO {
        return masked.scale(maximum_step_norm / norm);
    }
    masked
}
