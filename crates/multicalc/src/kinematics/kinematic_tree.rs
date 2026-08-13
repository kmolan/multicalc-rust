//! Kinematic tree: joints stored in topological order with parent indices.
#![deny(clippy::indexing_slicing)]

use crate::error::KinematicsError;
use crate::kinematics::joint::{Joint, JointKind, JointParent};
use crate::kinematics::kinematic_jacobian::{JacobianFrame, KinematicJacobian};
use crate::kinematics::kinematic_tree_state::KinematicTreeState;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::spatial::{SE3, SO3};

/// A jointed robot model: joints in topological order, each attached to the world or to an earlier
/// joint.
///
/// Fixed-size storage — `MAX_JOINTS` slots plus a runtime length — so the model is `Copy`, needs no
/// heap, and can sit in flash. A parent index is always strictly below the joint's own, so forward
/// kinematics resolves in one sweep. A fixed joint still takes a slot, so joint index equals
/// configuration index.
///
/// ```
/// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
/// use multicalc::linear_algebra::Vector;
/// use multicalc::spatial::{SE3, SO3};
///
/// let z = Vector::new([0.0, 0.0, 1.0]);
/// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
///
/// // Planar two-link arm: two revolute joints about z, unit link between them.
/// let tree = KinematicTree::<2, f64>::try_from_joints(
///     &[Joint::revolute(z, SE3::identity()), Joint::revolute(z, link)],
///     &[JointParent::World, JointParent::Joint(0)],
/// )
/// .unwrap();
///
/// assert_eq!(tree.len(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicTree<const MAX_JOINTS: usize, T: Numeric = f64> {
    /// Joints in topological order; indices past `length` are unused defaults.
    joints: [Joint<T>; MAX_JOINTS],
    /// Parent frame per joint, indexed alongside `joints`.
    parents: [JointParent; MAX_JOINTS],
    /// Live joint count.
    length: usize,
}

impl<const MAX_JOINTS: usize, T: Numeric> Default for KinematicTree<MAX_JOINTS, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_JOINTS: usize, T: Numeric> KinematicTree<MAX_JOINTS, T> {
    /// Maximum joint count.
    pub const CAPACITY: usize = MAX_JOINTS;

    /// An empty model.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            joints: [Joint::default(); MAX_JOINTS],
            parents: [JointParent::World; MAX_JOINTS],
            length: 0,
        }
    }

    /// Builds a model from joints and their parent frames, validating each in turn.
    ///
    /// Errors: [`JointCountMismatch`](KinematicsError::JointCountMismatch) if the slices differ in
    /// length, otherwise as [`push`](KinematicTree::push).
    pub fn try_from_joints(
        joints: &[Joint<T>],
        parents: &[JointParent],
    ) -> Result<Self, KinematicsError> {
        if joints.len() != parents.len() {
            return Err(KinematicsError::JointCountMismatch);
        }
        let mut tree = Self::new();
        for (joint, parent) in joints.iter().zip(parents.iter()) {
            tree.push(*joint, *parent)?;
        }
        Ok(tree)
    }

    /// Appends a joint attached to `parent`, normalizing a movable joint's axis before storage.
    ///
    /// The model's only fallible operation: validated here once, every query afterwards is total.
    ///
    /// Errors: [`CapacityExceeded`](KinematicsError::CapacityExceeded) if the model is full,
    /// [`ParentOutOfOrder`](KinematicsError::ParentOutOfOrder) if `parent` is not an earlier joint,
    /// [`NonFinite`](KinematicsError::NonFinite) on any non-finite model value,
    /// [`LimitsReversed`](KinematicsError::LimitsReversed) if the lower limit exceeds the upper, or
    /// [`AxisHasNoDirection`](KinematicsError::AxisHasNoDirection) on a zero axis.
    pub fn push(&mut self, joint: Joint<T>, parent: JointParent) -> Result<(), KinematicsError> {
        if self.length == MAX_JOINTS {
            return Err(KinematicsError::CapacityExceeded);
        }
        // The new joint takes index `self.length`, so a valid parent index is strictly below it.
        if matches!(parent, JointParent::Joint(index) if index >= self.length) {
            return Err(KinematicsError::ParentOutOfOrder);
        }

        let origin = joint.origin();
        let orientation = origin.rotation().quaternion().as_array();
        if !origin.translation().is_finite()
            || orientation.iter().any(|value| !value.is_finite())
            || !joint.axis().is_finite()
            || !joint.anchor().is_finite()
            || !joint.zero_offset().is_finite()
            || !joint.armature().is_finite()
            || !joint.damping().is_finite()
            || !joint.friction_loss().is_finite()
        {
            return Err(KinematicsError::NonFinite);
        }
        if let Some((lower, upper)) = joint.limits() {
            if !lower.is_finite() || !upper.is_finite() {
                return Err(KinematicsError::NonFinite);
            }
            if lower > upper {
                return Err(KinematicsError::LimitsReversed);
            }
        }

        let joint = match joint.kind() {
            JointKind::Revolute | JointKind::Prismatic => joint.with_axis(
                joint
                    .axis()
                    .try_normalized()
                    .ok_or(KinematicsError::AxisHasNoDirection)?,
            ),
            JointKind::Fixed => joint,
        };

        match (
            self.joints.get_mut(self.length),
            self.parents.get_mut(self.length),
        ) {
            (Some(joint_slot), Some(parent_slot)) => {
                *joint_slot = joint;
                *parent_slot = parent;
                self.length += 1;
                Ok(())
            }
            _ => Err(KinematicsError::CapacityExceeded),
        }
    }

    /// Joint count.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether the model has no joints.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The joint at `index`, or `None` past the joint count.
    #[inline]
    #[must_use]
    pub fn joint(&self, index: usize) -> Option<Joint<T>> {
        if index >= self.length {
            return None;
        }
        self.joints.get(index).copied()
    }

    /// The parent frame of the joint at `index`, or `None` past the joint count.
    #[inline]
    #[must_use]
    pub fn parent(&self, index: usize) -> Option<JointParent> {
        if index >= self.length {
            return None;
        }
        self.parents.get(index).copied()
    }

    /// World pose of every joint frame for configuration `joint_positions`.
    ///
    /// One reading per joint, in tree order; a fixed joint takes a slot and ignores it. Each pose is
    /// `parent · origin · joint transform`, resolved in a single forward sweep.
    ///
    /// Errors: [`NonFinite`](KinematicsError::NonFinite) on a non-finite reading at a movable joint.
    ///
    /// ```
    /// use core::f64::consts::FRAC_PI_2;
    /// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
    ///
    /// // Planar two-link arm, tool frame one unit past the elbow.
    /// let tree = KinematicTree::<3, f64>::try_from_joints(
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
    /// // Shoulder at +90°, elbow straight: tool at (0, 2, 0).
    /// let state = tree
    ///     .forward_kinematics(&Vector::new([FRAC_PI_2, 0.0, 0.0]))
    ///     .unwrap();
    /// let [x, y, z] = *state.pose(2).unwrap().translation().as_array();
    ///
    /// assert!(x.abs() < 1e-12 && (y - 2.0).abs() < 1e-12 && z.abs() < 1e-12);
    /// ```
    pub fn forward_kinematics(
        &self,
        joint_positions: &Vector<MAX_JOINTS, T>,
    ) -> Result<KinematicTreeState<MAX_JOINTS, T>, KinematicsError> {
        for index in 0..self.length {
            let joint = self
                .joints
                .get(index)
                .ok_or(KinematicsError::CapacityExceeded)?;
            let reading = joint_positions
                .get(index)
                .ok_or(KinematicsError::NonFinite)?;
            if joint.kind() != JointKind::Fixed && !reading.is_finite() {
                return Err(KinematicsError::NonFinite);
            }
        }

        let mut poses = [SE3::<T>::identity(); MAX_JOINTS];
        for index in 0..self.length {
            let joint = *self
                .joints
                .get(index)
                .ok_or(KinematicsError::CapacityExceeded)?;
            let parent_pose = match self.parents.get(index) {
                Some(JointParent::Joint(parent_index)) => *poses
                    .get(*parent_index)
                    .ok_or(KinematicsError::ParentOutOfOrder)?,
                _ => SE3::identity(),
            };
            let reading = *joint_positions
                .get(index)
                .ok_or(KinematicsError::NonFinite)?;
            let displacement = reading - joint.zero_offset();

            let local = match joint.kind() {
                // Rotation about an axis through the anchor:
                // translate(anchor) · rotate · translate(-anchor), composed out.
                JointKind::Revolute => {
                    let rotation = SO3::exp(joint.axis().scale(displacement));
                    let anchor = joint.anchor();
                    SE3::from_parts(rotation, anchor - rotation.act(anchor))
                }
                JointKind::Prismatic => {
                    SE3::from_parts(SO3::identity(), joint.axis().scale(displacement))
                }
                JointKind::Fixed => SE3::identity(),
            };

            *poses
                .get_mut(index)
                .ok_or(KinematicsError::CapacityExceeded)? = parent_pose * joint.origin() * local;
        }

        Ok(KinematicTreeState::from_poses(poses, self.length))
    }

    /// How each joint's rate moves the frame at `tool_index`, from already-solved poses.
    ///
    /// One column per joint, saying how fast that frame's origin travels and how fast it turns when
    /// that joint moves at unit rate. A joint only gets a nonzero column if the frame hangs off it,
    /// so a sibling branch, a later joint, and a weld all read as zero. `frame` chooses the axes the
    /// rows are read in.
    ///
    /// Takes the solved poses rather than a configuration, so a loop that already ran
    /// [`forward_kinematics`](KinematicTree::forward_kinematics) pays for one sweep, not two. Use
    /// [`geometric_jacobian_at`](KinematicTree::geometric_jacobian_at) to go straight from a
    /// configuration.
    ///
    /// Errors: [`ToolIndexOutOfRange`](KinematicsError::ToolIndexOutOfRange) if the model has no
    /// such joint, or [`StateShapeMismatch`](KinematicsError::StateShapeMismatch) if the poses were
    /// solved for a different joint count.
    pub fn geometric_jacobian(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        tool_index: usize,
        frame: JacobianFrame,
    ) -> Result<KinematicJacobian<MAX_JOINTS, T>, KinematicsError> {
        if tool_index >= self.length {
            return Err(KinematicsError::ToolIndexOutOfRange);
        }
        if state.len() != self.length {
            return Err(KinematicsError::StateShapeMismatch);
        }

        let tool_pose = state
            .pose(tool_index)
            .ok_or(KinematicsError::ToolIndexOutOfRange)?;
        let tool_origin = tool_pose.translation();

        // Mark the chain from the tool frame back to the world. Anything off that chain moves
        // without taking the tool frame with it. The joint count bounds the walk, so a malformed
        // model cannot circle forever.
        let mut carries_tool = [false; MAX_JOINTS];
        let mut current = tool_index;
        for _ in 0..MAX_JOINTS {
            *carries_tool
                .get_mut(current)
                .ok_or(KinematicsError::ParentOutOfOrder)? = true;
            match self.parents.get(current) {
                Some(JointParent::Joint(parent_index)) => current = *parent_index,
                _ => break,
            }
        }

        let into_tool_axes = tool_pose.rotation().inverse();
        let mut entries = Matrix::<6, MAX_JOINTS, T>::zeros();
        for index in 0..self.length {
            if !*carries_tool
                .get(index)
                .ok_or(KinematicsError::CapacityExceeded)?
            {
                continue;
            }
            let joint = *self
                .joints
                .get(index)
                .ok_or(KinematicsError::CapacityExceeded)?;
            // The stored pose is the one after this joint has moved, and reading it there is still
            // right: a joint's own turn leaves its own axis, and any point on that axis, exactly
            // where they were, and a joint's own slide leaves its own direction where it was. So one
            // forward sweep gives everything the columns need.
            let pose = state
                .pose(index)
                .ok_or(KinematicsError::StateShapeMismatch)?;
            let axis_in_world = pose.rotation().act(joint.axis());

            let (mut linear, mut angular) = match joint.kind() {
                // Turning sweeps the tool frame around the axis, faster the further out it sits.
                JointKind::Revolute => {
                    let axis_point = pose.act(joint.anchor());
                    (axis_in_world.cross(tool_origin - axis_point), axis_in_world)
                }
                // Sliding carries the tool frame along the axis without turning it.
                JointKind::Prismatic => (axis_in_world, Vector::zeros()),
                JointKind::Fixed => continue,
            };

            if frame == JacobianFrame::Body {
                linear = into_tool_axes.act(linear);
                angular = into_tool_axes.act(angular);
            }

            let linear_rows = *linear.as_array();
            let angular_rows = *angular.as_array();
            for (row, value) in linear_rows.iter().chain(angular_rows.iter()).enumerate() {
                *entries
                    .get_mut(row, index)
                    .ok_or(KinematicsError::CapacityExceeded)? = *value;
            }
        }

        Ok(KinematicJacobian::from_entries(entries, self.length, frame))
    }

    /// How each joint's rate moves the frame at `tool_index`, for configuration `joint_positions`.
    ///
    /// Solves the world poses first, then hands them to
    /// [`geometric_jacobian`](KinematicTree::geometric_jacobian).
    ///
    /// Errors: as [`forward_kinematics`](KinematicTree::forward_kinematics) and
    /// [`geometric_jacobian`](KinematicTree::geometric_jacobian).
    ///
    /// ```
    /// use multicalc::kinematics::{JacobianFrame, Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
    ///
    /// // Planar two-link arm, tool frame one unit past the elbow.
    /// let tree = KinematicTree::<3, f64>::try_from_joints(
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
    /// let jacobian = tree
    ///     .geometric_jacobian_at(&Vector::zeros(), 2, JacobianFrame::World)
    ///     .unwrap();
    ///
    /// // Stretched along x, the tool sits at (2, 0, 0). The shoulder is two units from it and the
    /// // elbow one, so turning the shoulder sweeps the tool sideways twice as fast.
    /// let shoulder = jacobian.column(0).unwrap();
    /// let elbow = jacobian.column(1).unwrap();
    /// let weld = jacobian.column(2).unwrap();
    ///
    /// assert!((shoulder.linear() - Vector::new([0.0, 2.0, 0.0])).norm() < 1e-12);
    /// assert!((elbow.linear() - Vector::new([0.0, 1.0, 0.0])).norm() < 1e-12);
    /// assert!((shoulder.angular() - Vector::new([0.0, 0.0, 1.0])).norm() < 1e-12);
    /// assert!((elbow.angular() - Vector::new([0.0, 0.0, 1.0])).norm() < 1e-12);
    /// assert!(weld.linear().norm() < 1e-12 && weld.angular().norm() < 1e-12);
    /// ```
    pub fn geometric_jacobian_at(
        &self,
        joint_positions: &Vector<MAX_JOINTS, T>,
        tool_index: usize,
        frame: JacobianFrame,
    ) -> Result<KinematicJacobian<MAX_JOINTS, T>, KinematicsError> {
        let state = self.forward_kinematics(joint_positions)?;
        self.geometric_jacobian(&state, tool_index, frame)
    }
}
