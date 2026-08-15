//! Kinematic tree: joints stored in topological order with parent indices.
#![deny(clippy::indexing_slicing)]

use crate::error::KinematicsError;
use crate::kinematics::joint::{Joint, JointKind, JointParent};
use crate::kinematics::kinematic_jacobian::{JacobianFrame, KinematicJacobian};
use crate::kinematics::kinematic_tree_state::KinematicTreeState;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::spatial::{FreeJointState, SE3, SO3};

/// A jointed robot model: joints in topological order, each attached to the world or to an earlier
/// joint.
///
/// Fixed-size storage — `MAX_JOINTS` slots plus a runtime length — so the model is `Copy`, needs no
/// heap, and can sit in flash. A parent index is always strictly below the joint's own, so forward
/// kinematics resolves in one sweep. A fixed joint still takes a joint slot, but only a floating
/// joint (always the first, if present) takes more than one configuration slot — see
/// [`KinematicTree::config_offset`].
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
/// let tree = KinematicTree::<2, 2, f64>::try_from_joints(
///     &[Joint::revolute(z, SE3::identity()), Joint::revolute(z, link)],
///     &[JointParent::World, JointParent::Joint(0)],
/// )
/// .unwrap();
///
/// assert_eq!(tree.len(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicTree<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric = f64> {
    /// Joints in topological order; indices past `length` are unused defaults.
    joints: [Joint<T>; MAX_JOINTS],
    /// Parent frame per joint, indexed alongside `joints`.
    parents: [JointParent; MAX_JOINTS],
    /// Live joint count.
    length: usize,
}

impl<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric> Default
    for KinematicTree<MAX_JOINTS, MAX_CONFIG, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_JOINTS: usize, const MAX_CONFIG: usize, T: Numeric>
    KinematicTree<MAX_JOINTS, MAX_CONFIG, T>
{
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

    /// Whether the tree's first joint is floating (its reading is seven numbers, not one).
    #[inline]
    #[must_use]
    pub fn has_floating_base(&self) -> bool {
        self.length > 0
            && self
                .joints
                .first()
                .is_some_and(|joint| joint.kind() == JointKind::Floating)
    }

    /// Where joint `index`'s reading starts in a `MAX_CONFIG`-wide configuration vector, or `None`
    /// past the joint count.
    ///
    /// A floating joint — only ever the first — takes seven slots; every other joint takes one, so
    /// a floating base shifts every later joint's slot by six.
    #[inline]
    #[must_use]
    pub fn config_offset(&self, index: usize) -> Option<usize> {
        if index >= self.length {
            return None;
        }
        if self.has_floating_base() && index > 0 {
            Some(index + 6)
        } else {
            Some(index)
        }
    }

    /// Where joint `index`'s rate starts in a `MAX_CONFIG`-wide velocity vector, or `None` past
    /// the joint count.
    ///
    /// A floating joint takes six slots; every other joint takes one, so a floating base shifts
    /// every later joint's slot by five.
    #[inline]
    #[must_use]
    pub fn velocity_offset(&self, index: usize) -> Option<usize> {
        if index >= self.length {
            return None;
        }
        if self.has_floating_base() && index > 0 {
            Some(index + 5)
        } else {
            Some(index)
        }
    }

    /// How many configuration numbers the model uses: one per joint, six more if the first is
    /// floating.
    #[inline]
    #[must_use]
    pub fn config_len(&self) -> usize {
        if self.length == 0 {
            0
        } else if self.has_floating_base() {
            self.length + 6
        } else {
            self.length
        }
    }

    /// How many velocity numbers the model uses: one per joint, five more if the first is
    /// floating.
    #[inline]
    #[must_use]
    pub fn velocity_len(&self) -> usize {
        if self.length == 0 {
            0
        } else if self.has_floating_base() {
            self.length + 5
        } else {
            self.length
        }
    }

    /// Appends a joint attached to `parent`, normalizing a movable joint's axis before storage.
    ///
    /// The model's only fallible operation: validated here once, every query afterwards is total.
    ///
    /// Errors: [`CapacityExceeded`](KinematicsError::CapacityExceeded) if the model is full,
    /// [`ParentOutOfOrder`](KinematicsError::ParentOutOfOrder) if `parent` is not an earlier joint,
    /// [`FloatingJointNotAtRoot`](KinematicsError::FloatingJointNotAtRoot) if a floating joint is
    /// not the tree's first joint attached to the world,
    /// [`NonFinite`](KinematicsError::NonFinite) on any non-finite model value,
    /// [`LimitsReversed`](KinematicsError::LimitsReversed) if the lower limit exceeds the upper,
    /// [`AxisHasNoDirection`](KinematicsError::AxisHasNoDirection) on a zero axis, or
    /// [`ConfigurationCapacityExceeded`](KinematicsError::ConfigurationCapacityExceeded) if the
    /// joint's reading does not fit the remaining configuration capacity.
    pub fn push(&mut self, joint: Joint<T>, parent: JointParent) -> Result<(), KinematicsError> {
        if self.length == MAX_JOINTS {
            return Err(KinematicsError::CapacityExceeded);
        }
        // The new joint takes index `self.length`, so a valid parent index is strictly below it.
        if matches!(parent, JointParent::Joint(index) if index >= self.length) {
            return Err(KinematicsError::ParentOutOfOrder);
        }
        if joint.kind() == JointKind::Floating && (self.length != 0 || parent != JointParent::World)
        {
            return Err(KinematicsError::FloatingJointNotAtRoot);
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
        if joint.kind() == JointKind::Continuous && joint.limits().is_some() {
            return Err(KinematicsError::ContinuousJointHasLimits);
        }

        let joint = match joint.kind() {
            JointKind::Revolute | JointKind::Prismatic | JointKind::Continuous => joint.with_axis(
                joint
                    .axis()
                    .try_normalized()
                    .ok_or(KinematicsError::AxisHasNoDirection)?,
            ),
            JointKind::Fixed | JointKind::Floating => joint,
        };

        let added_config_width = if joint.kind() == JointKind::Floating {
            7
        } else {
            1
        };
        if self.config_len() + added_config_width > MAX_CONFIG {
            return Err(KinematicsError::ConfigurationCapacityExceeded);
        }

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
        configuration: &Vector<MAX_CONFIG, T>,
    ) -> Result<KinematicTreeState<MAX_JOINTS, T>, KinematicsError> {
        for index in 0..self.length {
            let joint = self
                .joints
                .get(index)
                .ok_or(KinematicsError::CapacityExceeded)?;
            let offset = self
                .config_offset(index)
                .ok_or(KinematicsError::CapacityExceeded)?;
            let width = if joint.kind() == JointKind::Floating {
                7
            } else {
                1
            };
            for slot in 0..width {
                let reading = configuration
                    .get(offset + slot)
                    .ok_or(KinematicsError::NonFinite)?;
                if joint.kind() != JointKind::Fixed && !reading.is_finite() {
                    return Err(KinematicsError::NonFinite);
                }
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
            let offset = self
                .config_offset(index)
                .ok_or(KinematicsError::CapacityExceeded)?;

            let local = match joint.kind() {
                // Rotation about an axis through the anchor:
                // translate(anchor) · rotate · translate(-anchor), composed out.
                JointKind::Revolute | JointKind::Continuous => {
                    let reading = *configuration
                        .get(offset)
                        .ok_or(KinematicsError::NonFinite)?;
                    let displacement = reading - joint.zero_offset();
                    let rotation = SO3::exp(joint.axis().scale(displacement));
                    let anchor = joint.anchor();
                    SE3::from_parts(rotation, anchor - rotation.act(anchor))
                }
                JointKind::Prismatic => {
                    let reading = *configuration
                        .get(offset)
                        .ok_or(KinematicsError::NonFinite)?;
                    let displacement = reading - joint.zero_offset();
                    SE3::from_parts(SO3::identity(), joint.axis().scale(displacement))
                }
                JointKind::Fixed => SE3::identity(),
                JointKind::Floating => {
                    let mut place = [T::ZERO; 7];
                    for (slot, value) in place.iter_mut().enumerate() {
                        *value = *configuration
                            .get(offset + slot)
                            .ok_or(KinematicsError::NonFinite)?;
                    }
                    FreeJointState::from_generalized_vectors(place, [T::ZERO; 6])
                        .ok_or(KinematicsError::OrientationHasNoDirection)?
                        .pose()
                }
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
    ///
    /// ```
    /// use multicalc::kinematics::{JacobianFrame, Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
    ///
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
    /// // One forward sweep, then a Jacobian for each frame that hangs off it.
    /// let state = tree.forward_kinematics(&Vector::zeros()).unwrap();
    /// let at_elbow = tree
    ///     .geometric_jacobian(&state, 1, JacobianFrame::World)
    ///     .unwrap();
    /// let at_tool = tree
    ///     .geometric_jacobian(&state, 2, JacobianFrame::World)
    ///     .unwrap();
    ///
    /// // The shoulder's moment arm is 1 m to the elbow and 2 m to the tool.
    /// assert!((at_elbow.column(0).unwrap().linear() - Vector::new([0.0, 1.0, 0.0])).norm() < 1e-12);
    /// assert!((at_tool.column(0).unwrap().linear() - Vector::new([0.0, 2.0, 0.0])).norm() < 1e-12);
    ///
    /// // The elbow cannot move itself, so its own column is zero there.
    /// assert!(at_elbow.column(1).unwrap().linear().norm() < 1e-12);
    /// ```
    pub fn geometric_jacobian(
        &self,
        state: &KinematicTreeState<MAX_JOINTS, T>,
        tool_index: usize,
        frame: JacobianFrame,
    ) -> Result<KinematicJacobian<MAX_CONFIG, T>, KinematicsError> {
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
        let mut entries = Matrix::<6, MAX_CONFIG, T>::zeros();
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
            if joint.kind() == JointKind::Fixed {
                continue;
            }
            // The stored pose is the one after this joint has moved, and reading it there is still
            // right: a joint's own turn leaves its own axis, and any point on that axis, exactly
            // where they were, and a joint's own slide leaves its own direction where it was. So one
            // forward sweep gives everything the columns need.
            let pose = state
                .pose(index)
                .ok_or(KinematicsError::StateShapeMismatch)?;
            let velocity_offset = self
                .velocity_offset(index)
                .ok_or(KinematicsError::CapacityExceeded)?;

            // One (linear, angular) pair per column this joint contributes: one for a revolute or
            // prismatic joint, six for a floating one — three slide directions along its own
            // axes, then three turn directions about its own origin.
            let mut columns: [Option<(Vector<3, T>, Vector<3, T>)>; 6] = [None; 6];
            match joint.kind() {
                // Turning sweeps the tool frame around the axis, faster the further out it sits.
                JointKind::Revolute | JointKind::Continuous => {
                    let axis_in_world = pose.rotation().act(joint.axis());
                    let axis_point = pose.act(joint.anchor());
                    columns[0] =
                        Some((axis_in_world.cross(tool_origin - axis_point), axis_in_world));
                }
                // Sliding carries the tool frame along the axis without turning it.
                JointKind::Prismatic => {
                    let axis_in_world = pose.rotation().act(joint.axis());
                    columns[0] = Some((axis_in_world, Vector::zeros()));
                }
                JointKind::Floating => {
                    let joint_origin = pose.translation();
                    for local_axis in 0..3 {
                        let basis = Vector::<3, T>::from_fn(|row| {
                            if row == local_axis { T::ONE } else { T::ZERO }
                        });
                        // Sliding along a world axis carries every point in the subtree, tool
                        // included, by the same amount, in that fixed world direction — no
                        // turning. Unlike every other joint's own axis, this one is not carried
                        // by the joint's own orientation: a floating joint's slide reading is a
                        // world position directly (MuJoCo's own free-joint convention), so world
                        // axis 0 always means world axis 0, whichever way the body is turned.
                        if let Some(slot) = columns.get_mut(local_axis) {
                            *slot = Some((basis, Vector::zeros()));
                        }
                        // Turning about this axis sweeps the tool around the joint's own origin,
                        // the same formula a revolute joint's single axis already uses. The turn
                        // reading, unlike the slide, is carried by the joint's own orientation.
                        let direction_in_world = pose.rotation().act(basis);
                        if let Some(slot) = columns.get_mut(3 + local_axis) {
                            *slot = Some((
                                direction_in_world.cross(tool_origin - joint_origin),
                                direction_in_world,
                            ));
                        }
                    }
                }
                JointKind::Fixed => unreachable!("handled above"),
            }

            for (slot, column) in columns.iter().enumerate() {
                let Some((mut linear, mut angular)) = *column else {
                    continue;
                };
                if frame == JacobianFrame::Body {
                    linear = into_tool_axes.act(linear);
                    angular = into_tool_axes.act(angular);
                }
                let linear_rows = *linear.as_array();
                let angular_rows = *angular.as_array();
                for (row, value) in linear_rows.iter().chain(angular_rows.iter()).enumerate() {
                    *entries
                        .get_mut(row, velocity_offset + slot)
                        .ok_or(KinematicsError::CapacityExceeded)? = *value;
                }
            }
        }

        Ok(KinematicJacobian::from_entries(
            entries,
            self.velocity_len(),
            frame,
        ))
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
        configuration: &Vector<MAX_CONFIG, T>,
        tool_index: usize,
        frame: JacobianFrame,
    ) -> Result<KinematicJacobian<MAX_CONFIG, T>, KinematicsError> {
        let state = self.forward_kinematics(configuration)?;
        self.geometric_jacobian(&state, tool_index, frame)
    }

    /// Configuration-space distance between `a` and `b`: linear for `Revolute`/`Prismatic`, the
    /// shortest angular difference (wrapped mod 2π via [`Numeric::wrap_to_pi`]) for `Continuous`,
    /// an `SE3::log()`-based pose distance for `Floating`, and no contribution from a `Fixed`
    /// joint. Squared per-joint terms are summed then square-rooted, mixing metres and radians the
    /// same way every other joint-space quantity in this module already does.
    ///
    /// Used to decide whether two solutions of the same target are the same branch, and to pick
    /// the branch closest to a previous configuration near a singularity.
    ///
    /// ```
    /// use core::f64::consts::PI;
    /// use multicalc::kinematics::{Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::SE3;
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let tree = KinematicTree::<1, 1, f64>::try_from_joints(
    ///     &[Joint::continuous(z, SE3::identity())],
    ///     &[JointParent::World],
    /// )
    /// .unwrap();
    ///
    /// // 3 and -3 radians are only a shade over a quarter turn apart the short way, not 6 radians.
    /// let distance = tree.configuration_distance(&Vector::new([3.0]), &Vector::new([-3.0]));
    /// assert!((distance - (2.0 * PI - 6.0)).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn configuration_distance(
        &self,
        a: &Vector<MAX_CONFIG, T>,
        b: &Vector<MAX_CONFIG, T>,
    ) -> T {
        let mut total = T::ZERO;
        for index in 0..self.length {
            let Some(joint) = self.joint(index) else {
                continue;
            };
            let Some(offset) = self.config_offset(index) else {
                continue;
            };
            match joint.kind() {
                JointKind::Fixed => {}
                JointKind::Continuous => {
                    let Some(a_reading) = a.get(offset) else {
                        continue;
                    };
                    let Some(b_reading) = b.get(offset) else {
                        continue;
                    };
                    let difference = (*a_reading - *b_reading).wrap_to_pi();
                    total += difference * difference;
                }
                JointKind::Revolute | JointKind::Prismatic => {
                    let Some(a_reading) = a.get(offset) else {
                        continue;
                    };
                    let Some(b_reading) = b.get(offset) else {
                        continue;
                    };
                    let difference = *a_reading - *b_reading;
                    total += difference * difference;
                }
                JointKind::Floating => {
                    let mut a_place = [T::ZERO; 7];
                    let mut b_place = [T::ZERO; 7];
                    for (slot, value) in a_place.iter_mut().enumerate() {
                        *value = a.get(offset + slot).copied().unwrap_or(T::ZERO);
                    }
                    for (slot, value) in b_place.iter_mut().enumerate() {
                        *value = b.get(offset + slot).copied().unwrap_or(T::ZERO);
                    }
                    let Some(a_state) =
                        FreeJointState::from_generalized_vectors(a_place, [T::ZERO; 6])
                    else {
                        continue;
                    };
                    let Some(b_state) =
                        FreeJointState::from_generalized_vectors(b_place, [T::ZERO; 6])
                    else {
                        continue;
                    };
                    let position_difference =
                        a_state.pose().translation() - b_state.pose().translation();
                    total += position_difference.dot(position_difference);
                    let rotation_difference =
                        (a_state.pose().rotation().inverse() * b_state.pose().rotation()).log();
                    total += rotation_difference.dot(rotation_difference);
                }
            }
        }
        total.sqrt()
    }
}
