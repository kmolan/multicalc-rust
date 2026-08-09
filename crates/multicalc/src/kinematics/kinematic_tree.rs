//! A jointed model held as a parent-indexed tree of joints.
#![deny(clippy::indexing_slicing)]

use crate::error::KinematicsError;
use crate::kinematics::joint::{Joint, JointKind, JointParent};
use crate::scalar::Numeric;

/// A robot model: a set of joints, each attached to the world or to an earlier joint.
///
/// Storage is a fixed array of `MAX_JOINTS` joints plus a runtime length, so the model is
/// stack-allocated, `Copy`, and needs no heap. Every joint's parent must appear earlier in the
/// list, which is what lets forward kinematics resolve every pose in a single forward sweep. A
/// fixed joint still takes a slot, so joint index and configuration index agree.
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
///     &[
///         Joint::revolute(z, SE3::identity()),
///         Joint::revolute(z, link),
///     ],
///     &[JointParent::World, JointParent::Joint(0)],
/// )
/// .unwrap();
///
/// assert_eq!(tree.len(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicTree<const MAX_JOINTS: usize, T: Numeric = f64> {
    /// The joints, in tree order; indices past `length` hold defaults and are never read.
    joints: [Joint<T>; MAX_JOINTS],
    /// What each joint is attached to, indexed alongside `joints`.
    parents: [JointParent; MAX_JOINTS],
    /// How many joints the model actually has.
    length: usize,
}

impl<const MAX_JOINTS: usize, T: Numeric> Default for KinematicTree<MAX_JOINTS, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_JOINTS: usize, T: Numeric> KinematicTree<MAX_JOINTS, T> {
    /// How many joints the tree can hold.
    pub const CAPACITY: usize = MAX_JOINTS;

    /// A tree with no joints.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            joints: [Joint::default(); MAX_JOINTS],
            parents: [JointParent::World; MAX_JOINTS],
            length: 0,
        }
    }

    /// Builds a tree from a slice of joints and the frame each is attached to.
    ///
    /// Returns [`KinematicsError::JointCountMismatch`] if the two slices differ in length,
    /// [`KinematicsError::CapacityExceeded`] if more than `MAX_JOINTS` joints are supplied,
    /// [`KinematicsError::ParentOutOfOrder`] if a joint is attached to anything but an earlier
    /// joint, [`KinematicsError::NonFinite`] if any model value is not finite,
    /// [`KinematicsError::LimitsReversed`] if a joint's lower limit exceeds its upper, or
    /// [`KinematicsError::AxisHasNoDirection`] if a movable joint's axis is all zeros.
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

    /// Appends a joint attached to `parent`, validating it against the rest of the model.
    ///
    /// A movable joint's axis is normalized before it is stored, so an axis given at any scale is
    /// kept as a unit vector. This is the tree's only fallible operation: with the model checked
    /// once here, every query afterwards is total.
    ///
    /// Returns [`KinematicsError::CapacityExceeded`] if the tree is full,
    /// [`KinematicsError::ParentOutOfOrder`] if `parent` is not an earlier joint,
    /// [`KinematicsError::NonFinite`] if any model value is not finite,
    /// [`KinematicsError::LimitsReversed`] if the lower limit exceeds the upper, or
    /// [`KinematicsError::AxisHasNoDirection`] if a movable joint's axis is all zeros.
    pub fn push(&mut self, joint: Joint<T>, parent: JointParent) -> Result<(), KinematicsError> {
        if self.length == MAX_JOINTS {
            return Err(KinematicsError::CapacityExceeded);
        }
        // The joint being added takes index `self.length`, so an earlier parent is one below that.
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

    /// The number of joints in the model.
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

    /// The joint at `index`, or `None` past the end of the model.
    #[inline]
    #[must_use]
    pub fn joint(&self, index: usize) -> Option<Joint<T>> {
        if index >= self.length {
            return None;
        }
        self.joints.get(index).copied()
    }

    /// What the joint at `index` is attached to, or `None` past the end of the model.
    #[inline]
    #[must_use]
    pub fn parent(&self, index: usize) -> Option<JointParent> {
        if index >= self.length {
            return None;
        }
        self.parents.get(index).copied()
    }
}
