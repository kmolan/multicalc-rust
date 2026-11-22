//! Forward-kinematics output: world pose per joint frame.
#![deny(clippy::indexing_slicing)]

use crate::scalar::Numeric;
use crate::spatial::SE3;

/// World pose of every joint frame, for one configuration.
///
/// Returned by [`forward_kinematics`](crate::kinematics::KinematicTree::forward_kinematics) and
/// indexed by joint. Carries the joint count it was solved for, so a query past that returns `None`
/// rather than a stale pose from another model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicTreeState<const MAX_JOINTS: usize, T: Numeric = f64> {
    /// World pose per joint, in tree order; slots past `length` are identity.
    poses: [SE3<T>; MAX_JOINTS],
    /// Joint count solved for.
    length: usize,
}

impl<const MAX_JOINTS: usize, T: Numeric> KinematicTreeState<MAX_JOINTS, T> {
    /// Crate-private constructor: only a tree emits a state.
    #[inline]
    pub(crate) fn from_poses(poses: [SE3<T>; MAX_JOINTS], length: usize) -> Self {
        Self { poses, length }
    }

    /// Joint count solved for.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether the state holds no poses.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// World pose of joint `index`, or `None` past the joint count.
    #[inline]
    #[must_use]
    pub fn pose(&self, index: usize) -> Option<SE3<T>> {
        if index >= self.length {
            return None;
        }
        self.poses.get(index).copied()
    }

    /// The solved poses, in joint order.
    #[inline]
    #[must_use]
    pub fn poses(&self) -> &[SE3<T>] {
        self.poses.get(..self.length).unwrap_or(&[])
    }
}
