//! How each joint's rate moves a chosen frame on the robot.
#![deny(clippy::indexing_slicing)]

use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::spatial::Twist;

/// Which axes a Jacobian's rows are read in.
///
/// Both choices measure the motion of the same point — the chosen frame's origin — and differ only
/// in the axes the numbers are given along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianFrame {
    /// Rows are read in world axes, at the chosen frame's origin.
    World,
    /// Rows are read in the chosen frame's own axes.
    Body,
}

/// How each joint's rate moves a chosen frame on the robot.
///
/// One column per joint, saying how fast that frame's origin travels and how fast the frame turns
/// when that joint moves at unit rate. Joints that do not carry the frame — a sibling branch, a
/// later joint, a weld — get an all-zero column. Slots past the model's joint count are zero too,
/// so multiplying by a full-width vector of joint rates needs no masking.
///
/// Returned by
/// [`geometric_jacobian`](crate::kinematics::KinematicTree::geometric_jacobian) and carries the
/// joint count it was built for, so a query past that returns `None` rather than a column that
/// means nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicJacobian<const MAX_JOINTS: usize, T: Numeric = f64> {
    /// Six rows — three of straight-line motion, then three of turning — one column per joint.
    entries: Matrix<6, MAX_JOINTS, T>,
    /// How many columns belong to real joints.
    columns: usize,
    /// Which axes the rows are read in.
    frame: JacobianFrame,
}

impl<const MAX_JOINTS: usize, T: Numeric> KinematicJacobian<MAX_JOINTS, T> {
    /// Crate-private constructor: only a tree emits a Jacobian.
    #[inline]
    pub(crate) fn from_entries(
        entries: Matrix<6, MAX_JOINTS, T>,
        columns: usize,
        frame: JacobianFrame,
    ) -> Self {
        Self {
            entries,
            columns,
            frame,
        }
    }

    /// The six-row block itself, linear rows first.
    #[inline]
    pub fn matrix(&self) -> Matrix<6, MAX_JOINTS, T> {
        self.entries
    }

    /// Joint count the Jacobian was built for.
    #[inline]
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Whether the Jacobian holds no joint columns.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns == 0
    }

    /// Which axes the rows are read in.
    #[inline]
    #[must_use]
    pub fn frame(&self) -> JacobianFrame {
        self.frame
    }

    /// How joint `index` moves the frame at unit rate, or `None` past the joint count.
    #[must_use]
    pub fn column(&self, index: usize) -> Option<Twist<T>> {
        if index >= self.columns {
            return None;
        }
        let mut entries = [T::ZERO; 6];
        for (row, entry) in entries.iter_mut().enumerate() {
            *entry = *self.entries.get(row, index)?;
        }
        Some(Twist::from_array(entries))
    }

    /// How the frame moves when the joints run at `joint_velocities`.
    #[must_use]
    pub fn tool_twist(&self, joint_velocities: &Vector<MAX_JOINTS, T>) -> Twist<T> {
        Twist::from_vector(self.entries * *joint_velocities)
    }
}
