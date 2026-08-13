//! Geometric Jacobian of a kinematic tree: end-effector twist per unit joint rate.
#![deny(clippy::indexing_slicing)]

use crate::error::KinematicsError;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;
use crate::spatial::Twist;

/// Frame the Jacobian rows are expressed in.
///
/// Both are taken at the end-effector origin, so they differ by a rotation alone — there is no
/// translation term between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JacobianFrame {
    /// World-aligned, at the end-effector origin. This is the form MuJoCo's `mj_jac` returns,
    /// not the screw-theory spatial Jacobian taken at the world origin.
    World,
    /// Expressed in the end-effector frame: `blockdiag(Rᵀ, Rᵀ) · J_world`.
    Body,
}

/// Geometric Jacobian of a kinematic tree: a `6 × MAX_JOINTS` twist-per-unit-joint-rate map,
/// `[v; ω]` row order.
///
/// Columns for joints outside the end-effector's ancestor chain — sibling branches, later joints,
/// fixed joints — are zero, as are slots past the tree's joint count, so `J · q̇` needs no masking.
///
/// Returned by [`geometric_jacobian`](crate::kinematics::KinematicTree::geometric_jacobian) and
/// carrying its own active column count; queries past it return `None`.
///
/// ```
/// use multicalc::kinematics::{JacobianFrame, Joint, JointParent, KinematicTree};
/// use multicalc::linear_algebra::Vector;
/// use multicalc::spatial::{SE3, SO3};
///
/// let z = Vector::new([0.0, 0.0, 1.0]);
/// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
///
/// // Single revolute about z, end-effector 1 m out along x.
/// let tree = KinematicTree::<2, f64>::try_from_joints(
///     &[Joint::revolute(z, SE3::identity()), Joint::fixed(link)],
///     &[JointParent::World, JointParent::Joint(0)],
/// )
/// .unwrap();
///
/// let jacobian = tree
///     .geometric_jacobian_at(&Vector::zeros(), 1, JacobianFrame::World)
///     .unwrap();
///
/// // Unit rate about z at a 1 m moment arm: v = ω × r = (0, 1, 0), ω = (0, 0, 1).
/// let column = jacobian.column(0).unwrap();
/// assert!((column.linear() - Vector::new([0.0, 1.0, 0.0])).norm() < 1e-12);
/// assert!((column.angular() - Vector::new([0.0, 0.0, 1.0])).norm() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicJacobian<const MAX_JOINTS: usize, T: Numeric = f64> {
    /// `6 × MAX_JOINTS` block: linear rows 0–2, angular rows 3–5.
    entries: Matrix<6, MAX_JOINTS, T>,
    /// Active column count.
    columns: usize,
    /// Frame the rows are expressed in.
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

    /// The `6 × MAX_JOINTS` block, `[v; ω]` row order.
    #[inline]
    pub fn matrix(&self) -> Matrix<6, MAX_JOINTS, T> {
        self.entries
    }

    /// Active column count.
    #[inline]
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Whether there are no active columns.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns == 0
    }

    /// Frame the rows are expressed in.
    #[inline]
    #[must_use]
    pub fn frame(&self) -> JacobianFrame {
        self.frame
    }

    /// Column `index` as a twist, or `None` past the active columns.
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

    /// End-effector twist for joint rates `joint_velocities`: `J · q̇`.
    ///
    /// ```
    /// use multicalc::kinematics::{JacobianFrame, Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
    ///
    /// // Planar 2R arm, end-effector 1 m past the elbow, stretched along x.
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
    /// let jacobian = tree
    ///     .geometric_jacobian_at(&Vector::zeros(), 2, JacobianFrame::World)
    ///     .unwrap();
    ///
    /// // Moment arms of 2 m and 1 m, so driving both at unit rate sums to 3 m/s sideways and
    /// // 2 rad/s about z.
    /// let twist = jacobian.tool_twist(&Vector::new([1.0, 1.0, 0.0]));
    ///
    /// assert!((twist.linear() - Vector::new([0.0, 3.0, 0.0])).norm() < 1e-12);
    /// assert!((twist.angular() - Vector::new([0.0, 0.0, 2.0])).norm() < 1e-12);
    /// ```
    #[must_use]
    pub fn tool_twist(&self, joint_velocities: &Vector<MAX_JOINTS, T>) -> Twist<T> {
        Twist::from_vector(self.entries * *joint_velocities)
    }

    /// `W⁻¹ Jᵀ`, with `W = diag(joint_weights)`.
    ///
    /// Shared by every weighted-inverse path, so weight validation and inactive-slot masking sit
    /// in one place.
    ///
    /// Errors: [`NonPositiveWeight`](KinematicsError::NonPositiveWeight) on a non-positive weight
    /// in an active slot.
    pub(crate) fn weighted_transpose(
        &self,
        joint_weights: &Vector<MAX_JOINTS, T>,
    ) -> Result<Matrix<MAX_JOINTS, 6, T>, KinematicsError> {
        for index in 0..self.columns {
            let weight = joint_weights
                .get(index)
                .ok_or(KinematicsError::NonPositiveWeight)?;
            if *weight <= T::ZERO {
                return Err(KinematicsError::NonPositiveWeight);
            }
        }

        // Inactive slots carry no column; zero them rather than divide by an unvalidated weight.
        Ok(Matrix::from_fn(|index, row| {
            if index >= self.columns {
                return T::ZERO;
            }
            match (self.entries.get(row, index), joint_weights.get(index)) {
                (Some(entry), Some(weight)) => *entry / *weight,
                _ => T::ZERO,
            }
        }))
    }

    /// `σ_min(J)` — distance to a singular configuration.
    ///
    /// Identically zero for a chain with fewer than six actuated DOF, which never spans the task
    /// space; it only discriminates between poses from six DOF up.
    ///
    /// Errors: [`Linalg`](KinematicsError::Linalg) if the decomposition fails, i.e. a non-finite
    /// entry.
    ///
    /// ```
    /// use multicalc::kinematics::{JacobianFrame, Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
    /// let tree = KinematicTree::<2, f64>::try_from_joints(
    ///     &[Joint::revolute(z, SE3::identity()), Joint::revolute(z, link)],
    ///     &[JointParent::World, JointParent::Joint(0)],
    /// )
    /// .unwrap();
    ///
    /// // A 2-DOF chain never spans SE(3), so σ_min is identically zero regardless of pose.
    /// let jacobian = tree
    ///     .geometric_jacobian_at(&Vector::new([0.3, -0.7]), 1, JacobianFrame::World)
    ///     .unwrap();
    /// assert!(jacobian.smallest_singular_value().unwrap() < 1e-12);
    /// ```
    pub fn smallest_singular_value(&self) -> Result<T, KinematicsError> {
        // σ(J·Jᵀ) = σ(J)², and the Gram matrix is a fixed 6×6 whatever MAX_JOINTS is — `svd`
        // needs rows ≥ cols, which a wide J violates. Singular values come back descending.
        let singular_values = (self.entries * self.entries.transpose())
            .svd()?
            .singular_values();
        let smallest = singular_values.get(5).copied().unwrap_or(T::ZERO);
        Ok(smallest.max(T::ZERO).sqrt())
    }

    /// Weighted damped-least-squares inverse `W⁻¹ Jᵀ (J W⁻¹ Jᵀ + λ² I₆)⁺`.
    ///
    /// `damping = 0` gives the weighted Moore–Penrose pseudo-inverse; raising λ trades task-space
    /// tracking accuracy for bounded joint rates near a singularity.
    ///
    /// Errors: [`NonPositiveWeight`](KinematicsError::NonPositiveWeight) on a non-positive weight,
    /// or [`Linalg`](KinematicsError::Linalg) if the decomposition fails.
    ///
    /// ```
    /// use multicalc::kinematics::{JacobianFrame, Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
    /// let tree = KinematicTree::<2, f64>::try_from_joints(
    ///     &[Joint::revolute(z, SE3::identity()), Joint::revolute(z, link)],
    ///     &[JointParent::World, JointParent::Joint(0)],
    /// )
    /// .unwrap();
    /// let jacobian = tree
    ///     .geometric_jacobian_at(&Vector::new([0.3, -0.7]), 1, JacobianFrame::World)
    ///     .unwrap();
    ///
    /// // q̇ → twist → q̇′ reproduces the twist: J J⁺ is the identity on the reachable subspace.
    /// let rates = Vector::new([0.5, -1.25]);
    /// let wanted = jacobian.tool_twist(&rates).to_vector();
    /// let weights = Vector::new([1.0, 1.0]);
    /// let recovered = jacobian.matrix() * (jacobian.damped_pseudo_inverse(&weights, 0.0).unwrap() * wanted);
    ///
    /// assert!((recovered - wanted).norm() < 1e-9);
    /// ```
    pub fn damped_pseudo_inverse(
        &self,
        joint_weights: &Vector<MAX_JOINTS, T>,
        damping: T,
    ) -> Result<Matrix<MAX_JOINTS, 6, T>, KinematicsError> {
        let weighted_transpose = self.weighted_transpose(joint_weights)?;
        let damped = self.entries * weighted_transpose
            + Matrix::<6, 6, T>::identity().scale(damping * damping);
        Ok(weighted_transpose * damped.svd()?.pseudo_inverse())
    }

    /// Null-space projector `I − J⁺_λ J`, for redundancy resolution.
    ///
    /// Projects a joint-space bias onto the null space of the task, so a redundant chain can
    /// pursue a secondary objective at zero end-effector twist.
    ///
    /// Errors: as [`damped_pseudo_inverse`](KinematicJacobian::damped_pseudo_inverse).
    ///
    /// ```
    /// use multicalc::kinematics::{JacobianFrame, Joint, JointParent, KinematicTree};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// let z = Vector::new([0.0, 0.0, 1.0]);
    /// let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
    ///
    /// // Planar 4R chain: rank-3 Jacobian against 4 actuated DOF, so the null space is
    /// // one-dimensional. A 3R chain would project to zero and prove nothing.
    /// let tree = KinematicTree::<4, f64>::try_from_joints(
    ///     &[
    ///         Joint::revolute(z, SE3::identity()),
    ///         Joint::revolute(z, link),
    ///         Joint::revolute(z, link),
    ///         Joint::revolute(z, link),
    ///     ],
    ///     &[
    ///         JointParent::World,
    ///         JointParent::Joint(0),
    ///         JointParent::Joint(1),
    ///         JointParent::Joint(2),
    ///     ],
    /// )
    /// .unwrap();
    /// let jacobian = tree
    ///     .geometric_jacobian_at(&Vector::new([0.3, -0.7, 0.4, 0.2]), 3, JacobianFrame::World)
    ///     .unwrap();
    ///
    /// // An arbitrary bias survives the projection but produces no end-effector twist.
    /// let weights = Vector::new([1.0, 1.0, 1.0, 1.0]);
    /// let kept = jacobian.null_space_projector(&weights, 0.0).unwrap()
    ///     * Vector::new([1.0, -0.5, 0.25, 0.75]);
    ///
    /// assert!(kept.norm() > 0.1);
    /// assert!(jacobian.tool_twist(&kept).to_vector().norm() < 1e-9);
    /// ```
    pub fn null_space_projector(
        &self,
        joint_weights: &Vector<MAX_JOINTS, T>,
        damping: T,
    ) -> Result<Matrix<MAX_JOINTS, MAX_JOINTS, T>, KinematicsError> {
        // Inactive slots are left unprojected. Harmless: J's columns there are zero.
        let inverse = self.damped_pseudo_inverse(joint_weights, damping)?;
        Ok(Matrix::<MAX_JOINTS, MAX_JOINTS, T>::identity() - inverse * self.entries)
    }
}
