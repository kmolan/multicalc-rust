//! Joint model: degree of freedom, axis, and the fixed transform to its parent.
#![deny(clippy::indexing_slicing)]

use crate::linear_algebra::{Vector, Vector3D};
use crate::scalar::Numeric;
use crate::spatial::SE3;

/// Degree of freedom a joint provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointKind {
    /// One rotational degree of freedom about the joint axis.
    Revolute,
    /// One translational degree of freedom along the joint axis.
    Prismatic,
    /// One rotational degree of freedom about the joint axis, with no travel limit.
    Continuous,
    /// No degrees of freedom; a weld.
    Fixed,
    /// Free to move in all six directions
    Floating,
}

/// The frame a joint is attached to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointParent {
    /// Attached to the world frame.
    World,
    /// Attached to an earlier joint in the tree, by index.
    Joint(usize),
}

/// A single-degree-of-freedom joint: its type, axis, and fixed transform to its parent.
///
/// `origin` is the parent-to-joint transform at zero configuration. A revolute joint rotates about
/// `axis` through `anchor`, both in the joint frame; a prismatic joint translates along `axis` and
/// ignores `anchor`. `zero_offset` is the reference configuration (MuJoCo `ref`), so encoder zero
/// and model zero need not coincide.
///
/// `armature`, `damping` and `friction_loss` are model data for dynamics; forward kinematics does
/// not read them.
///
/// Construction is infallible; values are validated on entry to a
/// [`KinematicTree`](crate::kinematics::KinematicTree).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Joint<T: Numeric = f64> {
    kind: JointKind,
    axis: Vector3D<T>,
    anchor: Vector3D<T>,
    origin: SE3<T>,
    zero_offset: T,
    limits: Option<(T, T)>,
    armature: T,
    damping: T,
    friction_loss: T,
}

impl<T: Numeric> Joint<T> {
    /// Revolute joint about `axis`, with parent-to-joint transform `origin`.
    ///
    /// Defaults: anchor at the joint origin, zero reference offset, no limits, zero armature,
    /// damping and friction loss.
    ///
    /// ```
    /// use core::f64::consts::FRAC_PI_2;
    /// use multicalc::kinematics::{Joint, JointKind};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::SE3;
    ///
    /// // Rotates about z, travel limited to ±90°.
    /// let joint = Joint::revolute(Vector::new([0.0, 0.0, 1.0]), SE3::<f64>::identity())
    ///     .with_limits(-FRAC_PI_2, FRAC_PI_2);
    ///
    /// assert_eq!(joint.kind(), JointKind::Revolute);
    /// assert_eq!(joint.axis(), Vector::new([0.0, 0.0, 1.0]));
    /// assert_eq!(joint.limits(), Some((-FRAC_PI_2, FRAC_PI_2)));
    /// ```
    #[inline]
    #[must_use]
    pub fn revolute(axis: Vector3D<T>, origin: SE3<T>) -> Self {
        Joint::from_kind(JointKind::Revolute, axis, origin)
    }

    /// Continuous joint about `axis`, with parent-to-joint transform `origin`: turns like a
    /// `Revolute` joint but never stops — [`with_limits`](Joint::with_limits) is refused for it when
    /// the joint is added to a [`KinematicTree`](crate::kinematics::KinematicTree).
    ///
    /// Defaults are as for [`revolute`](Joint::revolute).
    ///
    /// ```
    /// use multicalc::kinematics::{Joint, JointKind};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::SE3;
    ///
    /// let joint = Joint::continuous(Vector::new([0.0, 0.0, 1.0]), SE3::<f64>::identity());
    ///
    /// assert_eq!(joint.kind(), JointKind::Continuous);
    /// assert_eq!(joint.limits(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn continuous(axis: Vector3D<T>, origin: SE3<T>) -> Self {
        Joint::from_kind(JointKind::Continuous, axis, origin)
    }

    /// Prismatic joint along `axis`, with parent-to-joint transform `origin`. The anchor is unused.
    ///
    /// Defaults are as for [`revolute`](Joint::revolute).
    ///
    /// ```
    /// use multicalc::kinematics::{Joint, JointKind};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::SE3;
    ///
    /// // Slides along x, 0.4 m of travel.
    /// let joint = Joint::prismatic(Vector::new([1.0, 0.0, 0.0]), SE3::<f64>::identity())
    ///     .with_limits(0.0, 0.4);
    ///
    /// assert_eq!(joint.kind(), JointKind::Prismatic);
    /// assert_eq!(joint.limits(), Some((0.0, 0.4)));
    /// ```
    #[inline]
    #[must_use]
    pub fn prismatic(axis: Vector3D<T>, origin: SE3<T>) -> Self {
        Joint::from_kind(JointKind::Prismatic, axis, origin)
    }

    /// Weld at `origin`, with no degrees of freedom.
    ///
    /// Takes a placeholder unit axis so it passes the tree's axis check; nothing reads it.
    ///
    /// ```
    /// use multicalc::kinematics::{Joint, JointKind};
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::{SE3, SO3};
    ///
    /// // Tool frame welded 0.1 m out along z.
    /// let origin = SE3::from_parts(SO3::<f64>::identity(), Vector::new([0.0, 0.0, 0.1]));
    /// let joint = Joint::fixed(origin);
    ///
    /// assert_eq!(joint.kind(), JointKind::Fixed);
    /// assert_eq!(joint.origin(), origin);
    /// ```
    #[inline]
    #[must_use]
    pub fn fixed(origin: SE3<T>) -> Self {
        Joint::from_kind(
            JointKind::Fixed,
            Vector::new([T::ONE, T::ZERO, T::ZERO]),
            origin,
        )
    }

    /// Floating (6-DOF) joint at `origin`: free to slide in any direction and turn about any
    /// axis, with no travel limit on any of the six.
    ///
    /// Its reading is not one number but seven — a position and a unit quaternion, `[x, y, z, w,
    /// qx, qy, qz]` — and its rate is six, not one; see
    /// [`FreeJointState`](crate::spatial::FreeJointState). Takes a placeholder unit axis so it
    /// passes the tree's axis check; nothing reads it. Has no travel limits, armature, damping, or
    /// friction loss — MJCF's `<freejoint>` carries none of these, and a six-DOF joint has no
    /// single number for any of them to describe.
    ///
    /// May only be a tree's first joint, attached to the world;
    /// [`KinematicTree::push`](crate::kinematics::KinematicTree::push) refuses one anywhere else.
    ///
    /// ```
    /// use multicalc::kinematics::{Joint, JointKind};
    /// use multicalc::spatial::SE3;
    ///
    /// let joint = Joint::<f64>::floating(SE3::identity());
    /// assert_eq!(joint.kind(), JointKind::Floating);
    /// ```
    #[inline]
    #[must_use]
    pub fn floating(origin: SE3<T>) -> Self {
        Joint::from_kind(
            JointKind::Floating,
            Vector::new([T::ONE, T::ZERO, T::ZERO]),
            origin,
        )
    }

    /// The same joint with its axis replaced, used by the tree to store a unit axis.
    #[inline]
    pub(crate) fn with_axis(self, axis: Vector3D<T>) -> Self {
        Joint { axis, ..self }
    }

    /// A joint of the given kind, with every optional value at its default.
    #[inline]
    fn from_kind(kind: JointKind, axis: Vector3D<T>, origin: SE3<T>) -> Self {
        Joint {
            kind,
            axis,
            anchor: Vector::zeros(),
            origin,
            zero_offset: T::ZERO,
            limits: None,
            armature: T::ZERO,
            damping: T::ZERO,
            friction_loss: T::ZERO,
        }
    }

    /// Sets the point the rotation axis passes through, in the joint frame.
    #[inline]
    #[must_use]
    pub fn with_anchor(self, anchor: Vector3D<T>) -> Self {
        Joint { anchor, ..self }
    }

    /// Sets the reference configuration: the value at which the joint sits at `origin`.
    #[inline]
    #[must_use]
    pub fn with_zero_offset(self, zero_offset: T) -> Self {
        Joint {
            zero_offset,
            ..self
        }
    }

    /// Sets the travel limits, lower first.
    #[inline]
    #[must_use]
    pub fn with_limits(self, lower: T, upper: T) -> Self {
        Joint {
            limits: Some((lower, upper)),
            ..self
        }
    }

    /// Sets the rotor inertia reflected through the gearing.
    #[inline]
    #[must_use]
    pub fn with_armature(self, armature: T) -> Self {
        Joint { armature, ..self }
    }

    /// Sets the viscous damping coefficient.
    #[inline]
    #[must_use]
    pub fn with_damping(self, damping: T) -> Self {
        Joint { damping, ..self }
    }

    /// Sets the dry-friction loss.
    #[inline]
    #[must_use]
    pub fn with_friction_loss(self, friction_loss: T) -> Self {
        Joint {
            friction_loss,
            ..self
        }
    }

    /// The joint's degree of freedom.
    #[inline]
    #[must_use]
    pub fn kind(self) -> JointKind {
        self.kind
    }

    /// Rotation or translation axis, in the joint frame.
    #[inline]
    pub fn axis(self) -> Vector3D<T> {
        self.axis
    }

    /// Point the rotation axis passes through, in the joint frame.
    #[inline]
    pub fn anchor(self) -> Vector3D<T> {
        self.anchor
    }

    /// Parent-to-joint transform at zero configuration.
    #[inline]
    #[must_use]
    pub fn origin(self) -> SE3<T> {
        self.origin
    }

    /// Reference configuration: the value at which the joint sits at `origin`.
    #[inline]
    #[must_use]
    pub fn zero_offset(self) -> T {
        self.zero_offset
    }

    /// Travel limits, lower first, or `None` when unlimited.
    #[inline]
    #[must_use]
    pub fn limits(self) -> Option<(T, T)> {
        self.limits
    }

    /// Reflected rotor inertia.
    #[inline]
    #[must_use]
    pub fn armature(self) -> T {
        self.armature
    }

    /// Viscous damping coefficient.
    #[inline]
    #[must_use]
    pub fn damping(self) -> T {
        self.damping
    }

    /// Dry-friction loss.
    #[inline]
    #[must_use]
    pub fn friction_loss(self) -> T {
        self.friction_loss
    }
}

impl<T: Numeric> Default for Joint<T> {
    /// A weld at the identity transform.
    #[inline]
    fn default() -> Self {
        Joint::fixed(SE3::identity())
    }
}
