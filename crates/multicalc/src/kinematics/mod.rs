//! Kinematics: maps between actuator motion and body motion, and pose integration.
//!
//! - [`DifferentialDrive`] — wheel ↔ body motion for a differential-drive base.
//! - [`Unicycle`] — the unicycle (body-twist) model.
//! - [`integrate`] — exact-arc SE(2) odometry from a body twist.
//! - [`Joint`] — one revolute, prismatic, continuous, fixed, or floating joint: axis, anchor, and
//!   transform to its parent.
//! - [`KinematicTree`] — a jointed model for solving the world pose of every joint frame.
//! - [`KinematicJacobian`] — how each joint's rate moves a chosen frame on the robot, with
//!   [`SingularityKind`] classifying a rank deficiency as positional, rotational or mixed.
//! - [`InverseKinematics`] — the joint readings that put a chosen frame where you want it.
//! - [`MultiStartInverseKinematics`] — the same solve run from several seeds, collecting the
//!   distinct branches found.
//!
//! Generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff — an odometry step through
//! [`Dual`](crate::Dual) gives exact Jacobians). SI units, radians, twists linear-first `[v; ω]`
//! (matching [`SE2`](crate::spatial::SE2)), poses advance by `X · exp(ξ)`. The velocity model and
//! arc integration follow Thrun/Burgard/Fox, *Probabilistic Robotics*, ch. 5, and
//! Siegwart/Nourbakhsh, *Introduction to Autonomous Mobile Robots*, ch. 3.

mod differential_drive;
mod inverse_kinematics;
mod joint;
mod kinematic_jacobian;
mod kinematic_tree;
mod kinematic_tree_state;
mod multi_start_inverse_kinematics;
mod odometry;
mod unicycle;

pub use differential_drive::{
    BodyArc, BodyTwist, DifferentialDrive, WheelRotations, WheelVelocities,
};
pub use inverse_kinematics::{
    InverseKinematics, InverseKinematicsReport, InverseKinematicsTermination, SecondaryObjective,
};
pub use joint::{Joint, JointKind, JointParent};
pub use kinematic_jacobian::{JacobianFrame, KinematicJacobian, SingularityKind};
pub use kinematic_tree::KinematicTree;
pub use kinematic_tree_state::KinematicTreeState;
pub use multi_start_inverse_kinematics::{MultiStartInverseKinematics, MultiStartReport};
pub use odometry::{OdometryStep, integrate};
pub use unicycle::Unicycle;
