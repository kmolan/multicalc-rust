//! Kinematics: maps between actuator motion and body motion, and pose integration.
//!
//! - [`DifferentialDrive`] — wheel ↔ body motion for a differential-drive base.
//! - [`Unicycle`] — the unicycle (body-twist) model.
//! - [`integrate`] — exact-arc SE(2) odometry from a body twist.
//! - [`Joint`] — one revolute, prismatic, or fixed joint: axis, anchor, and transform to its parent.
//! - [`KinematicTree`] — a jointed model for solving the world pose of every joint frame.
//!
//! Generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff — an odometry step through
//! [`Dual`](crate::Dual) gives exact Jacobians). SI units, radians, twists linear-first `[v; ω]`
//! (matching [`SE2`](crate::spatial::SE2)), poses advance by `X · exp(ξ)`. The velocity model and
//! arc integration follow Thrun/Burgard/Fox, *Probabilistic Robotics*, ch. 5, and
//! Siegwart/Nourbakhsh, *Introduction to Autonomous Mobile Robots*, ch. 3.

mod differential_drive;
mod joint;
mod kinematic_tree;
mod kinematic_tree_state;
mod odometry;
mod unicycle;

pub use differential_drive::{
    BodyArc, BodyTwist, DifferentialDrive, WheelRotations, WheelVelocities,
};
pub use joint::{Joint, JointKind, JointParent};
pub use kinematic_tree::KinematicTree;
pub use kinematic_tree_state::KinematicTreeState;
pub use odometry::{OdometryStep, integrate};
pub use unicycle::Unicycle;
