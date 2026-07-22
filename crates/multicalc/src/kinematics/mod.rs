//! Kinematics: maps between actuator motion and body motion, and pose integration.
//!
//! - [`DifferentialDrive`] — wheel ↔ body motion for a differential-drive base.
//! - [`Unicycle`] — the unicycle (body-twist) model.
//! - [`integrate`] — exact-arc SE(2) odometry from a body twist.
//!
//! Generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff — an odometry step through
//! [`Dual`](crate::Dual) gives exact Jacobians). SI units, radians, twists linear-first `[v; ω]`
//! (matching [`SE2`](crate::spatial::SE2)), poses advance by `X · exp(ξ)`. The velocity model and
//! arc integration follow Thrun/Burgard/Fox, *Probabilistic Robotics*, ch. 5, and
//! Siegwart/Nourbakhsh, *Introduction to Autonomous Mobile Robots*, ch. 3.

mod differential_drive;
mod odometry;
mod unicycle;

pub use differential_drive::{
    BodyArc, BodyTwist, DifferentialDrive, WheelRotations, WheelVelocities,
};
pub use odometry::{OdometryStep, integrate};
pub use unicycle::Unicycle;
