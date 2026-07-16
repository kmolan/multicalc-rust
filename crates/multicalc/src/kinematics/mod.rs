//! Kinematics: maps between actuator motion and body motion, and pose integration.
//!
//! Everything here is generic over [`Numeric`](crate::Numeric), so the same code runs at `f32` or
//! `f64` or through an autodiff scalar — differentiating an odometry step through
//! [`Dual`](crate::Dual) gives exact Jacobians with no hand-derived formulas.
//!
//! Units are SI and angles are radians. Twists are linear-first `[v; ω]`, matching
//! [`SE2`](crate::spatial::SE2). Poses advance by right-perturbation, `X · exp(ξ)`.
//!
//! The velocity motion model and its exact-arc integration follow Thrun, Burgard and Fox,
//! *Probabilistic Robotics*, Ch. 5, and Siegwart and Nourbakhsh, *Introduction to Autonomous
//! Mobile Robots*, Ch. 3.

mod diff_drive;
mod odometry;
mod unicycle;

pub use diff_drive::{ChassisDelta, ChassisRate, DiffDrive, WheelDeltas, WheelRates};
pub use odometry::{OdometryStep, integrate};
pub use unicycle::Unicycle;
