//! Control: feedback controllers, signal filters, and path-following laws.
//!
//! - [`Pid`] — PID with anti-windup.
//! - [`Lqr`] — optimal linear state feedback, with a check that the loop it closes settles.
//! - [`GeometricAttitudeController`] — attitude control on rotations, for a rigid body.
//! - [`thrust_command_from_acceleration`] — how to point and how hard to push, for a wanted
//!   acceleration. What joins a position loop to an attitude loop.
//! - [`OnePoleLowPass`] — the filter on the PID's derivative term, re-exported from
//!   [`signal_processing`](crate::signal_processing).
//! - [`pure_pursuit_curvature`] — the pure-pursuit path-following law (takes a lookahead point).
//! - [`FollowTheGap`] — reactive gap-following over a range scan.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), in SI units and
//! radians, on a fixed timestep `timestep`. Depends on [`linear_algebra`](crate::linear_algebra),
//! [`spatial`](crate::spatial), and [`kinematics`](crate::kinematics), not on
//! [`motion`](crate::motion).

mod follow_the_gap;
mod geometric_attitude;
mod lqr;
mod pid;
mod pure_pursuit;
mod thrust_command;

pub use crate::signal_processing::OnePoleLowPass;
pub use follow_the_gap::{FollowTheGap, FollowTheGapOutput};
pub use geometric_attitude::GeometricAttitudeController;
pub use lqr::Lqr;
pub use pid::Pid;
pub use pure_pursuit::{Curvature, pure_pursuit_curvature};
pub use thrust_command::{ThrustCommand, thrust_command_from_acceleration};
