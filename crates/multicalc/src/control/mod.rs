//! Control: feedback controllers, signal filters, and path-following laws.
//!
//! - [`Pid`] — PID with anti-windup.
//! - [`OnePoleLowPass`] — one-pole low-pass, for a filtered derivative.
//! - [`pure_pursuit_curvature`] — the pure-pursuit path-following law (takes a lookahead point).
//! - [`FollowTheGap`] — reactive gap-following over a range scan.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), in SI units and
//! radians, on a fixed timestep `dt`. Depends on [`spatial`](crate::spatial) and
//! [`kinematics`](crate::kinematics), not on [`motion`](crate::motion).

mod derivative_filter;
mod follow_the_gap;
mod pid;
mod pure_pursuit;

pub use derivative_filter::OnePoleLowPass;
pub use follow_the_gap::{FollowTheGap, FollowTheGapOutput};
pub use pid::Pid;
pub use pure_pursuit::{Curvature, pure_pursuit_curvature};
