//! Control: feedback controllers, signal filters, and path-following laws.
//!
//! Everything here is generic over [`Numeric`](crate::Numeric), so the same code runs at `f32` or
//! `f64` or through an autodiff scalar. Units are SI, angles are radians, and controllers operate on
//! a fixed timestep `dt`.
//!
//! The pure-pursuit law consumes a lookahead *point* rather than a path, so this module does not
//! depend on [`motion`](crate::motion); it depends on [`spatial`](crate::spatial) for poses and on
//! [`kinematics`](crate::kinematics) for the body-twist output.
//!
//! The gap-follower consumes a range scan and produces a body twist, so it depends on
//! [`kinematics`](crate::kinematics) for the output type and on nothing else.

mod derivative_filter;
mod follow_the_gap;
mod pid;
mod pure_pursuit;

pub use derivative_filter::OnePoleLowPass;
pub use follow_the_gap::{FollowTheGap, GapPlan};
pub use pid::Pid;
pub use pure_pursuit::{Curvature, pure_pursuit_curvature};
