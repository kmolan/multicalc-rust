//! Control: feedback controllers, signal filters, and path-following laws.
//!
//! - [`Pid`] — PID with anti-windup.
//! - [`Lqr`] — optimal linear state feedback, with a check that the loop it closes settles.
//! - [`GeometricAttitudeController`] — attitude control on rotations, for a rigid body.
//! - [`thrust_command_from_acceleration`] — the attitude and thrust realizing a desired
//!   acceleration; what joins a position loop to an attitude loop.
//! - [`OnePoleLowPass`] — the filter on the PID's derivative term, re-exported from
//!   [`signal_processing`](crate::signal_processing).
//! - [`pure_pursuit_curvature`] — the pure-pursuit path-following law (takes a lookahead point).
//! - [`FollowTheGap`] — reactive gap-following over a range scan.
//! - [`ComputedTorqueController`] — model-based joint tracking: the PD term driven through `H(q)`,
//!   so the tracking error obeys `ë + kd⊙ė + kp⊙e = 0` when the model is exact.
//! - [`JointImpedanceController`] — a spring-damper in joint space on top of bias compensation,
//!   leaving the model's natural inertia.
//! - [`JointPdController`] — joint-space PD, optionally with gravity cancelled. The torque-side
//!   view of position-controlled hardware.
//! - [`CartesianImpedanceController`] — a six-axis spring-damper at a tool frame, applied through
//!   `Jᵀ`, with an optional null-space posture term.
//! - [`JointReference`] / [`CartesianReference`] — the desired motion those laws take.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), in SI units and
//! radians, on a fixed timestep `timestep`. Depends on [`linear_algebra`](crate::linear_algebra),
//! [`spatial`](crate::spatial), [`kinematics`](crate::kinematics), and
//! [`dynamics`](crate::dynamics), not on [`motion`](crate::motion).

mod cartesian_impedance;
mod computed_torque;
mod follow_the_gap;
mod geometric_attitude;
mod joint_impedance;
mod joint_pd;
mod lqr;
mod motion_reference;
mod pid;
mod pure_pursuit;
mod thrust_command;

pub use crate::signal_processing::OnePoleLowPass;
pub use cartesian_impedance::CartesianImpedanceController;
pub use computed_torque::ComputedTorqueController;
pub use follow_the_gap::{FollowTheGap, FollowTheGapOutput};
pub use geometric_attitude::GeometricAttitudeController;
pub use joint_impedance::JointImpedanceController;
pub use joint_pd::JointPdController;
pub use lqr::Lqr;
pub use motion_reference::{CartesianReference, JointReference};
pub use pid::Pid;
pub use pure_pursuit::{Curvature, pure_pursuit_curvature};
pub use thrust_command::{ThrustCommand, thrust_command_from_acceleration};
