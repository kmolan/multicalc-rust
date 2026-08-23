//! Dynamics: how a rigid body moves under the forces put on it.
//!
//! - [`RigidBody`] — a single body free to move in all six directions. Give it the forces and
//!   turns acting on it and it says how it speeds up, which is what an integrator needs to move it
//!   forward in time.
//! - [`ArticulatedBody`] — a jointed robot with mass. Inverse dynamics by the recursive
//!   Newton-Euler algorithm, the joint-space inertia matrix by the composite-rigid-body algorithm,
//!   and forward dynamics by the articulated-body algorithm, with per-joint armature, viscous
//!   damping and Coulomb friction. Fixed-base.
//! - [`DynamicsWorkspace`] — caller-owned scratch for forward dynamics, so a control loop keeps a
//!   model-sized frame off the stack.
//! - [`RigidBodyAcceleration`] — how fast the body's motion is changing, straight-line and
//!   turning.
//! - [`RigidBody::stepped`] — moves the whole state one tick forward, carrying the direction the
//!   body faces as a turn so it stays a true rotation.
//! - [`state_vector_from_free_joint`] / [`free_joint_from_state_vector`] — the same state written
//!   as thirteen loose numbers, which is the form the ODE integrators take.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), in SI units
//! and radians. Positions and straight-line motion are in world axes; turning and the forces
//! applied to the body are in the body's own axes. Depends on
//! [`linear_algebra`](crate::linear_algebra), [`spatial`](crate::spatial),
//! [`kinematics`](crate::kinematics), and [`ode`](crate::ode).

mod articulated_body;
mod dynamics_workspace;
mod rigid_body;

pub use articulated_body::ArticulatedBody;
pub use dynamics_workspace::DynamicsWorkspace;
pub use rigid_body::{
    RigidBody, RigidBodyAcceleration, STATE_DIMENSION, free_joint_from_state_vector,
    state_vector_from_free_joint,
};
