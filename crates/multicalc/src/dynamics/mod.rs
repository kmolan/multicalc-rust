//! Dynamics: how a rigid body moves under the forces put on it.
//!
//! - [`RigidBody`] — a single body free to move in all six directions. Give it the forces and
//!   turns acting on it and it says how it speeds up, which is what an integrator needs to move it
//!   forward in time.
//! - [`RigidBodyAcceleration`] — how fast the body's motion is changing, straight-line and
//!   turning.
//! - [`state_vector_from_free_joint`] / [`free_joint_from_state_vector`] — the same state written
//!   as thirteen loose numbers, which is the form the ODE integrators take.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), in SI units
//! and radians. Positions and straight-line motion are in world axes; turning and the forces
//! applied to the body are in the body's own axes. Depends on
//! [`linear_algebra`](crate::linear_algebra) and [`spatial`](crate::spatial).

mod rigid_body;

pub use rigid_body::{
    RigidBody, RigidBodyAcceleration, STATE_DIMENSION, free_joint_from_state_vector,
    state_vector_from_free_joint,
};
