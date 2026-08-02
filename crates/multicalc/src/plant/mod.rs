//! Plant: the machinery between a command and the force a body actually feels.
//!
//! - [`MultirotorMixer`] — shares a wanted push and a wanted turn out across a set of rotors, and
//!   works the other way too, saying what push and turn a set of rotor thrusts adds up to.
//! - [`RotorCommands`] — what each rotor was asked for, and whether any of them was asked for more
//!   than it can give.
//! - [`RotorSpin`] — which way a rotor turns, which sets which way it twists the body.
//! - [`RotorLag`] — how quickly a rotor catches up to the thrust it was asked for, since it cannot
//!   change what it is giving the moment it is asked.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), in SI units.
//! Rotor positions and the push and turn they produce are all in the body's own axes. Depends on
//! [`linear_algebra`](crate::linear_algebra) and [`spatial`](crate::spatial).

mod multirotor_mixing;
mod rotor_lag;

pub use multirotor_mixing::{MultirotorMixer, RotorCommands, RotorSpin};
pub use rotor_lag::RotorLag;
