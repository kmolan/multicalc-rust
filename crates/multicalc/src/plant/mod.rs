//! Plant: what sits between a command and the wrench a body feels.
//!
//! - [`MultirotorMixer`] — body thrust and torque split across a set of rotors, and the inverse
//!   map from rotor thrusts back to the wrench they produce.
//! - [`RotorCommands`] — the per-rotor thrusts, and which of them saturated.
//! - [`RotorSpin`] — a rotor's spin direction, which sets the sign of its reaction torque.
//! - [`RotorLag`] — first-order rotor thrust lag.
//! - [`PositionServo`] — a joint that takes a commanded position rather than a torque, and its own
//!   servo's response. The stiff linear half of an operator-split step, advanced exactly.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) (so `f32`/`f64`/autodiff), in SI units.
//! Rotor positions and the wrench they produce are in body axes. Depends on
//! [`linear_algebra`](crate::linear_algebra), [`spatial`](crate::spatial), and
//! [`discretization`](crate::discretization).

mod multirotor_mixing;
mod position_servo;
mod rotor_lag;

pub use multirotor_mixing::{MultirotorMixer, RotorCommands, RotorSpin};
pub use position_servo::PositionServo;
pub use rotor_lag::RotorLag;
