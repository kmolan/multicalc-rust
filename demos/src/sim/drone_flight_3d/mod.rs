//! The world for the 3D drone flight demo: a quadrotor that holds a point, flies a path, and works
//! out where it is from what it can feel.
//!
//! - [`x2_model`]: the machine's mass read from its model file, and its rotor layout transcribed
//!   from the same file
//! - [`flight_plant`]: the truth motion, with rotors that take time to spin up and air that pushes
//!   back
//! - [`flight_controller`]: the outer loop that decides where to push and the inner one that
//!   decides how to point
//! - [`flight_estimator`]: the notches that make the inertial reading usable, and what integrating
//!   it on its own comes to
//! - [`flight_hangar`]: the room, as a floor plan the machine can match itself against
//! - [`flight_localization`]: the scanner, the cloud of guesses, and the turn on the spot that
//!   settles where in the room the machine is before it flies
//! - [`flight_reference`]: where the body should be at a given moment
//! - [`flight_world`]: the driver — one tick in, one record out
//!
//! Everything here is specific to this demo, and the numerics behind it all come from `multicalc`.
//! The sensors sit one level up in [`crate::sim`], because a beam or a compass is not particular to
//! this machine.

pub mod flight_controller;
pub mod flight_estimator;
pub mod flight_hangar;
pub mod flight_localization;
pub mod flight_plant;
pub mod flight_reference;
pub mod flight_world;
pub mod x2_model;

pub use flight_controller::{FlightCommand, FlightController};
pub use flight_estimator::{
    EstimatedState, FlightEstimator, StartingBelief, StartingSpreads, StateSource,
};
pub use flight_hangar::{FlightHangar, flight_hangar};
pub use flight_localization::{LIDAR_BEAMS, StartupLocalization};
pub use flight_plant::{FlightPlant, angle_from_upright, level_heading};
pub use flight_reference::{
    CIRCLE_RADIUS, CIRCLE_SPEED, FlightReference, PLAN_WAYPOINTS, ReferenceSample, STEP_DISTANCE,
    STEP_HOLD_SECONDS, lean_for_circle, planned_waypoints,
};
pub use flight_world::{
    FlightMetrics, FlightPhase, FlightWorld, HOVER_POINT, PUSH_JITTER, SEED, TIMESTEP,
    TURN_RATE_JITTER, TickRecord, WARMUP_TICKS, rotor_positions_in_world, rotor_tone_for,
};
pub use x2_model::{GRAVITY_STRENGTH, ROTOR_COUNT, ROTOR_TONE_HERTZ, X2Model};
