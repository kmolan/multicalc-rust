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
//! - [`flight_reference`]: where the body should be at a given moment
//! - [`flight_world`]: the driver — one tick in, one record out
//! - [`quality_gates`]: the measured bounds the demo has to clear
//!
//! Everything here is specific to this demo. The numerics it is built from — the body, the rotor
//! mixing, both loops, and the integrators — all come from `multicalc`.

pub mod flight_controller;
pub mod flight_estimator;
pub mod flight_hangar;
pub mod flight_plant;
pub mod flight_reference;
pub mod flight_world;
pub mod quality_gates;
pub mod x2_model;

pub use flight_controller::{FlightCommand, FlightController};
pub use flight_estimator::{
    EstimatedState, FlightEstimator, StartingBelief, StartingSpreads, StateSource,
};
pub use flight_hangar::{FlightHangar, flight_hangar};
pub use flight_plant::{FlightPlant, angle_from_upright, level_heading};
pub use flight_reference::{
    CIRCLE_RADIUS, CIRCLE_SPEED, FlightReference, PLAN_WAYPOINTS, ReferenceSample, STEP_DISTANCE,
    STEP_HOLD_SECONDS, lean_for_circle, planned_waypoints, step_started_at,
};
pub use flight_world::{
    FlightMetrics, FlightPhase, FlightWorld, HOVER_POINT, LIDAR_BEAMS, PUSH_JITTER, TIMESTEP,
    TURN_RATE_JITTER, TickRecord, WARMUP_TICKS, rotor_positions_in_world, rotor_tone_for,
};
pub use quality_gates::{GateOutcome, SEED, run_ladder};
pub use x2_model::{GRAVITY_STRENGTH, ROTOR_COUNT, ROTOR_TONE_HERTZ, X2Model};
