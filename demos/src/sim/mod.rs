//! A 2D sensor simulator for the demos: std-only and seeded, so a run reproduces exactly.
//!
//! - [`occupancy_grid`]: a grid of free/occupied cells with ray casting
//! - [`geometry`]: point lists tracing 2D shapes, and folding an angle back into range
//! - [`sensor_noise`]: the jitter every simulated sensor adds
//! - [`wheeled_vehicle`]: differential-drive truth motion and noisy wheel odometry
//! - [`inertial_measurement_unit`]: a noisy absolute heading and turn rate
//! - [`global_position_sensor`]: a noisy absolute position
//! - [`lidar`]: a forward-arc range scan over the grid
//! - [`kalman_filter_models`]: the motion and measurement models a Kalman filter needs
//! - `particle_filter_localizer` (needs the `alloc` feature): finds the robot on a known map
//!
//! None of those is tied to one demo. A demo's own world is built on top of them and lives in its
//! own submodule beside them, as [`localization_obstacle_avoidance_2d`] does.

pub mod geometry;
pub mod global_position_sensor;
pub mod inertial_measurement_unit;
pub mod kalman_filter_models;
pub mod lidar;
pub mod localization_obstacle_avoidance_2d;
pub mod occupancy_grid;
/// The localizer wraps `multicalc`'s heap-backed particle filter, so it needs `alloc`.
#[cfg(feature = "alloc")]
pub mod particle_filter_localizer;
pub mod sensor_noise;
pub mod wheeled_vehicle;

pub use geometry::{box_outline, circle_outline, rotate_points, rounded_rectangle, wrap_angle};
pub use global_position_sensor::GlobalPositionSensor;
pub use inertial_measurement_unit::{InertialMeasurementUnit, InertialReading};
pub use kalman_filter_models::{
    AttitudeHeadingModel, CoordinatedTurnModel, GlobalPositionModel, WheelOdometryModel,
    attitude_residual, diagonal,
};
pub use lidar::Lidar2d;
pub use occupancy_grid::{GridError, OccupancyGrid};
#[cfg(feature = "alloc")]
pub use particle_filter_localizer::{GlobalLocalizer, InitialParticleCloud};
pub use sensor_noise::gaussian_noise;
pub use wheeled_vehicle::{TruthStep, WheeledVehicle};
