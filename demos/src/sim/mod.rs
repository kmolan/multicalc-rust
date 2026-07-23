//! A 2D sensor simulator for the demos: std-only and seeded, so a run reproduces exactly.
//!
//! - [`occupancy_grid`]: a grid of free/occupied cells with ray casting
//! - [`lap_track_2d`]: the marked lap course rasterized into a grid
//! - [`wheeled_vehicle`]: differential-drive truth motion and noisy wheel odometry
//! - [`inertial_measurement_unit`]: a noisy absolute heading and turn rate
//! - [`global_position_sensor`]: a noisy absolute position
//! - [`lidar`]: a forward-arc range scan over the grid
//! - [`estimator`]: the EKF's coordinated-turn and measurement models

pub mod estimator;
pub mod global_position_sensor;
pub mod inertial_measurement_unit;
pub mod lap_track_2d;
pub mod lidar;
pub mod occupancy_grid;
pub mod wheeled_vehicle;

pub use estimator::{
    AttitudeHeadingModel, CoordinatedTurnModel, GlobalPositionModel, WheelOdometryModel,
    attitude_residual, diagonal,
};
pub use global_position_sensor::GlobalPositionSensor;
pub use inertial_measurement_unit::{InertialMeasurementUnit, InertialReading};
pub use lap_track_2d::{LapTrack2D, lap_track_2d, rounded_rectangle, wrap_angle};
pub use lidar::Lidar2d;
pub use occupancy_grid::{GridError, OccupancyGrid};
pub use wheeled_vehicle::{TruthStep, WheeledVehicle};
