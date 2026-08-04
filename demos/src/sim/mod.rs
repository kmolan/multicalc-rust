//! A 2D sensor simulator for the demos: std-only and seeded, so a run reproduces exactly.
//!
//! - [`sensor_noise`]: the jitter every simulated sensor adds
//! - [`wheeled_vehicle`]: differential-drive truth motion and noisy wheel odometry
//! - [`inertial_measurement_unit`]: a noisy absolute heading and turn rate
//! - [`global_position_sensor`]: a noisy absolute position
//! - [`lidar`]: a forward-arc range scan over an occupancy map
//! - [`geometry`]: point lists tracing 2D shapes, for drawing and for rasterizing walls
//! - [`grid_loading`]: reading an occupancy grid out of a CSV file
//!
//! The map itself, the filter models, and the localizer come from `multicalc`:
//! `mapping::DynamicOccupancyGrid` and `mapping::ScanGeometry` for the world a scan is cast
//! against, `estimation::ConstantTurnAndSpeed` and `estimation::DirectMeasurement` for the filter,
//! and `estimation::MonteCarloLocalizer` for finding the robot on a known map.
//!
//! None of what is here is tied to one demo. A demo's own world is built on top of it and lives in
//! its own submodule beside it, as [`localization_obstacle_avoidance_2d`] does.

pub mod geometry;
pub mod global_position_sensor;
pub mod grid_loading;
pub mod inertial_measurement_unit;
pub mod lidar;
pub mod localization_obstacle_avoidance_2d;
pub mod sensor_noise;
pub mod wheeled_vehicle;

pub use geometry::{box_outline, circle_outline, rotate_points, rounded_rectangle};
pub use global_position_sensor::GlobalPositionSensor;
pub use grid_loading::{GridFileError, load_occupancy_grid_csv};
pub use inertial_measurement_unit::{InertialMeasurementUnit, InertialReading};
pub use lidar::Lidar2d;
pub use sensor_noise::gaussian_noise;
pub use wheeled_vehicle::{TruthStep, WheeledVehicle};
