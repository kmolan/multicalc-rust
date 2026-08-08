//! A 2D sensor simulator for the demos: std-only and seeded, so a run reproduces exactly.
//!
//! - [`sensor_noise`]: the jitter every simulated sensor adds
//! - [`wheeled_vehicle`]: differential-drive truth motion and noisy wheel odometry
//! - [`inertial_measurement_unit`]: a noisy absolute heading and turn rate
//! - [`inertial_measurement_unit_3d`]: a noisy three-axis turn rate and push, shaking with the
//!   machine it is bolted to
//! - [`global_position_sensor`]: a noisy absolute position
//! - [`satellite_navigation_sensor`]: a noisy absolute position in three dimensions, worse up and
//!   down than across the ground, and a much finer reading of how fast the body is going
//! - [`height_rangefinder`]: a downward beam saying how high the body is
//! - [`magnetic_compass`]: a noisy reading of which way the nose points
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
//! its own submodule beside it, as [`localization_obstacle_avoidance_2d`] and [`drone_flight_3d`]
//! do.

pub mod drone_flight_3d;
pub mod geometry;
pub mod global_position_sensor;
pub mod grid_loading;
pub mod height_rangefinder;
pub mod inertial_measurement_unit;
pub mod inertial_measurement_unit_3d;
pub mod lidar;
pub mod localization_obstacle_avoidance_2d;
pub mod magnetic_compass;
pub mod satellite_navigation_sensor;
pub mod sensor_noise;
pub mod wheeled_vehicle;

pub use geometry::{box_outline, circle_outline, rotate_points, rounded_rectangle};
pub use global_position_sensor::GlobalPositionSensor;
pub use grid_loading::{GridFileError, load_occupancy_grid_csv};
pub use height_rangefinder::HeightRangefinder;
pub use inertial_measurement_unit::{InertialMeasurementUnit, InertialReading};
pub use inertial_measurement_unit_3d::{InertialMeasurementUnit3d, InertialReading3d};
pub use lidar::Lidar2d;
pub use magnetic_compass::MagneticCompass;
pub use satellite_navigation_sensor::SatelliteNavigationSensor;
pub use sensor_noise::gaussian_noise;
pub use wheeled_vehicle::{TruthStep, WheeledVehicle};
