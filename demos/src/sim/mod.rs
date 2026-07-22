//! A 2D sensor simulator for the demos: an occupancy grid and a forward-arc lidar. Std-only and
//! seeded, so a run reproduces exactly.

pub mod lidar;
pub mod occupancy_grid;

pub use lidar::Lidar2d;
pub use occupancy_grid::{GridError, OccupancyGrid};
