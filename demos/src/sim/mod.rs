//! A 2D sensor simulator for the demos: an occupancy grid and a forward-arc lidar. Std-only and
//! seeded, so a run reproduces exactly.

pub mod lap_track_2d;
pub mod lidar;
pub mod occupancy_grid;

pub use lap_track_2d::{LapTrack2D, lap_track_2d, rounded_rectangle, wrap_angle};
pub use lidar::Lidar2d;
pub use occupancy_grid::{GridError, OccupancyGrid};
