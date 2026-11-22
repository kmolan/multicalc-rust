//! The world for the 2D localization-and-obstacle-avoidance demo: a robot that finds itself on a
//! known map, then laps a course of obstacles on lidar alone.
//!
//! - [`lap_track_2d`]: the marked course, rasterized into an occupancy grid
//! - `lap_driver_2d` (needs the `alloc` feature): one tick of the run — localize, fuse odometry, an
//!   IMU, and GPS, then plan and steer
//!
//! Everything here is specific to this demo. The parts other demos can reuse — the grid, the lidar,
//! the sensors, the filter models, and the localizer — live in [`crate::sim`].

/// The driver holds the heap-backed particle filter, so it needs `alloc`.
#[cfg(feature = "alloc")]
pub mod lap_driver_2d;
pub mod lap_track_2d;

#[cfg(feature = "alloc")]
pub use lap_driver_2d::{
    BEAMS, GPS_NIS_LOWER_BOUND, GPS_NIS_UPPER_BOUND, LapMetrics, LapWorld, Phase, TickRecord,
};
pub use lap_track_2d::{LapTrack2D, TRACK_COLUMNS, TRACK_ROWS, lap_track_2d};
