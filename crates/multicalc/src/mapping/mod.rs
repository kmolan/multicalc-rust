//! Maps a robot can measure against: a grid of free and blocked cells, and the range a sensor
//! reads across it.
//!
//! - [`OccupancyMap`] — what any map must answer: its size, where it sits in the world, whether a
//!   cell is blocked, and how far a beam travels before it meets something.
//! - [`MutableOccupancyMap`] — marking cells from world geometry: single points, joined-up lines,
//!   and circles.
//! - [`OccupancyGrid`] — a bit-packed occupancy grid sized at compile time, one bit per cell.
//! - [`CellState`] — free, blocked, or not yet observed. A planner must not route through the last.
//! - [`LogOddsGrid`] — a belief map a robot builds itself, integrating scans by addition in
//!   log-odds.
//! - [`DistanceField`] — every cell's exact Euclidean distance to the nearest obstacle, with a
//!   bilinear query and its gradient.
//! - [`CostGrid`] — traversal cost inflated outward from those obstacles, for a planner to keep
//!   off the walls.
//! - [`GridGeometry`] — where a grid sits in the world, and the index arithmetic over it. Every
//!   grid here owns one.
//! - [`RayWalk`] — the cells a beam passes through, in order, as an iterator.
//! - [`DynamicOccupancyGrid`] — a heap-based occupancy grid for large maps (`alloc` only).
//! - [`ScanGeometry`] — the directions the beams of a forward-facing range scan point.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) and works without `std`.

mod cost_grid;
mod distance_field;
mod grid_geometry;
mod log_odds_grid;
mod occupancy_grid;
mod occupancy_grid_fixed;
mod ray_walk;
mod scan_geometry;

pub use cost_grid::CostGrid;
pub use distance_field::{DistanceField, DistanceTransformWorkspace};
pub use grid_geometry::GridGeometry;
pub use log_odds_grid::LogOddsGrid;
pub use occupancy_grid::{CellState, MutableOccupancyMap, OccupancyMap};
pub use occupancy_grid_fixed::OccupancyGrid;
pub use ray_walk::{RayStep, RayWalk};
pub use scan_geometry::ScanGeometry;

pub(crate) use scan_geometry::beam_angle_across;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use occupancy_grid::DynamicOccupancyGrid;
