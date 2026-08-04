//! Maps a robot can measure against: a grid of free and blocked cells, and the range a sensor
//! reads across it.
//!
//! - [`OccupancyMap`] — what any map must answer: its size, where it sits in the world, whether a
//!   cell is blocked, and how far a beam travels before it meets something.
//! - [`MutableOccupancyMap`] — marking cells from world geometry: single points, joined-up lines,
//!   and circles.
//! - [`DynamicOccupancyGrid`] — a heap-based occupancy grid for large maps (`alloc` only).
//! - [`ScanGeometry`] — the directions the beams of a forward-facing range scan point.
//!
//! Everything is generic over [`Numeric`](crate::Numeric) and works without `std`.

mod occupancy_grid;
mod scan_geometry;

pub use occupancy_grid::{MutableOccupancyMap, OccupancyMap};
pub use scan_geometry::ScanGeometry;

pub(crate) use scan_geometry::beam_angle_across;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use occupancy_grid::DynamicOccupancyGrid;
