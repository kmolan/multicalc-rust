//! Motion: waypoint paths and the geometric queries a path-following controller consumes.
//!
//! - [`PolylinePath`] — a dimension-generic, stack-allocated waypoint path (fixed capacity, runtime
//!   length) with total arc-length, closest-point, and lookahead-point queries (SI units).

mod polyline_path;

pub use polyline_path::{EndOfPath, PathProjection, PolylinePath};
