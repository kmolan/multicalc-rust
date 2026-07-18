//! Motion: waypoint paths and the geometric queries a path-following controller consumes.
//!
//! Paths are dimension-generic and stack-allocated, storing their waypoints in a fixed-capacity
//! array with a runtime length. Units are SI. The queries — total arc length, closest point, and
//! lookahead point — are the primitives a pursuit law needs to turn a path and a pose into a target.

mod polyline_path;

pub use polyline_path::{EndOfPath, PathProjection, PolylinePath};
