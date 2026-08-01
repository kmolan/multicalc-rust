//! Motion: waypoint paths, planned trajectories, and the geometric queries a path-following
//! controller consumes.
//!
//! - [`PolylinePath`] — a dimension-generic, stack-allocated waypoint path (fixed capacity, runtime
//!   length) with total arc-length, closest-point, and lookahead-point queries (SI units).
//! - [`MinimumSnapPlanner`] — the smoothest path through a set of waypoints, as a polynomial per
//!   segment. Planning is a one-off cost, not per-tick work.
//! - [`durations_from_average_speed`] — a first guess at how long each segment should take.
//!
//! A planned trajectory comes back as a [`PiecewisePolynomial`], which
//! gives position and as many derivatives as asked for at any time along it. That type belongs to
//! the polynomial module, so its calls report a
//! [`PolynomialError`](crate::error::PolynomialError) rather than a
//! [`MotionError`](crate::error::MotionError), and it calls a segment's length a span where this
//! module calls it a duration.

mod minimum_snap;
mod polyline_path;

pub use crate::polynomial::PiecewisePolynomial;
pub use minimum_snap::{BoundaryDerivatives, MinimumSnapPlanner, durations_from_average_speed};
pub use polyline_path::{EndOfPath, PathProjection, PolylinePath};
