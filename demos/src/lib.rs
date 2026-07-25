//! Std-only visualization adapter for `multicalc`.
//!
//! Maps core types to a small [`VizSink`] trait, with a CSV backend ([`CsvSink`]) for the
//! `plot.py` fallback and, behind the `rerun` feature, a Rerun backend ([`RerunSink`], live or
//! recorded). With the feature off the crate builds headless, with no Rerun in the dependency
//! tree. A satellite crate: never a dependency of the core, excluded from bare-metal builds.
//!
//! Also carries [`sim`], a std-only 2D sensor simulator (occupancy grid, lidar, sensors, filter
//! models) that the robot demos drive, along with the world of one such demo built on top of it.
//! All of it is demo scaffolding, never core numerics.

pub mod sim;
pub mod visualization;

#[doc(hidden)]
pub mod loop_util;

pub use multicalc::scalar::Primal;
#[cfg(feature = "rerun")]
pub use visualization::RerunSink;
pub use visualization::{CsvSink, Rgba, VizError, VizSink, VizSinkExt};
