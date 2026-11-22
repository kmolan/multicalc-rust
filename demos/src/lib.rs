//! Std-only visualization adapter for `multicalc`.
//!
//! Maps core types to a small [`VizSink`] trait, with a Rerun backend ([`RerunSink`], live or
//! recorded) behind the `rerun` feature. With the feature off the crate builds headless, with no
//! Rerun in the dependency tree and nothing to log to. A satellite crate: never a dependency of
//! the core, excluded from bare-metal builds.
//!
//! Also carries [`sim`], a std-only simulator the demos drive: rooms as occupancy grids, the
//! sensors a machine carries and what they get wrong, and two whole worlds built on top of them —
//! a wheeled vehicle finding its way around a floor plan, and a quadrotor that finds itself on one
//! and then flies a planned loop. All of it is demo scaffolding, never core numerics.

pub mod sim;
pub mod visualization;

#[doc(hidden)]
pub mod loop_util;

pub use multicalc::scalar::Primal;
#[cfg(feature = "rerun")]
pub use visualization::RerunSink;
pub use visualization::{Rgba, VizError, VizSink, VizSinkExt};
