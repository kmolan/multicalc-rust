//! Where a demo's numbers go: one logging trait and the backends behind it.
//!
//! - [`sink`]: the [`VizSink`] trait every backend implements
//! - `rerun_sink` (needs the `rerun` feature): streams to a live viewer, or records a `.rrd`
//! - [`ground_grid`]: a floor grid to log once as a backdrop for the 3D demos

pub mod ground_grid;
#[cfg(feature = "rerun")]
pub mod rerun_sink;
pub mod sink;

pub use ground_grid::ground_grid;
#[cfg(feature = "rerun")]
pub use rerun_sink::RerunSink;
pub use sink::{Rgba, VizError, VizSink, VizSinkExt};
