//! Std-only visualization adapter for `multicalc`.
//!
//! Maps core types to a small [`VizSink`] trait, with a CSV backend ([`CsvSink`]) for the
//! `plot.py` fallback and, behind the `rerun` feature, a Rerun backend ([`RerunSink`], live or
//! recorded). With the feature off the crate builds headless, with no Rerun in the dependency
//! tree. A satellite crate: never a dependency of the core, excluded from bare-metal builds.

mod csv_sink;
#[cfg(feature = "rerun")]
mod rerun_sink;
mod sink;

#[doc(hidden)]
pub mod loop_util;

pub use csv_sink::CsvSink;
pub use multicalc::scalar::Primal;
#[cfg(feature = "rerun")]
pub use rerun_sink::RerunSink;
pub use sink::{Rgba, VizError, VizSink, VizSinkExt};

#[cfg(all(test, feature = "rerun"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    // Records a few primitives to a temp file and checks a non-empty recording is produced.
    // Headless: `save` needs no viewer, so this runs in CI.
    #[test]
    fn record_writes_nonempty_rrd() -> Result<(), VizError> {
        let path = std::env::temp_dir().join("multicalc_demos_smoke.rrd");
        let _ = std::fs::remove_file(&path);

        let mut sink = RerunSink::record("multicalc-demos/smoke", &path)?;
        sink.set_sequence("iteration", 0);
        sink.scalar("objective", 1.0)?;
        sink.points2d("data", &[[0.0, 0.0], [1.0, 1.0]])?;
        sink.flush()?;
        drop(sink);

        let len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        assert!(len > 0, "recording should produce a non-empty .rrd");
        Ok(())
    }
}
