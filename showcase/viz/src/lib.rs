//! Thin, std-only Rerun visualization adapter for `multicalc`.
//!
//! Maps core types to Rerun archetypes behind a small [`VizSink`] trait, with a Rerun backend
//! ([`RerunSink`], live or recorded) and a CSV backend ([`CsvSink`]) for the `plot.py` fallback.
//! A satellite crate: never a dependency of the core, excluded from bare-metal builds.

mod convert;
mod csv_sink;
mod rerun_sink;
mod sink;

pub use convert::Plottable;
pub use csv_sink::CsvSink;
pub use rerun_sink::RerunSink;
pub use sink::{VizError, VizSink, VizSinkExt};

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    // Records a few primitives to a temp file and checks a non-empty recording is produced.
    // Headless: `save` needs no viewer, so this runs in CI.
    #[test]
    fn record_writes_nonempty_rrd() -> Result<(), VizError> {
        let path = std::env::temp_dir().join("multicalc_viz_smoke.rrd");
        let _ = std::fs::remove_file(&path);

        let mut sink = RerunSink::record("multicalc-viz/smoke", &path)?;
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
