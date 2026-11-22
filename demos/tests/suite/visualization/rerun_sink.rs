use multicalc_demos::{RerunSink, VizError, VizSink};

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
    sink.transform3d("pose", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])?;
    sink.boxes3d(
        "pose/body",
        &[[0.0, 0.0, 0.0]],
        &[[0.5, 0.5, 0.5]],
        &[[0x39, 0x87, 0xe5, 0xff]],
    )?;
    sink.arrows3d(
        "pose/axis",
        &[[0.0, 0.0, 0.0]],
        &[[1.0, 0.0, 0.0]],
        &[[0xc9, 0x85, 0x00, 0xff]],
    )?;
    sink.flush()?;
    drop(sink);

    let len = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
    assert!(len > 0, "recording should produce a non-empty .rrd");
    Ok(())
}
