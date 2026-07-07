//! Rerun backend: stream to a live viewer or record to a file.
//!
//! Both constructors are stock SDK calls; Rerun's own recording-stream thread does the live
//! streaming. `live()` spawns the external viewer found on PATH (version-matched to the SDK).

use crate::sink::{VizError, VizSink};
use rerun::RecordingStreamBuilder;
use std::path::Path;

/// A sink backed by a Rerun recording stream.
pub struct RerunSink {
    stream: rerun::RecordingStream,
}

impl RerunSink {
    /// Opens a live viewer and streams to it.
    ///
    /// On a normal host this spawns a local viewer. When `RERUN_VIZ_URL` is set it connects to
    /// that address instead; under WSL (where the virtualized GPU usually cannot launch the
    /// viewer) it connects to the default `127.0.0.1:9876`, reaching a Windows-side viewer over
    /// shared localhost. In the connecting cases the external viewer must already be running.
    pub fn live(app_id: &str) -> Result<Self, VizError> {
        let builder = RecordingStreamBuilder::new(app_id.to_owned());
        let stream = match std::env::var("RERUN_VIZ_URL") {
            Ok(url) => builder.connect_grpc_opts(url),
            Err(_) if std::env::var_os("WSL_DISTRO_NAME").is_some() => builder.connect_grpc(),
            Err(_) => builder.spawn(),
        }
        .map_err(|e| VizError::Backend(e.to_string()))?;
        Ok(Self { stream })
    }

    /// Records to a file that can be replayed in the viewer.
    pub fn record(app_id: &str, path: impl AsRef<Path>) -> Result<Self, VizError> {
        let stream = RecordingStreamBuilder::new(app_id.to_owned())
            .save(path.as_ref().to_path_buf())
            .map_err(|e| VizError::Backend(e.to_string()))?;
        Ok(Self { stream })
    }
}

impl VizSink for RerunSink {
    fn set_sequence(&mut self, timeline: &str, seq: i64) {
        self.stream.set_time_sequence(timeline, seq);
    }

    fn scalar(&mut self, path: &str, value: f64) -> Result<(), VizError> {
        self.stream
            .log(path, &rerun::Scalars::single(value))
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn points2d(&mut self, path: &str, xy: &[[f64; 2]]) -> Result<(), VizError> {
        let pts: Vec<[f32; 2]> = xy.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
        self.stream
            .log(path, &rerun::Points2D::new(pts))
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn points3d(&mut self, path: &str, xyz: &[[f64; 3]]) -> Result<(), VizError> {
        let pts: Vec<[f32; 3]> = xyz
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        self.stream
            .log(path, &rerun::Points3D::new(pts))
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn tensor(&mut self, path: &str, shape: [usize; 2], data: &[f64]) -> Result<(), VizError> {
        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let tensor_data = rerun::datatypes::TensorData::new(
            vec![shape[0] as u64, shape[1] as u64],
            rerun::datatypes::TensorBuffer::F32(data_f32.into()),
        );
        self.stream
            .log(path, &rerun::Tensor::new(tensor_data))
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn flush(&mut self) -> Result<(), VizError> {
        self.stream
            .flush_blocking()
            .map_err(|e| VizError::Backend(e.to_string()))
    }
}
