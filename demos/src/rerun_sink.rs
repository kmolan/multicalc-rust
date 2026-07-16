//! Rerun backend: stream to a live viewer or record to a file.
//!
//! Both constructors are stock SDK calls; Rerun's own recording-stream thread does the live
//! streaming. `live()` spawns the external viewer found on PATH (version-matched to the SDK).

use crate::sink::{Rgba, VizError, VizSink};
use rerun::RecordingStreamBuilder;
use std::path::Path;

/// Maps our plain `Rgba` arrays to Rerun colors for a `with_colors` call.
fn colors_iter(colors: &[Rgba]) -> impl Iterator<Item = rerun::Color> + '_ {
    colors
        .iter()
        .map(|c| rerun::Color::from_unmultiplied_rgba(c[0], c[1], c[2], c[3]))
}

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

    fn points2d_styled(
        &mut self,
        path: &str,
        xy: &[[f64; 2]],
        colors: &[Rgba],
        radii: &[f32],
    ) -> Result<(), VizError> {
        let pts: Vec<[f32; 2]> = xy.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
        let arch = rerun::Points2D::new(pts)
            .with_colors(colors_iter(colors))
            .with_radii(radii.iter().copied());
        self.stream
            .log(path, &arch)
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn points3d_styled(
        &mut self,
        path: &str,
        xyz: &[[f64; 3]],
        colors: &[Rgba],
        radii: &[f32],
    ) -> Result<(), VizError> {
        let pts: Vec<[f32; 3]> = xyz
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let arch = rerun::Points3D::new(pts)
            .with_colors(colors_iter(colors))
            .with_radii(radii.iter().copied());
        self.stream
            .log(path, &arch)
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn line_strips2d(
        &mut self,
        path: &str,
        strips: &[Vec<[f64; 2]>],
        colors: &[Rgba],
        widths: &[f32],
    ) -> Result<(), VizError> {
        let strips_f32: Vec<Vec<[f32; 2]>> = strips
            .iter()
            .map(|s| s.iter().map(|p| [p[0] as f32, p[1] as f32]).collect())
            .collect();
        let arch = rerun::LineStrips2D::new(strips_f32)
            .with_colors(colors_iter(colors))
            .with_radii(widths.iter().copied());
        self.stream
            .log(path, &arch)
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn line_strips3d(
        &mut self,
        path: &str,
        strips: &[Vec<[f64; 3]>],
        colors: &[Rgba],
        widths: &[f32],
    ) -> Result<(), VizError> {
        let strips_f32: Vec<Vec<[f32; 3]>> = strips
            .iter()
            .map(|s| {
                s.iter()
                    .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                    .collect()
            })
            .collect();
        let arch = rerun::LineStrips3D::new(strips_f32)
            .with_colors(colors_iter(colors))
            .with_radii(widths.iter().copied());
        self.stream
            .log(path, &arch)
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn transform3d(
        &mut self,
        path: &str,
        translation: [f64; 3],
        quat_wxyz: [f64; 4],
    ) -> Result<(), VizError> {
        let t = [
            translation[0] as f32,
            translation[1] as f32,
            translation[2] as f32,
        ];
        // Core stores [w, x, y, z]; Rerun's Quaternion is xyzw — the one conversion point.
        let rot = rerun::Quaternion::from_xyzw([
            quat_wxyz[1] as f32,
            quat_wxyz[2] as f32,
            quat_wxyz[3] as f32,
            quat_wxyz[0] as f32,
        ]);
        let arch = rerun::Transform3D::from_translation_rotation(t, rot);
        self.stream
            .log(path, &arch)
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn boxes3d(
        &mut self,
        path: &str,
        centers: &[[f64; 3]],
        half_sizes: &[[f64; 3]],
        colors: &[Rgba],
    ) -> Result<(), VizError> {
        let c: Vec<[f32; 3]> = centers
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let h: Vec<[f32; 3]> = half_sizes
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let arch =
            rerun::Boxes3D::from_centers_and_half_sizes(c, h).with_colors(colors_iter(colors));
        self.stream
            .log(path, &arch)
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn arrows3d(
        &mut self,
        path: &str,
        origins: &[[f64; 3]],
        vectors: &[[f64; 3]],
        colors: &[Rgba],
    ) -> Result<(), VizError> {
        let o: Vec<[f32; 3]> = origins
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let v: Vec<[f32; 3]> = vectors
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect();
        let arch = rerun::Arrows3D::from_vectors(v)
            .with_origins(o)
            .with_colors(colors_iter(colors));
        self.stream
            .log(path, &arch)
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn image_rgb8(
        &mut self,
        path: &str,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<(), VizError> {
        let arch = rerun::Image::from_rgb24(data.to_vec(), [width, height]);
        self.stream
            .log(path, &arch)
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn text(&mut self, path: &str, markdown: &str) -> Result<(), VizError> {
        self.stream
            .log(path, &rerun::TextDocument::from_markdown(markdown))
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn series_style(
        &mut self,
        path: &str,
        color: Rgba,
        name: &str,
        width: f32,
    ) -> Result<(), VizError> {
        let arch = rerun::SeriesLines::new()
            .with_colors(colors_iter(&[color]))
            .with_names([name])
            .with_widths([width]);
        self.stream
            .log_static(path, &arch)
            .map_err(|e| VizError::Backend(e.to_string()))
    }

    fn flush(&mut self) -> Result<(), VizError> {
        self.stream
            .flush_blocking()
            .map_err(|e| VizError::Backend(e.to_string()))
    }
}
