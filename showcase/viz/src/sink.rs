//! The logging trait shared by every backend.
//!
//! [`VizSink`] is kept object-safe so callers can hold a `&mut dyn VizSink` and swap backends at
//! runtime; that is why `scalar` takes a plain `f64`. The generic convenience form that accepts
//! any [`Primal`] scalar lives on the blanket [`VizSinkExt`].

use multicalc::scalar::Primal;

use core::fmt;

/// An error from a sink backend.
#[derive(Debug)]
pub enum VizError {
    /// A backend SDK call failed (stream setup, connection, log, or flush).
    Backend(String),
    /// A filesystem error (CSV writer, or writing a recording).
    Io(std::io::Error),
}

impl fmt::Display for VizError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VizError::Backend(m) => write!(f, "viz backend error: {m}"),
            VizError::Io(e) => write!(f, "viz io error: {e}"),
        }
    }
}

impl std::error::Error for VizError {}

impl From<std::io::Error> for VizError {
    fn from(e: std::io::Error) -> Self {
        VizError::Io(e)
    }
}

/// sRGB color with alpha, 0–255 per channel.
pub type Rgba = [u8; 4];

/// A destination for logged data. Methods cover the archetypes the current core types can
/// produce; more are added as new types appear.
pub trait VizSink {
    /// Advances a sequence timeline (e.g. an iteration or sample index).
    fn set_sequence(&mut self, timeline: &str, seq: i64);

    /// Logs one scalar at the current timeline position.
    fn scalar(&mut self, path: &str, value: f64) -> Result<(), VizError>;

    /// Logs a set of 2D points.
    fn points2d(&mut self, path: &str, xy: &[[f64; 2]]) -> Result<(), VizError>;

    /// Logs a set of 3D points.
    fn points3d(&mut self, path: &str, xyz: &[[f64; 3]]) -> Result<(), VizError>;

    /// Logs a row-major matrix as a 2D tensor.
    fn tensor(&mut self, path: &str, shape: [usize; 2], data: &[f64]) -> Result<(), VizError>;

    /// Logs 2D points with per-point styling. `colors` and `radii` are each either length 1
    /// (broadcast to every point) or equal in length to `xy`. Radii are in scene units.
    ///
    /// The default falls through to the unstyled [`points2d`](VizSink::points2d), so backends
    /// that cannot style points inherit correct-but-plain behavior.
    fn points2d_styled(
        &mut self,
        path: &str,
        xy: &[[f64; 2]],
        colors: &[Rgba],
        radii: &[f32],
    ) -> Result<(), VizError> {
        let _ = (colors, radii);
        self.points2d(path, xy)
    }

    /// Logs 3D points with per-point styling. Broadcast and unit conventions match
    /// [`points2d_styled`](VizSink::points2d_styled).
    fn points3d_styled(
        &mut self,
        path: &str,
        xyz: &[[f64; 3]],
        colors: &[Rgba],
        radii: &[f32],
    ) -> Result<(), VizError> {
        let _ = (colors, radii);
        self.points3d(path, xyz)
    }

    /// Logs a batch of 2D poly-lines. `colors` and `widths` are each either length 1 (broadcast
    /// to every strip) or equal in length to `strips`. Widths are radii in scene units.
    fn line_strips2d(
        &mut self,
        path: &str,
        strips: &[Vec<[f64; 2]>],
        colors: &[Rgba],
        widths: &[f32],
    ) -> Result<(), VizError>;

    /// Logs a batch of 3D poly-lines. Broadcast and unit conventions match
    /// [`line_strips2d`](VizSink::line_strips2d).
    fn line_strips3d(
        &mut self,
        path: &str,
        strips: &[Vec<[f64; 3]>],
        colors: &[Rgba],
        widths: &[f32],
    ) -> Result<(), VizError>;

    /// Logs a row-major RGB8 image; `data.len()` must equal `width * height * 3`.
    fn image_rgb8(
        &mut self,
        path: &str,
        width: u32,
        height: u32,
        data: &[u8],
    ) -> Result<(), VizError>;

    /// Logs a markdown text panel (the `hud/stats` headline).
    fn text(&mut self, path: &str, markdown: &str) -> Result<(), VizError>;

    /// Styles the scalar series at `path`: line `color`, legend `name`, and `width`. Applied
    /// statically, so it holds across the whole timeline; call it once alongside the scalars.
    ///
    /// The default is a no-op, so backends without series styling keep default plot colors.
    fn series_style(
        &mut self,
        path: &str,
        color: Rgba,
        name: &str,
        width: f32,
    ) -> Result<(), VizError> {
        let _ = (path, color, name, width);
        Ok(())
    }

    /// Flushes buffered data.
    fn flush(&mut self) -> Result<(), VizError>;
}

/// Convenience extensions kept off the object-safe [`VizSink`].
pub trait VizSinkExt: VizSink {
    /// Logs any [`Primal`] scalar without an explicit `to_f64`.
    fn scalar_of(&mut self, path: &str, value: impl Primal) -> Result<(), VizError> {
        self.scalar(path, value.to_f64())
    }
}

impl<T: VizSink + ?Sized> VizSinkExt for T {}
