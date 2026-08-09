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
    /// A filesystem error, from writing a recording.
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

    /// Marks whatever is logged next as belonging to no particular moment.
    ///
    /// Scene furniture — a floor grid, a course outline — belongs to the whole run rather than to
    /// the tick it happened to be logged on. Turn this on before logging it and off again
    /// afterwards. The default ignores the flag, so backends without the notion keep working.
    fn set_static(&mut self, always: bool) {
        let _ = always;
    }

    /// Logs one scalar at the current timeline position.
    fn scalar(&mut self, path: &str, value: f64) -> Result<(), VizError>;

    /// Logs several scalars that belong on one plot together, at the current timeline position.
    ///
    /// Everything logged under one path is drawn on one set of axes, which is the only way to put
    /// two quantities side by side rather than in plots of their own. Pair it with
    /// [`VizSink::series_styles`] to name and colour them.
    ///
    /// The default logs nothing, so backends that only understand one value at a time keep working.
    fn scalars(&mut self, path: &str, values: &[f64]) -> Result<(), VizError> {
        let _ = (path, values);
        Ok(())
    }

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

    /// Logs 2D points that each carry a text label, for a legend or a named landmark. `colors` and
    /// `radii` broadcast as in [`points2d_styled`](VizSink::points2d_styled); `labels` has one entry
    /// per point.
    ///
    /// The default drops the labels and falls through to
    /// [`points2d_styled`](VizSink::points2d_styled), so a backend that cannot draw text still
    /// places the points.
    fn points2d_labeled(
        &mut self,
        path: &str,
        xy: &[[f64; 2]],
        colors: &[Rgba],
        radii: &[f32],
        labels: &[&str],
    ) -> Result<(), VizError> {
        let _ = labels;
        self.points2d_styled(path, xy, colors, radii)
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

    /// Logs a rigid pose. `quat_wxyz` is `[w, x, y, z]` (core storage order); the sink converts to
    /// the backend's order. Children of `path` inherit this pose.
    fn transform3d(
        &mut self,
        path: &str,
        translation: [f64; 3],
        quat_wxyz: [f64; 4],
    ) -> Result<(), VizError>;

    /// Logs axis-aligned boxes in the entity's local frame. `colors` is length 1 (broadcast to
    /// every box) or equal in length to `centers`. `centers` and `half_sizes` have equal length.
    fn boxes3d(
        &mut self,
        path: &str,
        centers: &[[f64; 3]],
        half_sizes: &[[f64; 3]],
        colors: &[Rgba],
    ) -> Result<(), VizError>;

    /// Logs arrows from `origins` along `vectors`. `colors` is length 1 (broadcast) or equal in
    /// length to `vectors`. `origins` and `vectors` have equal length.
    fn arrows3d(
        &mut self,
        path: &str,
        origins: &[[f64; 3]],
        vectors: &[[f64; 3]],
        colors: &[Rgba],
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

    /// Logs a 3D model read from a file, such as an `.obj` or a `.glb`.
    ///
    /// The file is read and sent as it is; whatever reads it back is what understands the format.
    /// A model file says nothing about what units it was drawn in or which way up it was drawn, so
    /// a caller almost always pairs this with [`VizSink::transform3d_scaled`] on the same path.
    ///
    /// The default is a no-op, so backends that cannot draw a model keep working.
    fn model3d(&mut self, path: &str, file_path: &std::path::Path) -> Result<(), VizError> {
        let _ = (path, file_path);
        Ok(())
    }

    /// A pose that also resizes whatever sits under it.
    ///
    /// For anything drawn in units of its own — a model file measured in centimetres, say — where
    /// the resizing belongs to the thing itself and not to where it is.
    ///
    /// The default is a no-op, so backends without the notion keep working.
    fn transform3d_scaled(
        &mut self,
        path: &str,
        translation: [f64; 3],
        quat_wxyz: [f64; 4],
        scale: f64,
    ) -> Result<(), VizError> {
        let _ = (path, translation, quat_wxyz, scale);
        Ok(())
    }

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

    /// Styles several series sharing one path: a `color` and a `name` each, one `width` for all.
    ///
    /// The default is a no-op, so backends without series styling keep default plot colors.
    fn series_styles(
        &mut self,
        path: &str,
        colors: &[Rgba],
        names: &[&str],
        width: f32,
    ) -> Result<(), VizError> {
        let _ = (path, colors, names, width);
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
