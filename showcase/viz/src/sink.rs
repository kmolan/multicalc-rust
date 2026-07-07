//! The logging trait shared by every backend.
//!
//! [`VizSink`] is kept object-safe so callers can hold a `&mut dyn VizSink` and swap backends at
//! runtime; that is why `scalar` takes a plain `f64`. The generic convenience form that accepts
//! any [`Plottable`] scalar lives on the blanket [`VizSinkExt`].

use crate::convert::Plottable;
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

    /// Flushes buffered data.
    fn flush(&mut self) -> Result<(), VizError>;
}

/// Convenience extensions kept off the object-safe [`VizSink`].
pub trait VizSinkExt: VizSink {
    /// Logs any [`Plottable`] scalar without an explicit `to_plot_f64`.
    fn scalar_of(&mut self, path: &str, value: impl Plottable) -> Result<(), VizError> {
        self.scalar(path, value.to_plot_f64())
    }
}

impl<T: VizSink + ?Sized> VizSinkExt for T {}
