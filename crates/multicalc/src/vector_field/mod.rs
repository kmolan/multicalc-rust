//! Vector calculus on 2D/3D fields.
//!
//! - [`curl`] / [`divergence`] — differential operators on a vector field.
//! - [`line_integral`] / [`flux_integral`] — a field integrated along a curve or through a surface.

pub mod curl;
pub mod divergence;
pub mod flux_integral;
pub mod line_integral;
