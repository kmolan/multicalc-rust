//! Vector calculus on 2D/3D fields.
//!
//! - [`curl_2d`] / [`curl_3d`] / [`divergence_2d`] / [`divergence_3d`] — differential operators on
//!   a vector field.
//! - [`line_integral_2d`] / [`flux_integral_2d`] and their 3D forms — a field integrated along a
//!   curve or through a surface.

mod curl;
mod divergence;
mod flux_integral;
mod line_integral;

pub use curl::{curl_2d, curl_3d};
pub use divergence::{divergence_2d, divergence_3d};
pub use flux_integral::{
    flux_integral_2d, flux_integral_2d_custom, flux_integral_3d, flux_integral_3d_custom,
};
pub use line_integral::{
    line_integral_2d, line_integral_2d_custom, line_integral_3d, line_integral_3d_custom,
    line_integral_partial_2d, line_integral_partial_3d,
};
