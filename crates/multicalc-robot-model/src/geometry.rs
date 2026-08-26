//! Shapes a model file draws its bodies with.
//!
//! Read for viewing only: mass properties come from `<inertial>` or geom integration, never here.

use std::path::{Path, PathBuf};

use multicalc::linear_algebra::Vector3D;
use multicalc::spatial::SE3;

/// A shape as the file states it. Round forms stand along their own z axis.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GeometryShape {
    /// Radius.
    Sphere { radius: f64 },
    /// Semi-axes.
    Ellipsoid { semi_axes: Vector3D<f64> },
    /// Half-widths.
    Box { half_extents: Vector3D<f64> },
    /// Radius, and half the barrel length. Along local z.
    Cylinder { radius: f64, half_length: f64 },
    /// Radius, and half the barrel length between the cap centres. Along local z.
    Capsule { radius: f64, half_length: f64 },
    /// File as stated, relative to the model's directory unless absolute or naming a package, and
    /// the per-axis scale it is drawn at.
    Mesh { file: String, scale: Vector3D<f64> },
}

/// One drawable shape on a body, in body axes.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualGeometry {
    shape: GeometryShape,
    pose: SE3<f64>,
    color: [f64; 4],
    group: u32,
}

impl VisualGeometry {
    /// The shape and its dimensions.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &GeometryShape {
        &self.shape
    }

    /// Transform from the body frame.
    #[inline]
    #[must_use]
    pub fn pose(&self) -> SE3<f64> {
        self.pose
    }

    /// Colour as rgba in `0..=1`.
    #[inline]
    #[must_use]
    pub fn color(&self) -> [f64; 4] {
        self.color
    }

    /// MJCF geom group verbatim. URDF has none: `<visual>` is 0 and `<collision>` 3.
    #[inline]
    #[must_use]
    pub fn group(&self) -> u32 {
        self.group
    }

    pub(crate) fn new(shape: GeometryShape, pose: SE3<f64>, color: [f64; 4], group: u32) -> Self {
        VisualGeometry {
            shape,
            pose,
            color,
            group,
        }
    }
}

/// Resolves a mesh reference against a base directory and a package map.
pub(crate) fn resolve_mesh_path(
    file: &str,
    base_directory: Option<&Path>,
    packages: &[(String, PathBuf)],
) -> Option<PathBuf> {
    if let Some(rest) = file.strip_prefix("package://") {
        let (package, relative) = rest.split_once('/')?;
        let directory = packages
            .iter()
            .find(|(name, _)| name == package)
            .map(|(_, directory)| directory)?;
        return Some(directory.join(relative));
    }
    if let Some(rest) = file.strip_prefix("file://") {
        return Some(PathBuf::from(rest));
    }
    let path = Path::new(file);
    if path.is_absolute() {
        return Some(path.to_path_buf());
    }
    base_directory.map(|directory| directory.join(path))
}
