//! Loads MuJoCo MJCF model files into multicalc's rigid-body types.
//!
//! This reads one rigid body attached to the world by a free joint: its name, where it sits, and
//! how its mass is distributed. Where a model states its own mass properties they are used
//! directly; where it leaves them to be worked out from the shapes it is built from, they are
//! computed from those shapes.
//!
//! Anything outside that — chains of joints, meshes as a source of mass, files that pull in other
//! files — is refused by name rather than quietly ignored, so a model can never load with the wrong
//! mass.
//!
//! ```no_run
//! use std::path::Path;
//!
//! let model = multicalc_mjcf::load_path(Path::new("third_party/menagerie/skydio_x2/x2.xml"))?;
//! assert!(model.has_free_joint());
//! println!("{} weighs {} kg", model.name(), model.inertia().mass());
//! # Ok::<(), multicalc_mjcf::MjcfError>(())
//! ```

mod body;
mod defaults;
mod error;
mod geometry;

use std::path::Path;

use multicalc::spatial::{SE3, SpatialInertia};

pub use error::MjcfError;

/// One rigid body read from a model file.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidBodyModel {
    name: String,
    pose: SE3<f64>,
    has_free_joint: bool,
    inertia: SpatialInertia<f64>,
}

impl RigidBodyModel {
    /// The body's name, or `body` where the file gives none.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where the body sits, and how it is turned, relative to the world.
    #[inline]
    #[must_use]
    pub fn pose(&self) -> SE3<f64> {
        self.pose
    }

    /// Whether the body hangs off the world by a free joint.
    #[inline]
    #[must_use]
    pub fn has_free_joint(&self) -> bool {
        self.has_free_joint
    }

    /// The body's mass, the point it balances about, and how it resists being spun.
    #[inline]
    #[must_use]
    pub fn inertia(&self) -> SpatialInertia<f64> {
        self.inertia
    }
}

/// Reads a model from a file.
pub fn load_path(path: &Path) -> Result<RigidBodyModel, MjcfError> {
    let xml = std::fs::read_to_string(path).map_err(|e| MjcfError::Io(e.to_string()))?;
    load_str(&xml)
}

/// Reads a model from XML text already in memory.
pub fn load_str(xml: &str) -> Result<RigidBodyModel, MjcfError> {
    let document = roxmltree::Document::parse(xml).map_err(|e| MjcfError::Xml(e.to_string()))?;
    let body = body::read(&document)?;
    Ok(RigidBodyModel {
        name: body.name,
        pose: body.pose,
        has_free_joint: body.has_free_joint,
        inertia: body.inertia,
    })
}
