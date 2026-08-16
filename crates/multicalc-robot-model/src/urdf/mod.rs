//! Loads URDF model files.
//!
//! URDF states its links and the joints between them as one flat list, so the tree is worked out by
//! name rather than followed. Angles are always in radians and lengths in metres — URDF has no
//! setting for either.

mod joint;
mod link;
mod tree;

use std::path::Path;

use crate::xml::ignored_sections;
use crate::{ModelError, ModelFormat, RobotModel};

/// The top-level sections this reader takes something from.
const READ_SECTIONS: [&str; 2] = ["link", "joint"];

/// Loads a model from a file path.
pub fn load_path(path: &Path) -> Result<RobotModel, ModelError> {
    let xml = std::fs::read_to_string(path).map_err(|e| ModelError::Io(e.to_string()))?;
    load_str(&xml)
}

/// Parses a model from an in-memory XML string.
///
/// URDF has no way to pull in another file, so text and a file path read the same.
pub fn load_str(xml: &str) -> Result<RobotModel, ModelError> {
    let document = roxmltree::Document::parse(xml).map_err(|e| ModelError::Xml(e.to_string()))?;
    let root = document.root_element();
    if root.tag_name().name() != "robot" {
        return Err(ModelError::UnexpectedRootElement {
            found: root.tag_name().name().to_owned(),
        });
    }

    let bodies = tree::build(&link::read_links(root)?, &joint::read_joints(root)?)?;
    if bodies.is_empty() {
        return Err(ModelError::NoBodies);
    }

    Ok(RobotModel {
        name: root.attribute("name").unwrap_or("model").to_owned(),
        format: ModelFormat::Urdf,
        bodies,
        // Whether a robot is bolted down or free to move is a caller's decision when loading it,
        // not something a URDF states, so a model read from one is always bolted down.
        floating_base: false,
        ignored: ignored_sections(root, &READ_SECTIONS),
    })
}
