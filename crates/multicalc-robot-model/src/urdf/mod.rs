//! URDF reader.
//!
//! `<link>` and `<joint>` are a flat, unordered list, so the kinematic tree is resolved by name.
//! Units are fixed by the spec: metres and radians.

mod joint;
mod link;
mod tree;
mod visual;

use std::path::Path;

use crate::xml::ignored_sections;
use crate::{ModelError, ModelFormat, RobotModel};

/// Top-level elements this reader consumes.
const READ_SECTIONS: [&str; 3] = ["link", "joint", "material"];

/// Reads a URDF file.
pub fn load_path(path: &Path) -> Result<RobotModel, ModelError> {
    let xml = std::fs::read_to_string(path).map_err(|err| ModelError::FileRead(err.to_string()))?;
    let mut model = load_str(&xml)?;
    model.base_directory = path.parent().map(Path::to_path_buf);
    Ok(model)
}

/// Parses URDF from a string.
///
/// URDF has no include mechanism, so this is equivalent to [`load_path`].
pub fn load_str(xml: &str) -> Result<RobotModel, ModelError> {
    let document =
        roxmltree::Document::parse(xml).map_err(|err| ModelError::Xml(err.to_string()))?;
    let root = document.root_element();
    if root.tag_name().name() != "robot" {
        return Err(ModelError::UnexpectedRootElement {
            found: root.tag_name().name().to_owned(),
        });
    }

    let materials = visual::read_materials(root)?;
    let bodies = tree::build(
        &link::read_links(root, &materials)?,
        &joint::read_joints(root)?,
    )?;
    if bodies.is_empty() {
        return Err(ModelError::NoBodies);
    }

    Ok(RobotModel {
        name: root.attribute("name").unwrap_or("model").to_owned(),
        format: ModelFormat::Urdf,
        bodies,
        // Floating base is a load-time choice by the caller, not a file property. See tree.rs.
        floating_base: false,
        ignored: ignored_sections(root, &READ_SECTIONS),
        base_directory: None,
    })
}
