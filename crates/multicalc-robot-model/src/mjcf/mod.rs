//! MJCF reader.

mod body;
mod compiler;
mod defaults;
mod document;
mod geometry;
mod joint;
mod orientation;

use std::path::Path;

use crate::{BodyDescription, ModelError, ModelFormat, RobotModel};

/// Reads an MJCF file, resolving its `<include>` elements.
pub fn load_path(path: &Path) -> Result<RobotModel, ModelError> {
    let xml = document::assemble(path)?;
    load_str(&xml)
}

/// Parses MJCF from a string.
///
/// An `<include>` has no base directory to resolve against here and is rejected; use
/// [`load_path`].
pub fn load_str(xml: &str) -> Result<RobotModel, ModelError> {
    let document = roxmltree::Document::parse(xml).map_err(|e| ModelError::Xml(e.to_string()))?;
    if document
        .descendants()
        .any(|node| node.is_element() && node.tag_name().name() == "include")
    {
        return Err(ModelError::IncludeNeedsFile);
    }
    let parsed = body::read(&document)?;
    let bodies = parsed
        .bodies
        .into_iter()
        .map(|body| BodyDescription {
            name: body.name,
            parent: body.parent,
            pose: body.pose,
            inertia: body.inertia,
            joint: body.joint,
        })
        .collect();
    Ok(RobotModel {
        name: parsed.name,
        format: ModelFormat::Mjcf,
        bodies,
        floating_base: parsed.floating_base,
        ignored: parsed.ignored,
    })
}
