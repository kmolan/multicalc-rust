//! Loads MuJoCo MJCF model files.

mod body;
mod compiler;
mod defaults;
mod document;
mod geometry;
mod joint;

use std::path::Path;

use crate::{BodyRecord, ModelError, RobotModel};

/// Loads a model from a file path, resolving any `<include>` elements it pulls in.
pub fn load_path(path: &Path) -> Result<RobotModel, ModelError> {
    let xml = document::assemble(path)?;
    load_str(&xml)
}

/// Parses a model from an in-memory XML string.
///
/// Text has no directory to resolve an `<include>` against, so a document that pulls in another
/// file is refused here; [`load_path`] resolves those first.
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
        .map(|body| BodyRecord {
            name: body.name,
            parent: body.parent,
            pose: body.pose,
            inertia: body.inertia,
            joint: body.joint,
        })
        .collect();
    Ok(RobotModel {
        name: parsed.name,
        bodies,
        floating_base: parsed.floating_base,
        ignored: parsed.ignored,
    })
}
