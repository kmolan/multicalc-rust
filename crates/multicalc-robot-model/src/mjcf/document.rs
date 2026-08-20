//! `<include file="...">` splicing, done before the document is parsed.

use std::ops::Range;
use std::path::Path;

use roxmltree::{Document, Node};

use crate::ModelError;
use crate::xml::bad_attribute;

/// Include nesting depth limit. Also what catches an include cycle.
const MAX_INCLUDE_DEPTH: usize = 8;

/// Reads a file and splices in its includes, yielding one document to parse.
pub(crate) fn assemble(path: &Path) -> Result<String, ModelError> {
    read(path, 0)
}

/// Reads one file, recursing into each `<include>` it names.
fn read(path: &Path, depth: usize) -> Result<String, ModelError> {
    if depth > MAX_INCLUDE_DEPTH {
        return Err(ModelError::IncludeTooDeep { depth });
    }

    let mut text =
        std::fs::read_to_string(path).map_err(|err| ModelError::FileRead(err.to_string()))?;
    let document = Document::parse(&text).map_err(|err| ModelError::Xml(err.to_string()))?;

    let mut includes: Vec<(Range<usize>, String)> = Vec::new();
    for node in document.descendants() {
        if node.is_element() && node.tag_name().name() == "include" {
            let file = node
                .attribute("file")
                .ok_or_else(|| bad_attribute(node, "file", ""))?;
            includes.push((node.range(), file.to_owned()));
        }
    }

    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    for (range, file) in includes.into_iter().rev() {
        let included = read(&directory.join(&file), depth + 1)?;
        text.replace_range(range, &inner_source(&included)?);
    }

    Ok(text)
}

/// The source text of the root's element children, joined with `"\n"`: the contents of the
/// `<mujoco>` wrapper, without the wrapper.
fn inner_source(xml: &str) -> Result<String, ModelError> {
    let document = Document::parse(xml).map_err(|err| ModelError::Xml(err.to_string()))?;
    let pieces: Vec<&str> = document
        .root_element()
        .children()
        .filter(Node::is_element)
        .map(|child| &xml[child.range()])
        .collect();
    Ok(pieces.join("\n"))
}
