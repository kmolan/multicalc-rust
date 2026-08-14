//! Resolves `<include file="...">` elements before the document is parsed for real.

use std::ops::Range;
use std::path::Path;

use roxmltree::{Document, Node};

use crate::MjcfError;
use crate::defaults::bad_attribute;

/// How deep a chain of includes may nest before this loader gives up. Also what catches a cycle.
const MAX_INCLUDE_DEPTH: usize = 8;

/// Reads a file and splices in everything it pulls in, giving one piece of text to parse.
pub(crate) fn assemble(path: &Path) -> Result<String, MjcfError> {
    read(path, 0)
}

/// Reads one file and resolves its own `<include>` elements, recursing into each included file.
fn read(path: &Path, depth: usize) -> Result<String, MjcfError> {
    if depth > MAX_INCLUDE_DEPTH {
        return Err(MjcfError::IncludeTooDeep { depth });
    }

    let mut text = std::fs::read_to_string(path).map_err(|e| MjcfError::Io(e.to_string()))?;
    let document = Document::parse(&text).map_err(|e| MjcfError::Xml(e.to_string()))?;

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

/// The source text of `xml`'s root's element children, joined with `"\n"` — everything inside
/// its `<mujoco>...</mujoco>` wrapper, with the wrapper itself dropped.
fn inner_source(xml: &str) -> Result<String, MjcfError> {
    let document = Document::parse(xml).map_err(|e| MjcfError::Xml(e.to_string()))?;
    let pieces: Vec<&str> = document
        .root_element()
        .children()
        .filter(Node::is_element)
        .map(|child| &xml[child.range()])
        .collect();
    Ok(pieces.join("\n"))
}
