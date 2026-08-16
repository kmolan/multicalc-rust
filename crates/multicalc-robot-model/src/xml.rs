//! XML attribute and element access, shared by both readers.

use roxmltree::Node;

use crate::ModelError;

/// Child elements of `node` with the given tag.
pub(crate) fn elements<'a, 'input>(
    node: Node<'a, 'input>,
    tag: &'static str,
) -> impl Iterator<Item = Node<'a, 'input>> {
    node.children()
        .filter(move |child| child.is_element() && child.tag_name().name() == tag)
}

/// First child element of `node` with the given tag.
#[must_use]
pub(crate) fn element<'a, 'input>(
    node: Node<'a, 'input>,
    tag: &'static str,
) -> Option<Node<'a, 'input>> {
    elements(node, tag).next()
}

/// Error for an attribute that does not parse as the numbers it should hold.
#[must_use]
pub(crate) fn bad_attribute(node: Node, attribute: &str, text: &str) -> ModelError {
    ModelError::BadAttribute {
        element: node.tag_name().name().to_owned(),
        attribute: attribute.to_owned(),
        value: text.to_owned(),
    }
}

/// All values in a whitespace-separated attribute, of any count.
pub(crate) fn parse_list(node: Node, attribute: &str) -> Result<Option<Vec<f64>>, ModelError> {
    let Some(text) = node.attribute(attribute) else {
        return Ok(None);
    };
    let mut values = Vec::new();
    for field in text.split_ascii_whitespace() {
        match field.parse::<f64>() {
            Ok(value) => values.push(value),
            Err(_) => return Err(bad_attribute(node, attribute, text)),
        }
    }
    Ok(Some(values))
}

/// A scalar attribute.
pub(crate) fn parse_scalar(node: Node, attribute: &str) -> Result<Option<f64>, ModelError> {
    Ok(fixed::<1>(node, attribute)?.map(|[value]| value))
}

/// A 2-vector attribute.
#[cfg(feature = "mjcf")]
pub(crate) fn parse_vector2(node: Node, attribute: &str) -> Result<Option<[f64; 2]>, ModelError> {
    fixed::<2>(node, attribute)
}

/// A 3-vector attribute.
pub(crate) fn parse_vector3(node: Node, attribute: &str) -> Result<Option<[f64; 3]>, ModelError> {
    fixed::<3>(node, attribute)
}

/// A 4-vector attribute.
#[cfg(feature = "mjcf")]
pub(crate) fn parse_vector4(node: Node, attribute: &str) -> Result<Option<[f64; 4]>, ModelError> {
    fixed::<4>(node, attribute)
}

/// A 6-vector attribute.
#[cfg(feature = "mjcf")]
pub(crate) fn parse_vector6(node: Node, attribute: &str) -> Result<Option<[f64; 6]>, ModelError> {
    fixed::<6>(node, attribute)
}

/// A fixed-count attribute. A wrong count is an error, not a truncation.
pub(crate) fn fixed<const N: usize>(
    node: Node,
    attribute: &str,
) -> Result<Option<[f64; N]>, ModelError> {
    let Some(values) = parse_list(node, attribute)? else {
        return Ok(None);
    };
    match <[f64; N]>::try_from(values) {
        Ok(fixed) => Ok(Some(fixed)),
        Err(_) => Err(bad_attribute(
            node,
            attribute,
            node.attribute(attribute).unwrap_or_default(),
        )),
    }
}

/// Top-level elements the reader consumes nothing from. Sorted and deduplicated, so generated
/// output is stable across runs.
#[must_use]
pub(crate) fn ignored_sections(root: Node, read: &[&str]) -> Vec<String> {
    let mut names: Vec<String> = root
        .children()
        .filter(Node::is_element)
        .map(|section| section.tag_name().name())
        .filter(|name| !read.contains(name))
        .map(str::to_owned)
        .collect();
    names.sort();
    names.dedup();
    names
}
