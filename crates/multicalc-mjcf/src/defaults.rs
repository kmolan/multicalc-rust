//! The shape settings a model states once in a `<default>` block and reuses.
//!
//! A block can name itself with a class, and a shape can name the block it wants. Blocks nest, an
//! inner one starting from its parent's settings, and a body can name a block for every shape
//! inside it. Only `<geom>` settings are read: nothing else in a block can change a mass.

use std::collections::HashMap;

use multicalc::spatial::Quaternion;
use roxmltree::Node;

use crate::MjcfError;

/// The geom settings one default class supplies, with `None` meaning "not set here".
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct GeomDefaults {
    pub geom_type: Option<String>,
    pub size: Option<Vec<f64>>,
    pub pos: Option<[f64; 3]>,
    pub quat: Option<[f64; 4]>,
    pub mass: Option<f64>,
    pub density: Option<f64>,
}

impl GeomDefaults {
    /// The settings written on one `<geom>` element.
    pub(crate) fn read(node: Node) -> Result<Self, MjcfError> {
        Ok(GeomDefaults {
            geom_type: node.attribute("type").map(str::to_owned),
            size: parse_list(node, "size")?,
            pos: parse_vector3(node, "pos")?,
            quat: parse_vector4(node, "quat")?,
            mass: parse_scalar(node, "mass")?,
            density: parse_scalar(node, "density")?,
        })
    }

    /// A copy of these settings with everything `other` states written over the top.
    pub(crate) fn overridden_by(&self, other: &GeomDefaults) -> GeomDefaults {
        GeomDefaults {
            geom_type: other.geom_type.clone().or_else(|| self.geom_type.clone()),
            size: other.size.clone().or_else(|| self.size.clone()),
            pos: other.pos.or(self.pos),
            quat: other.quat.or(self.quat),
            mass: other.mass.or(self.mass),
            density: other.density.or(self.density),
        }
    }
}

/// Every named class in the file, already flattened so a lookup needs no walking.
#[derive(Debug, Default)]
pub(crate) struct DefaultTable {
    root: GeomDefaults,
    classes: HashMap<String, GeomDefaults>,
}

impl DefaultTable {
    /// Reads every `<default>` block in the file.
    pub(crate) fn build(root: Node) -> Result<Self, MjcfError> {
        let mut table = DefaultTable::default();
        for node in elements(root, "default") {
            table.walk(node, &GeomDefaults::default())?;
        }
        Ok(table)
    }

    /// The settings a class supplies, or the unnamed block's settings when no class is named.
    pub(crate) fn resolve(&self, class: Option<&str>) -> Result<&GeomDefaults, MjcfError> {
        match class {
            None => Ok(&self.root),
            Some(name) => self
                .classes
                .get(name)
                .ok_or_else(|| MjcfError::UndefinedClass {
                    name: name.to_owned(),
                }),
        }
    }

    /// Records one block, then its nested blocks, each starting from what it inherits.
    fn walk(&mut self, node: Node, inherited: &GeomDefaults) -> Result<(), MjcfError> {
        let mut settings = inherited.clone();
        for geom in elements(node, "geom") {
            settings = settings.overridden_by(&GeomDefaults::read(geom)?);
        }

        match node.attribute("class") {
            Some(name) => {
                self.classes.insert(name.to_owned(), settings.clone());
            }
            None => self.root = settings.clone(),
        }

        for nested in elements(node, "default") {
            self.walk(nested, &settings)?;
        }
        Ok(())
    }
}

/// The child elements of `node` with the given tag.
pub(crate) fn elements<'a, 'input>(
    node: Node<'a, 'input>,
    tag: &'static str,
) -> impl Iterator<Item = Node<'a, 'input>> {
    node.children()
        .filter(move |child| child.is_element() && child.tag_name().name() == tag)
}

/// The first child element of `node` with the given tag.
pub(crate) fn element<'a, 'input>(
    node: Node<'a, 'input>,
    tag: &'static str,
) -> Option<Node<'a, 'input>> {
    elements(node, tag).next()
}

/// The turn a `quat` attribute describes, as a unit quaternion. MJCF writes the scalar part
/// first, which is also how the crate stores it, so the four numbers carry straight over.
pub(crate) fn unit_quaternion(
    node: Node,
    quat: [f64; 4],
    attribute: &'static str,
) -> Result<Quaternion<f64>, MjcfError> {
    Quaternion::new(quat[0], quat[1], quat[2], quat[3])
        .try_normalized()
        .ok_or_else(|| {
            bad_attribute(
                node,
                attribute,
                node.attribute(attribute).unwrap_or_default(),
            )
        })
}

/// The error for an attribute that does not hold the numbers it should.
pub(crate) fn bad_attribute(node: Node, attribute: &str, text: &str) -> MjcfError {
    MjcfError::BadAttribute {
        element: node.tag_name().name().to_owned(),
        attribute: attribute.to_owned(),
        value: text.to_owned(),
    }
}

/// Every number an attribute holds, however many there are.
pub(crate) fn parse_list(node: Node, attribute: &str) -> Result<Option<Vec<f64>>, MjcfError> {
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

/// The one number an attribute holds.
pub(crate) fn parse_scalar(node: Node, attribute: &str) -> Result<Option<f64>, MjcfError> {
    Ok(fixed::<1>(node, attribute)?.map(|[value]| value))
}

/// The three numbers an attribute holds.
pub(crate) fn parse_vector3(node: Node, attribute: &str) -> Result<Option<[f64; 3]>, MjcfError> {
    fixed::<3>(node, attribute)
}

/// The four numbers an attribute holds.
pub(crate) fn parse_vector4(node: Node, attribute: &str) -> Result<Option<[f64; 4]>, MjcfError> {
    fixed::<4>(node, attribute)
}

/// The numbers an attribute holds, where the count is fixed. Too many or too few is as much an
/// error as text that is not a number at all.
fn fixed<const N: usize>(node: Node, attribute: &str) -> Result<Option<[f64; N]>, MjcfError> {
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
