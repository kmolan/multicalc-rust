//! The shape settings a model states once in a `<default>` block and reuses.
//!
//! A block can name itself with a class, and a shape can name the block it wants. Blocks nest, an
//! inner one starting from its parent's settings, and a body can name a block for every shape
//! inside it. Only `<geom>` settings are read: nothing else in a block can change a mass.

use std::collections::HashMap;

use roxmltree::Node;

use crate::MjcfError;
use crate::orientation::Orientation;

/// The geom settings one default class supplies, with `None` meaning "not set here".
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct GeomDefaults {
    pub geom_type: Option<String>,
    pub size: Option<Vec<f64>>,
    pub fromto: Option<[f64; 6]>,
    pub pos: Option<[f64; 3]>,
    pub orientation: Orientation,
    pub mass: Option<f64>,
    pub density: Option<f64>,
}

impl GeomDefaults {
    /// The settings written on one `<geom>` element.
    pub(crate) fn read(node: Node) -> Result<Self, MjcfError> {
        Ok(GeomDefaults {
            geom_type: node.attribute("type").map(str::to_owned),
            size: parse_list(node, "size")?,
            fromto: fixed::<6>(node, "fromto")?,
            pos: parse_vector3(node, "pos")?,
            orientation: Orientation::read(node)?,
            mass: parse_scalar(node, "mass")?,
            density: parse_scalar(node, "density")?,
        })
    }

    /// A copy of these settings with everything `other` states written over the top.
    #[must_use]
    pub(crate) fn overridden_by(&self, other: &GeomDefaults) -> GeomDefaults {
        GeomDefaults {
            geom_type: other.geom_type.clone().or_else(|| self.geom_type.clone()),
            size: other.size.clone().or_else(|| self.size.clone()),
            fromto: other.fromto.or(self.fromto),
            pos: other.pos.or(self.pos),
            orientation: self.orientation.overridden_by(other.orientation),
            mass: other.mass.or(self.mass),
            density: other.density.or(self.density),
        }
    }
}

/// The joint settings one default class supplies, with `None` meaning "not set here".
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct JointDefaults {
    pub joint_type: Option<String>,
    pub axis: Option<[f64; 3]>,
    pub pos: Option<[f64; 3]>,
    pub range: Option<[f64; 2]>,
    pub limited: Option<String>,
    pub armature: Option<f64>,
    pub damping: Option<f64>,
    pub friction_loss: Option<f64>,
    pub reference: Option<f64>,
    pub spring_reference: Option<f64>,
    pub stiffness: Option<f64>,
}

impl JointDefaults {
    /// The settings written on one `<joint>` element.
    pub(crate) fn read(node: Node) -> Result<Self, MjcfError> {
        Ok(JointDefaults {
            joint_type: node.attribute("type").map(str::to_owned),
            axis: parse_vector3(node, "axis")?,
            pos: parse_vector3(node, "pos")?,
            range: parse_vector2(node, "range")?,
            limited: node.attribute("limited").map(str::to_owned),
            armature: parse_scalar(node, "armature")?,
            damping: parse_scalar(node, "damping")?,
            friction_loss: parse_scalar(node, "frictionloss")?,
            reference: parse_scalar(node, "ref")?,
            spring_reference: parse_scalar(node, "springref")?,
            stiffness: parse_scalar(node, "stiffness")?,
        })
    }

    /// A copy of these settings with everything `other` states written over the top.
    #[must_use]
    pub(crate) fn overridden_by(&self, other: &JointDefaults) -> JointDefaults {
        JointDefaults {
            joint_type: other.joint_type.clone().or_else(|| self.joint_type.clone()),
            axis: other.axis.or(self.axis),
            pos: other.pos.or(self.pos),
            range: other.range.or(self.range),
            limited: other.limited.clone().or_else(|| self.limited.clone()),
            armature: other.armature.or(self.armature),
            damping: other.damping.or(self.damping),
            friction_loss: other.friction_loss.or(self.friction_loss),
            reference: other.reference.or(self.reference),
            spring_reference: other.spring_reference.or(self.spring_reference),
            stiffness: other.stiffness.or(self.stiffness),
        }
    }
}

/// One default class's geom and joint settings together.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ClassDefaults {
    pub geom: GeomDefaults,
    pub joint: JointDefaults,
}

/// Every named class in the file, already flattened so a lookup needs no walking.
#[derive(Debug, Default)]
pub(crate) struct DefaultTable {
    root: ClassDefaults,
    classes: HashMap<String, ClassDefaults>,
}

impl DefaultTable {
    /// Reads every `<default>` block in the file.
    pub(crate) fn build(root: Node) -> Result<Self, MjcfError> {
        let mut table = DefaultTable::default();
        for node in elements(root, "default") {
            table.walk(node, &ClassDefaults::default())?;
        }
        Ok(table)
    }

    /// The settings a class supplies, or the unnamed block's settings when no class is named.
    pub(crate) fn resolve(&self, class: Option<&str>) -> Result<&ClassDefaults, MjcfError> {
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
    fn walk(&mut self, node: Node, inherited: &ClassDefaults) -> Result<(), MjcfError> {
        let mut settings = inherited.clone();
        for geom in elements(node, "geom") {
            settings.geom = settings.geom.overridden_by(&GeomDefaults::read(geom)?);
        }
        for joint in elements(node, "joint") {
            settings.joint = settings.joint.overridden_by(&JointDefaults::read(joint)?);
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
#[must_use]
pub(crate) fn element<'a, 'input>(
    node: Node<'a, 'input>,
    tag: &'static str,
) -> Option<Node<'a, 'input>> {
    elements(node, tag).next()
}

/// The error for an attribute that does not hold the numbers it should.
#[must_use]
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

/// The two numbers an attribute holds.
pub(crate) fn parse_vector2(node: Node, attribute: &str) -> Result<Option<[f64; 2]>, MjcfError> {
    fixed::<2>(node, attribute)
}

/// The three numbers an attribute holds.
pub(crate) fn parse_vector3(node: Node, attribute: &str) -> Result<Option<[f64; 3]>, MjcfError> {
    fixed::<3>(node, attribute)
}

/// The four numbers an attribute holds.
pub(crate) fn parse_vector4(node: Node, attribute: &str) -> Result<Option<[f64; 4]>, MjcfError> {
    fixed::<4>(node, attribute)
}

/// The six numbers an attribute holds.
pub(crate) fn parse_vector6(node: Node, attribute: &str) -> Result<Option<[f64; 6]>, MjcfError> {
    fixed::<6>(node, attribute)
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
