//! `<default>` class inheritance.
//!
//! Blocks nest, each inheriting its parent's settings; a geom selects one by `class`, a body sets
//! one for its subtree by `childclass`. Only `<geom>` and `<joint>` settings are read — nothing
//! else in a block can affect mass properties.

use std::collections::HashMap;

use roxmltree::Node;

use crate::ModelError;
use crate::xml::{
    elements, fixed, parse_list, parse_scalar, parse_vector2, parse_vector3, parse_vector4,
};

/// Geom settings from one default class. `None` means unset at this level.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct GeomDefaults {
    pub geom_type: Option<String>,
    pub size: Option<Vec<f64>>,
    pub fromto: Option<[f64; 6]>,
    pub pos: Option<[f64; 3]>,
    pub quat: Option<[f64; 4]>,
    pub mass: Option<f64>,
    pub density: Option<f64>,
}

impl GeomDefaults {
    /// Settings on one `<geom>` element.
    pub(crate) fn read(node: Node) -> Result<Self, ModelError> {
        Ok(GeomDefaults {
            geom_type: node.attribute("type").map(str::to_owned),
            size: parse_list(node, "size")?,
            fromto: fixed::<6>(node, "fromto")?,
            pos: parse_vector3(node, "pos")?,
            quat: parse_vector4(node, "quat")?,
            mass: parse_scalar(node, "mass")?,
            density: parse_scalar(node, "density")?,
        })
    }

    /// These settings with everything `other` states applied over the top.
    #[must_use]
    pub(crate) fn overridden_by(&self, other: &GeomDefaults) -> GeomDefaults {
        GeomDefaults {
            geom_type: other.geom_type.clone().or_else(|| self.geom_type.clone()),
            size: other.size.clone().or_else(|| self.size.clone()),
            fromto: other.fromto.or(self.fromto),
            pos: other.pos.or(self.pos),
            quat: other.quat.or(self.quat),
            mass: other.mass.or(self.mass),
            density: other.density.or(self.density),
        }
    }
}

/// Joint settings from one default class. `None` means unset at this level.
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
    /// Settings on one `<joint>` element.
    pub(crate) fn read(node: Node) -> Result<Self, ModelError> {
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

    /// These settings with everything `other` states applied over the top.
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

/// One default class: its geom and joint settings.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ClassDefaults {
    pub geom: GeomDefaults,
    pub joint: JointDefaults,
}

/// Every named class, pre-flattened so lookup needs no traversal.
#[derive(Debug, Default)]
pub(crate) struct DefaultTable {
    root: ClassDefaults,
    classes: HashMap<String, ClassDefaults>,
}

impl DefaultTable {
    /// Parses every `<default>` block.
    pub(crate) fn build(root: Node) -> Result<Self, ModelError> {
        let mut table = DefaultTable::default();
        for node in elements(root, "default") {
            table.walk(node, &ClassDefaults::default())?;
        }
        Ok(table)
    }

    /// A class's settings, or the unnamed block's where no class is named.
    pub(crate) fn resolve(&self, class: Option<&str>) -> Result<&ClassDefaults, ModelError> {
        match class {
            None => Ok(&self.root),
            Some(name) => self
                .classes
                .get(name)
                .ok_or_else(|| ModelError::UndefinedClass {
                    name: name.to_owned(),
                }),
        }
    }

    /// Flattens one block, then its nested blocks, each starting from what it inherits.
    fn walk(&mut self, node: Node, inherited: &ClassDefaults) -> Result<(), ModelError> {
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

/// Orientation attributes this reader does not handle; only `pos`/`quat` are read.
const UNSUPPORTED_ORIENTATION_ATTRIBUTES: [&str; 4] = ["euler", "axisangle", "xyaxes", "zaxis"];

/// Rejects an element stating orientation other than by `quat`.
pub(crate) fn reject_orientation_attributes(node: Node, element: &str) -> Result<(), ModelError> {
    for attribute in UNSUPPORTED_ORIENTATION_ATTRIBUTES {
        if node.has_attribute(attribute) {
            return Err(ModelError::UnsupportedOrientation {
                element: element.to_owned(),
                attribute: attribute.to_owned(),
            });
        }
    }
    Ok(())
}
