//! `<joint>` parsing: topology, placement, axis, limits, dynamics, mimic.

use multicalc::kinematics::JointKind;
use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;
use roxmltree::Node;

use crate::urdf::link::read_origin;
use crate::xml::{bad_attribute, element, elements, parse_scalar, parse_vector3};
use crate::{JointDescription, MimicDescription, ModelError};

/// URDF's default joint axis. Note MJCF's is `[0, 0, 1]`.
const DEFAULT_AXIS: [f64; 3] = [1.0, 0.0, 0.0];

/// A `<joint>` as stated, before tree resolution.
pub(crate) struct ParsedJoint {
    pub name: String,
    pub parent_link: String,
    pub child_link: String,
    pub origin: SE3<f64>,
    /// `None` for a fixed joint: the child body is welded and carries no joint.
    pub description: Option<JointDescription>,
}

/// Every `<joint>` child of `<robot>`, in document order.
pub(crate) fn read_joints(root: Node) -> Result<Vec<ParsedJoint>, ModelError> {
    let mut joints = Vec::new();
    for node in elements(root, "joint") {
        joints.push(read_joint(node)?);
    }
    Ok(joints)
}

/// One `<joint>`. Types outside the supported set are rejected by name.
fn read_joint(node: Node) -> Result<ParsedJoint, ModelError> {
    let name = node.attribute("name").unwrap_or("joint").to_owned();
    let parent_link = linked(node, "parent")?;
    let child_link = linked(node, "child")?;

    let kind = match node.attribute("type").unwrap_or_default() {
        "revolute" => JointKind::Revolute,
        "continuous" => JointKind::Continuous,
        "prismatic" => JointKind::Prismatic,
        "fixed" => JointKind::Fixed,
        "floating" => JointKind::Floating,
        other => {
            return Err(ModelError::UnsupportedJoint {
                // The joint belongs to its child link, which is the body it maps to.
                body: child_link,
                joint_type: other.to_owned(),
            });
        }
    };

    let origin = read_origin(node)?;

    let description = match kind {
        JointKind::Fixed => None,
        JointKind::Floating => Some(JointDescription::floating(name.clone())),
        JointKind::Revolute | JointKind::Continuous | JointKind::Prismatic => {
            Some(movable(node, &name, kind)?)
        }
    };

    Ok(ParsedJoint {
        name,
        parent_link,
        child_link,
        origin,
        description,
    })
}

/// A movable joint: axis, travel limits, and joint dynamics.
fn movable(node: Node, name: &str, kind: JointKind) -> Result<JointDescription, ModelError> {
    let axis = match element(node, "axis") {
        Some(axis_node) => {
            let stated = parse_vector3(axis_node, "xyz")?.unwrap_or(DEFAULT_AXIS);
            Vector::new(stated).try_normalized().ok_or_else(|| {
                bad_attribute(
                    axis_node,
                    "xyz",
                    axis_node.attribute("xyz").unwrap_or_default(),
                )
            })?
        }
        None => Vector::new(DEFAULT_AXIS),
    };

    // Continuous joints are unbounded. URDF still requires `<limit>` on them for effort and
    // velocity, where lower/upper carry no meaning.
    let limits = match kind {
        JointKind::Continuous => None,
        _ => Some(read_limits(node, name)?),
    };

    let dynamics = element(node, "dynamics");
    let setting = |attribute: &'static str| -> Result<f64, ModelError> {
        match dynamics {
            Some(dynamics_node) => Ok(parse_scalar(dynamics_node, attribute)?.unwrap_or(0.0)),
            None => Ok(0.0),
        }
    };

    Ok(JointDescription {
        name: name.to_owned(),
        kind,
        axis,
        // A URDF joint sits at its child link frame's origin, so the anchor offset is always zero.
        anchor: Vector::zeros(),
        limits,
        damping: setting("damping")?,
        friction_loss: setting("friction")?,
        // No URDF equivalent: no `ref`, armature, springref or stiffness.
        zero_offset: 0.0,
        armature: 0.0,
        spring_reference: 0.0,
        spring_stiffness: 0.0,
        mimic: read_mimic(node)?,
    })
}

/// Travel limits. Both bounds are required on revolute and prismatic joints.
fn read_limits(node: Node, name: &str) -> Result<(f64, f64), ModelError> {
    let needs_limit = || ModelError::JointNeedsLimit {
        joint: name.to_owned(),
    };
    let limit_node = element(node, "limit").ok_or_else(needs_limit)?;
    let lower = parse_scalar(limit_node, "lower")?.ok_or_else(needs_limit)?;
    let upper = parse_scalar(limit_node, "upper")?.ok_or_else(needs_limit)?;
    Ok((lower, upper))
}

/// The `<mimic>` coupling, if stated. Defaults: multiplier 1, offset 0.
fn read_mimic(node: Node) -> Result<Option<MimicDescription>, ModelError> {
    let Some(mimic_node) = element(node, "mimic") else {
        return Ok(None);
    };
    let joint = mimic_node
        .attribute("joint")
        .ok_or_else(|| bad_attribute(mimic_node, "joint", ""))?
        .to_owned();
    Ok(Some(MimicDescription {
        joint,
        multiplier: parse_scalar(mimic_node, "multiplier")?.unwrap_or(1.0),
        offset: parse_scalar(mimic_node, "offset")?.unwrap_or(0.0),
    }))
}

/// The link named by a `<parent>` or `<child>` element.
fn linked(node: Node, tag: &'static str) -> Result<String, ModelError> {
    element(node, tag)
        .and_then(|child| child.attribute("link"))
        .map(str::to_owned)
        .ok_or_else(|| bad_attribute(node, tag, ""))
}
