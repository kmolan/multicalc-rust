//! Reading the joints a URDF file describes: which links they connect, where they sit, how they
//! move, and what they are driven by.

use multicalc::kinematics::JointKind;
use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;
use roxmltree::Node;

use crate::urdf::link::read_origin;
use crate::xml::{bad_attribute, element, elements, parse_scalar, parse_vector3};
use crate::{JointDescription, MimicDescription, ModelError};

/// The axis a joint turns or slides about where the file states none.
const ASSUMED_AXIS: [f64; 3] = [1.0, 0.0, 0.0];

/// One joint as the file states it, before the tree is worked out.
pub(crate) struct ParsedJoint {
    pub name: String,
    pub parent_link: String,
    pub child_link: String,
    pub origin: SE3<f64>,
    /// How the child link moves, or `None` where the joint is fixed and the link is welded to its
    /// parent — which is the one kind a body carries no joint for.
    pub description: Option<JointDescription>,
}

/// Reads every `<joint>` child of `<robot>`, in document order.
pub(crate) fn read_joints(root: Node) -> Result<Vec<ParsedJoint>, ModelError> {
    let mut joints = Vec::new();
    for node in elements(root, "joint") {
        joints.push(read_joint(node)?);
    }
    Ok(joints)
}

/// Reads one `<joint>`, refusing a kind outside this reader's subset by name.
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
                // The joint belongs to the link it drives, which is the body it becomes here.
                body: child_link,
                joint_type: other.to_owned(),
            });
        }
    };

    let origin = read_origin(node)?;

    let description = match kind {
        // A welded link carries no joint at all, and a free one takes every setting from the
        // shared constructor.
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

/// A joint that can travel: its axis, how far it goes, and what resists it.
fn movable(node: Node, name: &str, kind: JointKind) -> Result<JointDescription, ModelError> {
    let axis = match element(node, "axis") {
        Some(axis_node) => {
            let stated = parse_vector3(axis_node, "xyz")?.unwrap_or(ASSUMED_AXIS);
            Vector::new(stated).try_normalized().ok_or_else(|| {
                bad_attribute(
                    axis_node,
                    "xyz",
                    axis_node.attribute("xyz").unwrap_or_default(),
                )
            })?
        }
        None => Vector::new(ASSUMED_AXIS),
    };

    // A joint that can only turn round and round has nowhere to stop. URDF still asks for a
    // `<limit>` there, to carry the effort and speed figures, and its lower and upper mean nothing.
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
        // A URDF joint sits at the origin of the link it drives, so there is no offset to carry.
        anchor: Vector::zeros(),
        limits,
        damping: setting("damping")?,
        friction_loss: setting("friction")?,
        // URDF states none of these, so they stay at the value that makes them do nothing.
        zero_offset: 0.0,
        armature: 0.0,
        spring_reference: 0.0,
        spring_stiffness: 0.0,
        mimic: read_mimic(node)?,
    })
}

/// How far a joint that can stop is allowed to travel. URDF states both ends or neither.
fn read_limits(node: Node, name: &str) -> Result<(f64, f64), ModelError> {
    let needs_limit = || ModelError::JointNeedsLimit {
        joint: name.to_owned(),
    };
    let limit_node = element(node, "limit").ok_or_else(needs_limit)?;
    let lower = parse_scalar(limit_node, "lower")?.ok_or_else(needs_limit)?;
    let upper = parse_scalar(limit_node, "upper")?.ok_or_else(needs_limit)?;
    Ok((lower, upper))
}

/// The joint this one follows, where the file says it follows one.
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

/// The link a `<parent>` or `<child>` element names.
fn linked(node: Node, tag: &'static str) -> Result<String, ModelError> {
    element(node, tag)
        .and_then(|child| child.attribute("link"))
        .map(str::to_owned)
        .ok_or_else(|| bad_attribute(node, tag, ""))
}
