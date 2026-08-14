//! Reads the joint one body carries: kind, axis, anchor, travel limits, and the dynamics
//! parameters forward kinematics does not use.

use multicalc::kinematics::JointKind;
use multicalc::linear_algebra::Vector;
use roxmltree::Node;

use crate::JointRecord;
use crate::MjcfError;
use crate::compiler::CompilerSettings;
use crate::defaults::{DefaultTable, JointDefaults, bad_attribute};

/// MJCF's default joint type when a `<joint>` states none.
const ASSUMED_JOINT_TYPE: &str = "hinge";

/// Reads the joint a body carries, or `None` where it carries none.
///
/// A free joint is resolved by the caller before this is reached; a body arriving here carries at
/// most one ordinary (hinge or slide) joint.
pub(crate) fn read_joint(
    body: Node,
    table: &DefaultTable,
    class_chain: Option<&str>,
    settings: &CompilerSettings,
    body_name: &str,
) -> Result<Option<JointRecord>, MjcfError> {
    let joints: Vec<Node> = body
        .children()
        .filter(|child| {
            child.is_element() && matches!(child.tag_name().name(), "joint" | "freejoint")
        })
        .collect();

    let node = match joints.as_slice() {
        [] => return Ok(None),
        [only] => *only,
        many => {
            return Err(MjcfError::MultipleJoints {
                body: body_name.to_owned(),
                count: many.len(),
            });
        }
    };

    let resolved = effective(node, table, class_chain)?;

    let joint_type = resolved.joint_type.as_deref().unwrap_or(ASSUMED_JOINT_TYPE);
    let kind = match joint_type {
        "hinge" => JointKind::Revolute,
        "slide" => JointKind::Prismatic,
        other => {
            return Err(MjcfError::UnsupportedJoint {
                body: body_name.to_owned(),
                joint_type: other.to_owned(),
            });
        }
    };
    let is_revolute = matches!(kind, JointKind::Revolute);
    let to_radians = |value: f64| {
        if is_revolute {
            settings.to_radians(value)
        } else {
            value
        }
    };

    let axis = Vector::new(resolved.axis.unwrap_or([0.0, 0.0, 1.0]))
        .try_normalized()
        .ok_or_else(|| bad_attribute(node, "axis", node.attribute("axis").unwrap_or_default()))?;
    let anchor = Vector::new(resolved.pos.unwrap_or([0.0; 3]));
    let limits = read_limits(node, &resolved, settings.auto_limits, to_radians, body_name)?;

    let name = node.attribute("name").unwrap_or("joint").to_owned();

    Ok(Some(JointRecord {
        name,
        kind,
        axis,
        anchor,
        limits,
        zero_offset: to_radians(resolved.reference.unwrap_or(0.0)),
        armature: resolved.armature.unwrap_or(0.0),
        damping: resolved.damping.unwrap_or(0.0),
        friction_loss: resolved.friction_loss.unwrap_or(0.0),
        spring_reference: to_radians(resolved.spring_reference.unwrap_or(0.0)),
        spring_stiffness: resolved.stiffness.unwrap_or(0.0),
    }))
}

/// A joint's travel limits, from `range` and `limited`.
fn read_limits(
    node: Node,
    resolved: &JointDefaults,
    auto_limits: bool,
    to_radians: impl Fn(f64) -> f64,
    body_name: &str,
) -> Result<Option<(f64, f64)>, MjcfError> {
    let needs_range = || MjcfError::LimitsNeedRange {
        body: body_name.to_owned(),
    };

    match resolved.limited.as_deref().unwrap_or("auto") {
        "false" => Ok(None),
        "true" => {
            let [lower, upper] = resolved.range.ok_or_else(needs_range)?;
            Ok(Some((to_radians(lower), to_radians(upper))))
        }
        "auto" => match resolved.range {
            Some([lower, upper]) if auto_limits => Ok(Some((to_radians(lower), to_radians(upper)))),
            Some(_) => Err(needs_range()),
            None => Ok(None),
        },
        other => Err(bad_attribute(node, "limited", other)),
    }
}

/// A joint's settings, with its own attributes winning over its named class, that over the body's
/// inherited `childclass`, that over the unnamed block.
fn effective(
    node: Node,
    table: &DefaultTable,
    class_chain: Option<&str>,
) -> Result<JointDefaults, MjcfError> {
    let mut settings = table.resolve(None)?.joint.clone();
    if let Some(name) = class_chain {
        settings = settings.overridden_by(&table.resolve(Some(name))?.joint);
    }
    if let Some(name) = node.attribute("class") {
        settings = settings.overridden_by(&table.resolve(Some(name))?.joint);
    }
    Ok(settings.overridden_by(&JointDefaults::read(node)?))
}
