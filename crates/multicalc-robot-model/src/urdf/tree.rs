//! Topology resolution: flat `<link>`/`<joint>` lists to a topologically ordered body list.
//!
//! Index links by name, find the single link that is never a child, then walk depth-first from it.
//! Visit order guarantees parent index < child index, which `KinematicTree::push` requires.

use std::collections::HashMap;

use multicalc::kinematics::JointKind;
use multicalc::spatial::SE3;

use crate::urdf::joint::ParsedJoint;
use crate::urdf::link::ParsedLink;
use crate::{BodyDescription, ModelError};

/// Resolves the tree, depth-first from the root link.
///
/// One body per link, in visit order, so every parent precedes its children.
pub(crate) fn build(
    links: &[ParsedLink],
    joints: &[ParsedJoint],
) -> Result<Vec<BodyDescription>, ModelError> {
    // Duplicate link names: first wins, and joints resolve to it.
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for (index, link) in links.iter().enumerate() {
        by_name.entry(link.name.as_str()).or_insert(index);
    }

    let find = |joint: &ParsedJoint, link: &str| -> Result<usize, ModelError> {
        by_name
            .get(link)
            .copied()
            .ok_or_else(|| ModelError::UnknownLink {
                joint: joint.name.clone(),
                link: link.to_owned(),
            })
    };

    // Per link: its incoming joint and its outgoing joints. Per joint: the child link it drives.
    let mut incoming: Vec<Option<usize>> = vec![None; links.len()];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); links.len()];
    let mut driven: Vec<usize> = Vec::with_capacity(joints.len());
    for (joint_index, joint) in joints.iter().enumerate() {
        let parent = find(joint, &joint.parent_link)?;
        let child = find(joint, &joint.child_link)?;
        driven.push(child);

        if let Some(claimed) = incoming[child] {
            let mut names = vec![joints[claimed].name.clone(), joint.name.clone()];
            names.sort();
            return Err(ModelError::LinkHasTwoParents {
                link: links[child].name.clone(),
                joints: names,
            });
        }
        incoming[child] = Some(joint_index);
        children[parent].push(joint_index);
    }

    let roots: Vec<usize> = (0..links.len())
        .filter(|&index| incoming[index].is_none())
        .collect();
    let root = match roots.as_slice() {
        [] => return Err(ModelError::MissingRootLink),
        [only] => *only,
        several => {
            let mut names: Vec<String> = several
                .iter()
                .map(|&index| links[index].name.clone())
                .collect();
            names.sort();
            return Err(ModelError::MultipleRootLinks { names });
        }
    };

    let mut bodies = Vec::with_capacity(links.len());
    let mut visited = vec![false; links.len()];
    walk(
        root,
        None,
        None,
        links,
        joints,
        &children,
        &driven,
        &mut visited,
        &mut bodies,
    );

    // The walk reaches every link unless some form a cycle disconnected from the root.
    if let Some(stranded) = visited.iter().position(|seen| !seen) {
        return Err(ModelError::CyclicLinkage {
            link: links[stranded].name.clone(),
        });
    }

    // A URDF joint belongs to its child link and the root link is never a child, so a floating
    // joint can never land on body 0, where `KinematicTree` requires it. Fixed versus floating base
    // is a load-time choice by the caller (cf. Pinocchio's root-joint argument, RBDL's
    // `floating_base`), so a floating joint stated in the file is rejected.
    for body in &bodies {
        if body
            .joint
            .as_ref()
            .is_some_and(|joint| joint.kind == JointKind::Floating)
        {
            return Err(ModelError::FreeJointNotAtRoot {
                body: body.name.clone(),
            });
        }
    }

    Ok(bodies)
}

/// Emits one body, then recurses into its children.
///
/// `incoming` is the link's parent joint, supplying its pose and joint. The root has none and sits
/// at identity.
#[expect(
    clippy::too_many_arguments,
    reason = "the recursion threads both source lists, the resolved topology, and the output"
)]
fn walk(
    link_index: usize,
    parent_body: Option<usize>,
    incoming: Option<usize>,
    links: &[ParsedLink],
    joints: &[ParsedJoint],
    children: &[Vec<usize>],
    driven: &[usize],
    visited: &mut [bool],
    bodies: &mut Vec<BodyDescription>,
) {
    visited[link_index] = true;
    let link = &links[link_index];
    let joint = incoming.map(|index| &joints[index]);

    let body_index = bodies.len();
    bodies.push(BodyDescription {
        name: link.name.clone(),
        parent: parent_body,
        pose: joint.map_or_else(SE3::identity, |joint| joint.origin),
        inertia: link.inertia,
        joint: joint.and_then(|joint| joint.description.clone()),
        visual_geometry: link.visual_geometry.clone(),
    });

    for &joint_index in &children[link_index] {
        walk(
            driven[joint_index],
            Some(body_index),
            Some(joint_index),
            links,
            joints,
            children,
            driven,
            visited,
            bodies,
        );
    }
}
