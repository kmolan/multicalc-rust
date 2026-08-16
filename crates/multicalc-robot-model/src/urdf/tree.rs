//! Working out the shape of the robot from a flat list of links and the joints between them.
//!
//! URDF states the links and the joints separately and in no particular order, so the tree has to
//! be found rather than followed: index the links by name, find the one link nothing hangs off,
//! and walk down from it.

use std::collections::HashMap;

use multicalc::kinematics::JointKind;
use multicalc::spatial::SE3;

use crate::urdf::joint::ParsedJoint;
use crate::urdf::link::ParsedLink;
use crate::{BodyDescription, ModelError};

/// Orders the links into a tree, depth-first from the one link with no parent.
///
/// Returns one body per link, in visit order, so a body's parent always sits earlier in the list —
/// which is what a `KinematicTree` needs.
pub(crate) fn build(
    links: &[ParsedLink],
    joints: &[ParsedJoint],
) -> Result<Vec<BodyDescription>, ModelError> {
    // A repeated link name keeps the first, and a joint naming it resolves to that one.
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

    // The joint that hangs each link off its parent, the joints going the other way, and the link
    // each joint drives.
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

    // A walk down from the one root reaches every link unless some of them hang off each other in
    // a ring that the root never leads into.
    if let Some(stranded) = visited.iter().position(|seen| !seen) {
        return Err(ModelError::CyclicLinkage {
            link: links[stranded].name.clone(),
        });
    }

    // A URDF joint belongs to the link it drives, and the link at the top of the model is nothing's
    // child, so a joint freeing the whole robot can never land there. Whether a robot is bolted
    // down or free to move is something a caller settles when loading it, not something the file
    // states, so one written into the file is refused rather than read.
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

/// Pushes one link as a body, then walks down into the links hanging off it.
///
/// `incoming` is the joint the link hangs off, which is where its pose and its joint come from;
/// the root link has none and sits at the origin.
#[expect(
    clippy::too_many_arguments,
    reason = "the walk carries the file's two lists, the shape worked out from them, and what it \
              has built so far"
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
