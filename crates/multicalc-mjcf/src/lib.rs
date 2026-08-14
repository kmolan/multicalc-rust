//! Loads MuJoCo MJCF model files into multicalc's robot types.
//!
//! Parses the `<worldbody>` tree into a flat, depth-first list of bodies: name, parent index, pose,
//! spatial inertia, and joint (if any). Inertia comes from `<inertial>` when stated, otherwise it is
//! computed from the body's `<geom>` primitives.
//!
//! Converts on request to a [`KinematicTree`](multicalc::kinematics::KinematicTree), either the
//! whole model or the chain from the root to a named tip body.
//!
//! Unsupported constructs — ball joints, mesh inertia, non-quaternion orientations — are rejected by
//! name rather than silently dropped, so a model never loads with incorrect mass properties. Sections
//! this loader does not consume (tendons, actuators, sensors, ...) are skipped and listed in
//! [`RobotModel::ignored`].

mod body;
mod codegen;
mod compiler;
mod defaults;
mod document;
mod error;
mod geometry;
mod joint;

use std::path::Path;

use multicalc::kinematics::{Joint, JointKind, JointParent, KinematicTree};
use multicalc::linear_algebra::Vector3D;
use multicalc::spatial::{SE3, SpatialInertia};

pub use codegen::{GeneratedScalar, RustSourceOptions};
pub use error::MjcfError;

/// A parsed MJCF model: its body tree and per-body mass properties.
#[derive(Debug, Clone, PartialEq)]
pub struct RobotModel {
    name: String,
    bodies: Vec<BodyRecord>,
    floating_base: bool,
    ignored: Vec<String>,
}

impl RobotModel {
    /// Model name (`model` if unspecified).
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// All bodies, depth-first in document order.
    #[inline]
    #[must_use]
    pub fn bodies(&self) -> &[BodyRecord] {
        &self.bodies
    }

    /// Body at `index`, or `None` if out of range.
    #[inline]
    #[must_use]
    pub fn body(&self, index: usize) -> Option<&BodyRecord> {
        self.bodies.get(index)
    }

    /// First body named `name`.
    #[inline]
    #[must_use]
    pub fn body_named(&self, name: &str) -> Option<&BodyRecord> {
        self.bodies.iter().find(|record| record.name == name)
    }

    /// Body count.
    #[inline]
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Number of bodies with a joint (degrees of freedom).
    #[inline]
    #[must_use]
    pub fn movable_joint_count(&self) -> usize {
        self.bodies
            .iter()
            .filter(|record| record.joint.is_some())
            .count()
    }

    /// Whether the root body has a free joint (floating base) rather than being welded to the
    /// world.
    #[inline]
    #[must_use]
    pub fn has_floating_base(&self) -> bool {
        self.floating_base
    }

    /// Top-level sections not consumed by this loader (sorted, deduplicated). Empty if every
    /// section was read.
    ///
    /// Only sections that cannot affect mass properties can land here — anything that could is
    /// rejected outright, not ignored.
    ///
    /// ```
    /// let xml = r#"<mujoco>
    ///                <worldbody>
    ///                  <body><freejoint/><inertial mass="1" diaginertia="1 1 1"/></body>
    ///                </worldbody>
    ///                <actuator><motor name="thrust" gear="0 0 1 0 0 0"/></actuator>
    ///              </mujoco>"#;
    ///
    /// let model = multicalc_mjcf::load_str(xml)?;
    /// assert_eq!(model.body(0).unwrap().inertia().mass(), 1.0);
    /// assert_eq!(model.ignored(), ["actuator".to_owned()]);
    /// # Ok::<(), multicalc_mjcf::MjcfError>(())
    /// ```
    #[inline]
    #[must_use]
    pub fn ignored(&self) -> &[String] {
        &self.ignored
    }

    /// The whole model as a jointed tree, one slot per body.
    ///
    /// A body with no joint of its own takes a slot as a weld, so a slot index and a body index
    /// are the same number. Every joint carries what the file said about it, the resistance and
    /// friction figures included.
    ///
    /// Errors: [`FloatingBaseUnsupported`](MjcfError::FloatingBaseUnsupported) where the model
    /// floats free of the world, [`TreeCapacityExceeded`](MjcfError::TreeCapacityExceeded) where
    /// the model has more bodies than `MAX_JOINTS`, and [`Kinematics`](MjcfError::Kinematics)
    /// where a joint's own numbers do not describe a usable joint.
    pub fn kinematic_tree<const MAX_JOINTS: usize>(
        &self,
    ) -> Result<KinematicTree<MAX_JOINTS, f64>, MjcfError> {
        let slots: Vec<usize> = (0..self.bodies.len()).collect();
        self.build_tree(&slots)
    }

    /// The chain running from the world down to the body you name, and nothing else.
    ///
    /// Slot `k` is the `k`-th body along that chain, so a model with a gripper on the end can be
    /// read as the arm alone.
    ///
    /// Errors: as [`kinematic_tree`](RobotModel::kinematic_tree), plus
    /// [`UnknownBody`](MjcfError::UnknownBody) where the model has no body by that name.
    pub fn kinematic_tree_to<const MAX_JOINTS: usize>(
        &self,
        tip: &str,
    ) -> Result<KinematicTree<MAX_JOINTS, f64>, MjcfError> {
        let slots = self.path_to(tip)?;
        self.build_tree(&slots)
    }

    /// The bodies from the world down to the one you name, by index, the world end first.
    ///
    /// Errors: [`UnknownBody`](MjcfError::UnknownBody) where the model has no body by that name.
    pub fn path_to(&self, tip: &str) -> Result<Vec<usize>, MjcfError> {
        let mut index = self
            .bodies
            .iter()
            .position(|body| body.name == tip)
            .ok_or_else(|| MjcfError::UnknownBody {
                name: tip.to_owned(),
            })?;

        let mut path = vec![index];
        while let Some(parent) = self.bodies[index].parent {
            path.push(parent);
            index = parent;
        }
        path.reverse();
        Ok(path)
    }

    /// Builds a tree over the given body indices, in slot order.
    ///
    /// A slot's parent is `World` where the body's own parent is absent from `slots`, and
    /// otherwise the slot the parent body landed in — which is the body's own index for
    /// [`kinematic_tree`](RobotModel::kinematic_tree) (every body is present, in order) and
    /// `k - 1` for [`kinematic_tree_to`](RobotModel::kinematic_tree_to)'s slot `k` (`slots` is a
    /// single root-to-tip chain, so a body's parent is always the previous entry).
    fn build_tree<const MAX_JOINTS: usize>(
        &self,
        slots: &[usize],
    ) -> Result<KinematicTree<MAX_JOINTS, f64>, MjcfError> {
        if self.floating_base {
            return Err(MjcfError::FloatingBaseUnsupported {
                body: self.bodies[0].name.clone(),
            });
        }
        if slots.len() > MAX_JOINTS {
            return Err(MjcfError::TreeCapacityExceeded {
                needed: slots.len(),
                capacity: MAX_JOINTS,
            });
        }

        let (joints, parents) = self.joints_and_parents(slots);
        KinematicTree::try_from_joints(&joints, &parents).map_err(MjcfError::Kinematics)
    }

    /// The `Joint` and `JointParent` each slot needs, in slot order. A slot's parent is `World`
    /// where the body's own parent is absent from `slots`, and otherwise the slot the parent body
    /// landed in.
    pub(crate) fn joints_and_parents(
        &self,
        slots: &[usize],
    ) -> (Vec<Joint<f64>>, Vec<JointParent>) {
        let mut joints = Vec::with_capacity(slots.len());
        let mut parents = Vec::with_capacity(slots.len());
        for &index in slots {
            let body = &self.bodies[index];
            joints.push(build_joint(body));
            parents.push(match body.parent {
                None => JointParent::World,
                Some(parent_index) => {
                    let position = slots
                        .iter()
                        .position(|&slot| slot == parent_index)
                        .unwrap_or_else(|| {
                            unreachable!("a body's parent always sits earlier in its own chain")
                        });
                    JointParent::Joint(position)
                }
            });
        }
        (joints, parents)
    }
}

/// The `Joint` one body's record describes: its geometry, then the dynamics data carried
/// alongside it.
fn build_joint(body: &BodyRecord) -> Joint<f64> {
    let Some(record) = &body.joint else {
        return Joint::fixed(body.pose);
    };

    let joint = match record.kind {
        JointKind::Revolute => Joint::revolute(record.axis, body.pose).with_anchor(record.anchor),
        JointKind::Prismatic => Joint::prismatic(record.axis, body.pose),
        JointKind::Fixed => {
            unreachable!("a JointRecord's kind is only ever Revolute or Prismatic")
        }
    };
    let joint = joint
        .with_zero_offset(record.zero_offset)
        .with_armature(record.armature)
        .with_damping(record.damping)
        .with_friction_loss(record.friction_loss);
    match record.limits {
        Some((lower, upper)) => joint.with_limits(lower, upper),
        None => joint,
    }
}

/// One body: pose, spatial inertia, and its joint (if any).
#[derive(Debug, Clone, PartialEq)]
pub struct BodyRecord {
    name: String,
    parent: Option<usize>,
    pose: SE3<f64>,
    inertia: SpatialInertia<f64>,
    joint: Option<JointRecord>,
}

impl BodyRecord {
    /// Body name (`body` if unspecified).
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Parent body index, or `None` at the root.
    #[inline]
    #[must_use]
    pub fn parent(&self) -> Option<usize> {
        self.parent
    }

    /// Pose relative to the parent body frame.
    #[inline]
    #[must_use]
    pub fn pose(&self) -> SE3<f64> {
        self.pose
    }

    /// Mass, center of mass, and rotational inertia.
    #[inline]
    #[must_use]
    pub fn inertia(&self) -> SpatialInertia<f64> {
        self.inertia
    }

    /// Joint connecting the body to its parent, or `None` if welded.
    #[inline]
    #[must_use]
    pub fn joint(&self) -> Option<&JointRecord> {
        self.joint.as_ref()
    }
}

/// One joint: kinematic and dynamic parameters as read from the file.
#[derive(Debug, Clone, PartialEq)]
pub struct JointRecord {
    name: String,
    kind: JointKind,
    axis: Vector3D<f64>,
    anchor: Vector3D<f64>,
    limits: Option<(f64, f64)>,
    zero_offset: f64,
    armature: f64,
    damping: f64,
    friction_loss: f64,
    spring_reference: f64,
    spring_stiffness: f64,
}

impl JointRecord {
    /// Joint name (`joint` if unspecified).
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Revolute or prismatic.
    #[inline]
    #[must_use]
    pub fn kind(&self) -> JointKind {
        self.kind
    }

    /// Joint axis (unit vector).
    #[inline]
    pub fn axis(&self) -> Vector3D<f64> {
        self.axis
    }

    /// Revolute joint's center of rotation; unused for prismatic joints.
    #[inline]
    pub fn anchor(&self) -> Vector3D<f64> {
        self.anchor
    }

    /// Travel limits `(lower, upper)`, or `None` if unlimited.
    #[inline]
    #[must_use]
    pub fn limits(&self) -> Option<(f64, f64)> {
        self.limits
    }

    /// Reference position (MJCF `ref`); joint reading at zero configuration.
    #[inline]
    #[must_use]
    pub fn zero_offset(&self) -> f64 {
        self.zero_offset
    }

    /// Reflected rotor inertia added to the joint-space inertia (MJCF `armature`).
    #[inline]
    #[must_use]
    pub fn armature(&self) -> f64 {
        self.armature
    }

    /// Velocity-proportional damping coefficient.
    #[inline]
    #[must_use]
    pub fn damping(&self) -> f64 {
        self.damping
    }

    /// Coulomb friction (breakaway force/torque).
    #[inline]
    #[must_use]
    pub fn friction_loss(&self) -> f64 {
        self.friction_loss
    }

    /// Spring equilibrium position (MJCF `springref`).
    #[inline]
    #[must_use]
    pub fn spring_reference(&self) -> f64 {
        self.spring_reference
    }

    /// Spring stiffness coefficient.
    #[inline]
    #[must_use]
    pub fn spring_stiffness(&self) -> f64 {
        self.spring_stiffness
    }
}

/// Loads a model from a file path, resolving any `<include>` elements it pulls in.
pub fn load_path(path: &Path) -> Result<RobotModel, MjcfError> {
    let xml = document::assemble(path)?;
    load_str(&xml)
}

/// Parses a model from an in-memory XML string.
///
/// Text has no directory to resolve an `<include>` against, so a document that pulls in another
/// file is refused here; [`load_path`] resolves those first.
pub fn load_str(xml: &str) -> Result<RobotModel, MjcfError> {
    let document = roxmltree::Document::parse(xml).map_err(|e| MjcfError::Xml(e.to_string()))?;
    if document
        .descendants()
        .any(|node| node.is_element() && node.tag_name().name() == "include")
    {
        return Err(MjcfError::IncludeNeedsFile);
    }
    let parsed = body::read(&document)?;
    let bodies = parsed
        .bodies
        .into_iter()
        .map(|body| BodyRecord {
            name: body.name,
            parent: body.parent,
            pose: body.pose,
            inertia: body.inertia,
            joint: body.joint,
        })
        .collect();
    Ok(RobotModel {
        name: parsed.name,
        bodies,
        floating_base: parsed.floating_base,
        ignored: parsed.ignored,
    })
}
