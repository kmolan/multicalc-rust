//! Loads robot model files into multicalc's robot types.
//!
//! Reads MuJoCo MJCF and URDF into one [`RobotModel`]: a flat, depth-first list of bodies, each
//! with a name, a parent index, a pose, mass properties where the file states them, and a joint
//! where the body has one.
//!
//! Converts on request to a [`KinematicTree`](multicalc::kinematics::KinematicTree), either the
//! whole model or the chain from the root to a named tip body.
//!
//! Constructs a reader does not handle are rejected by name rather than silently dropped, so a
//! model never loads with incorrect mass properties. Sections a reader does not consume are
//! skipped and listed in [`RobotModel::ignored`].

mod codegen;
mod error;
// Only a reader reads XML, so with neither compiled in the crate is the model types and the
// codegen alone.
#[cfg(any(feature = "mjcf", feature = "urdf"))]
mod xml;

#[cfg(feature = "mjcf")]
pub mod mjcf;

use multicalc::kinematics::{Joint, JointKind, JointParent, KinematicTree};
use multicalc::linear_algebra::Vector3D;
use multicalc::spatial::{SE3, SpatialInertia};

pub use codegen::{GeneratedScalar, RustSourceOptions};
pub use error::ModelError;

/// Which file format a model was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelFormat {
    /// MuJoCo MJCF.
    Mjcf,
    /// URDF.
    Urdf,
}

impl core::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ModelFormat::Mjcf => f.write_str("MJCF"),
            ModelFormat::Urdf => f.write_str("URDF"),
        }
    }
}

/// A robot as a model file describes it: its body tree and per-body mass properties.
#[derive(Debug, Clone, PartialEq)]
pub struct RobotModel {
    name: String,
    format: ModelFormat,
    bodies: Vec<BodyDescription>,
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

    /// The format the model was read from.
    #[inline]
    #[must_use]
    pub fn format(&self) -> ModelFormat {
        self.format
    }

    /// All bodies, depth-first in document order.
    #[inline]
    #[must_use]
    pub fn bodies(&self) -> &[BodyDescription] {
        &self.bodies
    }

    /// Body at `index`, or `None` if out of range.
    #[inline]
    #[must_use]
    pub fn body(&self, index: usize) -> Option<&BodyDescription> {
        self.bodies.get(index)
    }

    /// First body named `name`.
    #[inline]
    #[must_use]
    pub fn body_named(&self, name: &str) -> Option<&BodyDescription> {
        self.bodies.iter().find(|body| body.name == name)
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
            .filter(|body| body.joint.is_some())
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
    /// let model = multicalc_robot_model::mjcf::load_str(xml)?;
    /// assert_eq!(model.body(0).unwrap().inertia().unwrap().mass(), 1.0);
    /// assert_eq!(model.ignored(), ["actuator".to_owned()]);
    /// # Ok::<(), multicalc_robot_model::ModelError>(())
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
    /// Errors: [`TreeCapacityExceeded`](ModelError::TreeCapacityExceeded) where the model has more
    /// bodies than `MAX_JOINTS`, and [`Kinematics`](ModelError::Kinematics) where a joint's own
    /// numbers do not describe a usable joint.
    pub fn kinematic_tree<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
        &self,
    ) -> Result<KinematicTree<MAX_JOINTS, MAX_CONFIG, f64>, ModelError> {
        let slots: Vec<usize> = (0..self.bodies.len()).collect();
        self.build_tree(&slots)
    }

    /// The chain running from the world down to the body you name, and nothing else.
    ///
    /// Slot `k` is the `k`-th body along that chain, so a model with a gripper on the end can be
    /// read as the arm alone.
    ///
    /// Errors: as [`kinematic_tree`](RobotModel::kinematic_tree), plus
    /// [`UnknownBody`](ModelError::UnknownBody) where the model has no body by that name.
    pub fn kinematic_tree_to<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
        &self,
        tip: &str,
    ) -> Result<KinematicTree<MAX_JOINTS, MAX_CONFIG, f64>, ModelError> {
        let slots = self.path_to(tip)?;
        self.build_tree(&slots)
    }

    /// The bodies from the world down to the one you name, by index, the world end first.
    ///
    /// Errors: [`UnknownBody`](ModelError::UnknownBody) where the model has no body by that name.
    pub fn path_to(&self, tip: &str) -> Result<Vec<usize>, ModelError> {
        let mut index = self
            .bodies
            .iter()
            .position(|body| body.name == tip)
            .ok_or_else(|| ModelError::UnknownBody {
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
    fn build_tree<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
        &self,
        slots: &[usize],
    ) -> Result<KinematicTree<MAX_JOINTS, MAX_CONFIG, f64>, ModelError> {
        if slots.len() > MAX_JOINTS {
            return Err(ModelError::TreeCapacityExceeded {
                needed: slots.len(),
                capacity: MAX_JOINTS,
            });
        }

        let (joints, parents) = self.joints_and_parents(slots);
        KinematicTree::try_from_joints(&joints, &parents).map_err(ModelError::Kinematics)
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

/// The `Joint` a body's own description asks for: its geometry, then the dynamics data carried
/// alongside it.
fn build_joint(body: &BodyDescription) -> Joint<f64> {
    let Some(description) = &body.joint else {
        return Joint::fixed(body.pose);
    };

    let joint = match description.kind {
        JointKind::Revolute => Joint::revolute(description.axis, body.pose).with_anchor(description.anchor),
        JointKind::Continuous => {
            Joint::continuous(description.axis, body.pose).with_anchor(description.anchor)
        }
        JointKind::Prismatic => Joint::prismatic(description.axis, body.pose),
        // MuJoCo does not compose a free-jointed body's own pos/quat onto its qpos at runtime —
        // qpos is the world pose directly, and pos/quat only ever seed qpos0's default. Passing
        // body.pose here would place an identity reading away from the origin, which does not
        // match MuJoCo's own solve of the same file.
        JointKind::Floating => return Joint::floating(SE3::identity()),
        JointKind::Fixed => {
            unreachable!(
                "a JointDescription's kind is only ever Revolute, Prismatic, Continuous, or Floating"
            )
        }
    };
    let joint = joint
        .with_zero_offset(description.zero_offset)
        .with_armature(description.armature)
        .with_damping(description.damping)
        .with_friction_loss(description.friction_loss);
    match description.limits {
        Some((lower, upper)) => joint.with_limits(lower, upper),
        None => joint,
    }
}

/// One body: pose, spatial inertia, and its joint (if any).
#[derive(Debug, Clone, PartialEq)]
pub struct BodyDescription {
    name: String,
    parent: Option<usize>,
    pose: SE3<f64>,
    inertia: Option<SpatialInertia<f64>>,
    joint: Option<JointDescription>,
}

impl BodyDescription {
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

    /// Mass, center of mass, and rotational inertia, or `None` where the file stated none.
    ///
    /// A body with no mass properties is still a real body: it takes a slot in the tree and its
    /// pose is read like any other. URDF uses such bodies for tool and sensor frames.
    #[inline]
    #[must_use]
    pub fn inertia(&self) -> Option<SpatialInertia<f64>> {
        self.inertia
    }

    /// Joint connecting the body to its parent, or `None` if welded.
    #[inline]
    #[must_use]
    pub fn joint(&self) -> Option<&JointDescription> {
        self.joint.as_ref()
    }
}

/// One joint following another, at a fixed ratio and offset.
///
/// A tree of joints that each move on their own cannot describe this, so a model carrying one is
/// read in full but cannot be turned into a whole-model tree.
#[derive(Debug, Clone, PartialEq)]
pub struct MimicDescription {
    joint: String,
    multiplier: f64,
    offset: f64,
}

impl MimicDescription {
    /// The joint this one follows.
    #[inline]
    #[must_use]
    pub fn joint(&self) -> &str {
        &self.joint
    }

    /// How far this joint moves for each unit the joint it follows moves.
    #[inline]
    #[must_use]
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Where this joint sits when the joint it follows reads zero.
    #[inline]
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }
}

/// One joint: kinematic and dynamic parameters as read from the file.
#[derive(Debug, Clone, PartialEq)]
pub struct JointDescription {
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
    mimic: Option<MimicDescription>,
}

impl JointDescription {
    /// Joint name (`joint` if unspecified).
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Revolute, prismatic, continuous, or floating.
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

    /// The joint this one is driven by, or `None` where it moves on its own.
    #[inline]
    #[must_use]
    pub fn mimic(&self) -> Option<&MimicDescription> {
        self.mimic.as_ref()
    }

    /// A floating (6-DOF) joint: a file that frees a body to move in all six directions states
    /// none of a `JointDescription`'s other fields, so every one of them is left at its inert
    /// default.
    #[cfg(any(feature = "mjcf", feature = "urdf"))]
    pub(crate) fn floating(name: String) -> Self {
        JointDescription {
            name,
            kind: JointKind::Floating,
            axis: Vector3D::new([1.0, 0.0, 0.0]),
            anchor: Vector3D::zeros(),
            limits: None,
            zero_offset: 0.0,
            armature: 0.0,
            damping: 0.0,
            friction_loss: 0.0,
            spring_reference: 0.0,
            spring_stiffness: 0.0,
            mimic: None,
        }
    }
}

