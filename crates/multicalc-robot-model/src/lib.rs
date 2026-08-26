//! Robot model file readers for multicalc.
//!
//! MJCF and URDF both parse into one [`RobotModel`]: a topologically ordered body list carrying
//! name, parent index, parent-relative transform, spatial inertia where stated, and joint.
//!
//! Bodies also carry the shapes the file draws them with, for viewing only.
//!
//! Converts on demand to a [`KinematicTree`](multicalc::kinematics::KinematicTree), whole or as
//! the root-to-tip chain for a named body.
//!
//! Unsupported constructs are rejected by name rather than dropped, so a model never loads with
//! wrong mass properties. Unconsumed top-level elements are listed in [`RobotModel::ignored`].

mod codegen;
mod error;
mod geometry;
mod xml;

pub mod mjcf;
pub mod urdf;
pub mod viewer;

use std::path::{Path, PathBuf};

use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{Joint, JointKind, JointParent, KinematicTree};
use multicalc::linear_algebra::Vector3D;
use multicalc::spatial::{SE3, SpatialInertia};

pub use codegen::{GeneratedScalar, RustSourceOptions};
pub use error::ModelError;
pub use geometry::{GeometryShape, VisualGeometry};

/// The format a model was parsed from.
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

/// A parsed robot: body tree and per-body mass properties.
#[derive(Debug, Clone, PartialEq)]
pub struct RobotModel {
    name: String,
    format: ModelFormat,
    bodies: Vec<BodyDescription>,
    floating_base: bool,
    ignored: Vec<String>,
    base_directory: Option<PathBuf>,
}

impl RobotModel {
    /// Model name (`model` if unspecified).
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The format this model was parsed from.
    #[inline]
    #[must_use]
    pub fn format(&self) -> ModelFormat {
        self.format
    }

    /// All bodies, in topological (depth-first document) order.
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

    /// Bodies carrying a joint. Welded bodies are excluded.
    #[inline]
    #[must_use]
    pub fn movable_joint_count(&self) -> usize {
        self.bodies
            .iter()
            .filter(|body| body.joint.is_some())
            .count()
    }

    /// Whether the root body has a 6-DoF joint rather than being welded to the world.
    #[inline]
    #[must_use]
    pub fn has_floating_base(&self) -> bool {
        self.floating_base
    }

    /// Top-level elements the reader consumed nothing from, sorted and deduplicated.
    ///
    /// Only elements that cannot affect mass properties reach here; anything that could is
    /// rejected by name.
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

    /// Directory the model file sat in. `None` for a model parsed from text, which has nothing to
    /// resolve a mesh path against.
    #[inline]
    #[must_use]
    pub fn base_directory(&self) -> Option<&Path> {
        self.base_directory.as_deref()
    }

    /// Absolute path to a mesh file, or `None` where it cannot be resolved.
    ///
    /// `package://<name>/<rest>` looks `<name>` up in `packages`; `file://` and absolute paths are
    /// taken as they are; anything else resolves against [`RobotModel::base_directory`].
    #[must_use]
    pub fn mesh_path(&self, file: &str, packages: &[(String, PathBuf)]) -> Option<PathBuf> {
        geometry::resolve_mesh_path(file, self.base_directory(), packages)
    }

    /// The whole model as a kinematic tree, one slot per body.
    ///
    /// A jointless body takes a slot as a weld, so slot index equals body index. Joint dynamics
    /// carry through as stated.
    ///
    /// Errors: [`TreeCapacityExceeded`](ModelError::TreeCapacityExceeded) where the model has more
    /// bodies than `MAX_JOINTS`, [`Kinematics`](ModelError::Kinematics) where a joint's own
    /// numbers do not describe a usable joint, and
    /// [`MimicJointInTree`](ModelError::MimicJointInTree) where any joint in the model follows
    /// another.
    pub fn kinematic_tree<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
        &self,
    ) -> Result<KinematicTree<MAX_JOINTS, MAX_CONFIG, f64>, ModelError> {
        let slots: Vec<usize> = (0..self.bodies.len()).collect();
        self.build_tree(&slots)
    }

    /// The root-to-tip chain for a named body, and nothing else.
    ///
    /// Slot `k` is the `k`-th body along the chain, so an arm can be taken out of a model that
    /// also carries a gripper.
    ///
    /// Errors: as [`kinematic_tree`](RobotModel::kinematic_tree), plus
    /// [`UnknownBody`](ModelError::UnknownBody) where the model has no body by that name, and
    /// [`MimicJointInTree`](ModelError::MimicJointInTree) where a joint on the chain follows
    /// another.
    pub fn kinematic_tree_to<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
        &self,
        tip: &str,
    ) -> Result<KinematicTree<MAX_JOINTS, MAX_CONFIG, f64>, ModelError> {
        let slots = self.path_to(tip)?;
        self.build_tree(&slots)
    }

    /// The whole model as an articulated body, one slot per body, with `gravity` in world axes.
    ///
    /// [`kinematic_tree`](RobotModel::kinematic_tree)'s slot layout, each slot carrying that
    /// body's stated mass properties. A body the file gives no `<inertial>` carries none.
    ///
    /// Errors: as [`kinematic_tree`](RobotModel::kinematic_tree), plus
    /// [`Dynamics`](ModelError::Dynamics) where the mass properties do not describe a usable model.
    ///
    /// ```
    /// use multicalc::linear_algebra::Vector;
    ///
    /// let xml = r#"<mujoco>
    ///                <worldbody>
    ///                  <body>
    ///                    <joint type="hinge" axis="0 1 0"/>
    ///                    <inertial pos="0.5 0 0" mass="2" diaginertia="0.01 0.01 0.01"/>
    ///                  </body>
    ///                </worldbody>
    ///              </mujoco>"#;
    ///
    /// let earth_gravity = Vector::new([0.0, 0.0, -9.81]);
    /// let model = multicalc_robot_model::mjcf::load_str(xml)?;
    /// let body = model.articulated_body::<1, 1>(earth_gravity)?;
    /// assert_eq!(body.len(), 1);
    /// # Ok::<(), multicalc_robot_model::ModelError>(())
    /// ```
    pub fn articulated_body<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
        &self,
        gravity: Vector3D<f64>,
    ) -> Result<ArticulatedBody<MAX_JOINTS, MAX_CONFIG, f64>, ModelError> {
        let slots: Vec<usize> = (0..self.bodies.len()).collect();
        self.build_articulated_body(&slots, gravity)
    }

    /// The root-to-tip chain for a named body as an articulated body, and nothing else.
    ///
    /// Errors: as [`articulated_body`](RobotModel::articulated_body), plus
    /// [`UnknownBody`](ModelError::UnknownBody) where the model has no body by that name.
    pub fn articulated_body_to<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
        &self,
        tip: &str,
        gravity: Vector3D<f64>,
    ) -> Result<ArticulatedBody<MAX_JOINTS, MAX_CONFIG, f64>, ModelError> {
        let slots = self.path_to(tip)?;
        self.build_articulated_body(&slots, gravity)
    }

    /// Builds the tree over these slots, then pairs it with each slot's stated mass properties.
    fn build_articulated_body<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
        &self,
        slots: &[usize],
        gravity: Vector3D<f64>,
    ) -> Result<ArticulatedBody<MAX_JOINTS, MAX_CONFIG, f64>, ModelError> {
        let tree = self.build_tree::<MAX_JOINTS, MAX_CONFIG>(slots)?;
        let inertias: Vec<Option<SpatialInertia<f64>>> = slots
            .iter()
            .map(|&index| self.bodies[index].inertia)
            .collect();
        ArticulatedBody::new(tree, &inertias, gravity).map_err(ModelError::Dynamics)
    }

    /// Body indices from the root to the named body, root first.
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
    /// A slot's parent is `World` where the body's parent is absent from `slots`, otherwise the
    /// slot the parent landed in: the body's own index for
    /// [`kinematic_tree`](RobotModel::kinematic_tree), and `k - 1` for
    /// [`kinematic_tree_to`](RobotModel::kinematic_tree_to)'s slot `k`.
    fn build_tree<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
        &self,
        slots: &[usize],
    ) -> Result<KinematicTree<MAX_JOINTS, MAX_CONFIG, f64>, ModelError> {
        self.reject_mimic_joints(slots)?;
        if slots.len() > MAX_JOINTS {
            return Err(ModelError::TreeCapacityExceeded {
                needed: slots.len(),
                capacity: MAX_JOINTS,
            });
        }

        let (joints, parents) = self.joints_and_parents(slots);
        KinematicTree::try_from_joints(&joints, &parents).map_err(ModelError::Kinematics)
    }

    /// Rejects the first mimic joint among these slots.
    ///
    /// Checking only the requested slots lets a chain that excludes the mimic joint still build.
    pub(crate) fn reject_mimic_joints(&self, slots: &[usize]) -> Result<(), ModelError> {
        for &index in slots {
            let body = &self.bodies[index];
            if let Some(description) = &body.joint {
                if let Some(mimic) = &description.mimic {
                    return Err(ModelError::MimicJointInTree {
                        joint: description.name.clone(),
                        follows: mimic.joint.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// The `Joint` and `JointParent` per slot, in slot order.
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

/// The `Joint` a body describes: geometry first, then joint dynamics.
fn build_joint(body: &BodyDescription) -> Joint<f64> {
    let Some(description) = &body.joint else {
        return Joint::fixed(body.pose);
    };

    let joint = match description.kind {
        JointKind::Revolute => {
            Joint::revolute(description.axis, body.pose).with_anchor(description.anchor)
        }
        JointKind::Continuous => {
            Joint::continuous(description.axis, body.pose).with_anchor(description.anchor)
        }
        JointKind::Prismatic => Joint::prismatic(description.axis, body.pose),
        // A free joint's qpos is the world pose directly; pos/quat only seed qpos0's default and
        // are never composed onto it. Passing body.pose would put an identity reading off the
        // origin, where MuJoCo's own solve does not.
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

/// One body: parent-relative transform, spatial inertia, and its joint.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyDescription {
    name: String,
    parent: Option<usize>,
    pose: SE3<f64>,
    inertia: Option<SpatialInertia<f64>>,
    joint: Option<JointDescription>,
    visual_geometry: Vec<VisualGeometry>,
}

impl BodyDescription {
    /// Body name (`body` if unspecified).
    #[inline]
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Parent body index, `None` at the root.
    #[inline]
    #[must_use]
    pub fn parent(&self) -> Option<usize> {
        self.parent
    }

    /// Transform from the parent body frame.
    #[inline]
    #[must_use]
    pub fn pose(&self) -> SE3<f64> {
        self.pose
    }

    /// Mass, COM and rotational inertia, or `None` where the file states none.
    ///
    /// A massless body still takes a tree slot and its transform is read normally. URDF uses these
    /// for tool and sensor frames.
    #[inline]
    #[must_use]
    pub fn inertia(&self) -> Option<SpatialInertia<f64>> {
        self.inertia
    }

    /// Joint to the parent body, `None` if welded.
    #[inline]
    #[must_use]
    pub fn joint(&self) -> Option<&JointDescription> {
        self.joint.as_ref()
    }

    /// Shapes the file draws this body with, in document order. Empty where it states none.
    #[inline]
    #[must_use]
    pub fn visual_geometry(&self) -> &[VisualGeometry] {
        &self.visual_geometry
    }
}

/// A joint coupled to another by `child = multiplier * driver + offset`.
///
/// A `KinematicTree` carries no constraints, so a model with one parses but cannot convert whole.
#[derive(Debug, Clone, PartialEq)]
pub struct MimicDescription {
    joint: String,
    multiplier: f64,
    offset: f64,
}

impl MimicDescription {
    /// The driving joint's name.
    #[inline]
    #[must_use]
    pub fn joint(&self) -> &str {
        &self.joint
    }

    /// Ratio to the driving joint.
    #[inline]
    #[must_use]
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Value at zero driving-joint position.
    #[inline]
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.offset
    }
}

/// One joint's kinematic and dynamic parameters, as stated.
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

    /// Revolute, prismatic, continuous or floating.
    #[inline]
    #[must_use]
    pub fn kind(&self) -> JointKind {
        self.kind
    }

    /// Unit joint axis, in the child body frame.
    #[inline]
    pub fn axis(&self) -> Vector3D<f64> {
        self.axis
    }

    /// Revolute centre of rotation, as an offset in the body frame. Unused for prismatic joints.
    #[inline]
    pub fn anchor(&self) -> Vector3D<f64> {
        self.anchor
    }

    /// Travel limits `(lower, upper)`, `None` if unbounded.
    #[inline]
    #[must_use]
    pub fn limits(&self) -> Option<(f64, f64)> {
        self.limits
    }

    /// Joint value at zero configuration (MJCF `ref`). Always 0 for URDF.
    #[inline]
    #[must_use]
    pub fn zero_offset(&self) -> f64 {
        self.zero_offset
    }

    /// Reflected rotor inertia (MJCF `armature`). Always 0 for URDF.
    #[inline]
    #[must_use]
    pub fn armature(&self) -> f64 {
        self.armature
    }

    /// Viscous damping coefficient.
    #[inline]
    #[must_use]
    pub fn damping(&self) -> f64 {
        self.damping
    }

    /// Coulomb friction (breakaway force or torque).
    #[inline]
    #[must_use]
    pub fn friction_loss(&self) -> f64 {
        self.friction_loss
    }

    /// Spring equilibrium position (MJCF `springref`). Always 0 for URDF.
    #[inline]
    #[must_use]
    pub fn spring_reference(&self) -> f64 {
        self.spring_reference
    }

    /// Spring stiffness. Always 0 for URDF.
    #[inline]
    #[must_use]
    pub fn spring_stiffness(&self) -> f64 {
        self.spring_stiffness
    }

    /// The joint driving this one, `None` if independent.
    #[inline]
    #[must_use]
    pub fn mimic(&self) -> Option<&MimicDescription> {
        self.mimic.as_ref()
    }

    /// A 6-DoF joint. No other field applies, so each is left at its inert default.
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

/// Reads a model file, dispatching on extension: `.urdf` reads URDF, anything else MJCF.
///
/// Errors: whatever the chosen reader rejects.
pub fn load_path(path: &Path) -> Result<RobotModel, ModelError> {
    let is_urdf = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("urdf"));
    if is_urdf {
        urdf::load_path(path)
    } else {
        mjcf::load_path(path)
    }
}

/// Parses a model, dispatching on root element: `<robot>` reads URDF, `<mujoco>` MJCF.
///
/// Errors: as [`load_path`], plus
/// [`UnexpectedRootElement`](ModelError::UnexpectedRootElement) for any other root.
pub fn load_str(xml: &str) -> Result<RobotModel, ModelError> {
    let document =
        roxmltree::Document::parse(xml).map_err(|err| ModelError::Xml(err.to_string()))?;
    match document.root_element().tag_name().name() {
        "robot" => urdf::load_str(xml),
        "mujoco" => mjcf::load_str(xml),
        found => Err(ModelError::UnexpectedRootElement {
            found: found.to_owned(),
        }),
    }
}
