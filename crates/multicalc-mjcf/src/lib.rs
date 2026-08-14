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
mod compiler;
mod defaults;
mod document;
mod error;
mod geometry;

use std::path::Path;

use multicalc::kinematics::JointKind;
use multicalc::linear_algebra::Vector3D;
use multicalc::spatial::{SE3, SpatialInertia};

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
pub fn load_str(xml: &str) -> Result<RobotModel, MjcfError> {
    let document = roxmltree::Document::parse(xml).map_err(|e| MjcfError::Xml(e.to_string()))?;
    let body = body::read(&document)?;
    let record = BodyRecord {
        name: body.name.clone(),
        parent: None,
        pose: body.pose,
        inertia: body.inertia,
        joint: None,
    };
    Ok(RobotModel {
        name: body.name,
        bodies: vec![record],
        floating_base: body.has_free_joint,
        ignored: body.ignored,
    })
}
