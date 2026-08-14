//! Reading the tree of bodies a model file describes.
//!
//! Where the file states a body's mass properties they are used as written. Where it does not,
//! they are worked out from the shapes the body is built from.

use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use multicalc::spatial::{SE3, SO3, SpatialInertia};
use roxmltree::{Document, Node};

use crate::JointRecord;
use crate::MjcfError;
use crate::compiler::{CompilerSettings, InertiaFromGeom};
use crate::defaults::{
    DefaultTable, bad_attribute, element, elements, parse_scalar, parse_vector3, parse_vector4,
    parse_vector6, reject_orientation_attributes, unit_quaternion,
};
use crate::geometry::{GeomMass, read_geom};
use crate::joint::read_joint;

/// The top-level sections this loader takes something from. Every other section a file carries is
/// passed over, and named in the record rather than going quietly.
const READ_SECTIONS: [&str; 3] = ["compiler", "default", "worldbody"];

/// Everything one model file says, as a flat list of bodies in the order they appear.
pub(crate) struct ParsedModel {
    pub name: String,
    pub bodies: Vec<ParsedBody>,
    pub floating_base: bool,
    pub ignored: Vec<String>,
}

/// One body, as read from the file: its place in the tree, its pose, its mass, and its joint.
pub(crate) struct ParsedBody {
    pub name: String,
    pub parent: Option<usize>,
    pub pose: SE3<f64>,
    pub inertia: SpatialInertia<f64>,
    pub joint: Option<JointRecord>,
}

/// Reads the tree of bodies a file describes, refusing anything outside this loader's subset by
/// name.
pub(crate) fn read(document: &Document) -> Result<ParsedModel, MjcfError> {
    let root = document.root_element();
    let name = root.attribute("model").unwrap_or("model").to_owned();
    let settings = CompilerSettings::read(root)?;
    let table = DefaultTable::build(root)?;
    let worldbody = element(root, "worldbody").ok_or(MjcfError::MissingWorldbody)?;

    let mut bodies = Vec::new();
    let mut floating_base = false;
    for node in elements(worldbody, "body") {
        walk_body(
            node,
            None,
            None,
            &table,
            &settings,
            &mut bodies,
            &mut floating_base,
        )?;
    }

    if bodies.is_empty() {
        return Err(MjcfError::NoBodies);
    }

    Ok(ParsedModel {
        name,
        bodies,
        floating_base,
        ignored: ignored_sections(root),
    })
}

/// Reads one body, pushes it, then recurses into its own `<body>` children.
///
/// `inherited_class` is the nearest enclosing `childclass`, carried down until a descendant states
/// its own.
fn walk_body(
    node: Node,
    parent: Option<usize>,
    inherited_class: Option<&str>,
    table: &DefaultTable,
    settings: &CompilerSettings,
    bodies: &mut Vec<ParsedBody>,
    floating_base: &mut bool,
) -> Result<(), MjcfError> {
    let name = node.attribute("name").unwrap_or("body").to_owned();
    reject_orientation_attributes(node, "body")?;

    let position = parse_vector3(node, "pos")?.unwrap_or([0.0; 3]);
    let quat = parse_vector4(node, "quat")?.unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let pose = SE3::from_parts(
        SO3::from_quaternion(unit_quaternion(node, quat, "quat")?),
        Vector::new(position),
    );

    let class_chain = node.attribute("childclass").or(inherited_class);

    let joint_like: Vec<Node> = node
        .children()
        .filter(|child| {
            child.is_element() && matches!(child.tag_name().name(), "joint" | "freejoint")
        })
        .collect();
    let joint = match joint_like.as_slice() {
        [only] if is_free_joint(*only) => {
            if parent.is_some() {
                return Err(MjcfError::FreeJointNotAtRoot { body: name });
            }
            *floating_base = true;
            None
        }
        _ => read_joint(node, table, class_chain, settings, &name)?,
    };

    let inertia = match settings.inertia_from_geom {
        InertiaFromGeom::Always => synthesized_inertia(node, table, class_chain, &name)?,
        InertiaFromGeom::Never => {
            let inertial = element(node, "inertial").ok_or_else(|| MjcfError::NoInertiaSource {
                body: name.clone(),
            })?;
            reject_orientation_attributes(inertial, "inertial")?;
            stated_inertia(inertial)?
        }
        InertiaFromGeom::Auto => match element(node, "inertial") {
            Some(inertial) => {
                reject_orientation_attributes(inertial, "inertial")?;
                stated_inertia(inertial)?
            }
            None => synthesized_inertia(node, table, class_chain, &name)?,
        },
    };

    let index = bodies.len();
    bodies.push(ParsedBody {
        name,
        parent,
        pose,
        inertia,
        joint,
    });

    for child in elements(node, "body") {
        walk_body(
            child,
            Some(index),
            class_chain,
            table,
            settings,
            bodies,
            floating_base,
        )?;
    }
    Ok(())
}

/// Whether a joint-shaped element is a free joint: `<freejoint/>`, or `<joint type="free">`.
fn is_free_joint(node: Node) -> bool {
    match node.tag_name().name() {
        "freejoint" => true,
        "joint" => node.attribute("type") == Some("free"),
        _ => false,
    }
}

/// The sections a file carries that this loader reads nothing out of, named once each and in a
/// settled order so a table generated from them does not shuffle between runs.
#[must_use]
fn ignored_sections(root: Node) -> Vec<String> {
    let mut names: Vec<String> = root
        .children()
        .filter(Node::is_element)
        .map(|section| section.tag_name().name())
        .filter(|name| !READ_SECTIONS.contains(name))
        .map(str::to_owned)
        .collect();
    names.sort();
    names.dedup();
    names
}

/// The mass properties a file states outright, from either `diaginertia` or `fullinertia`.
fn stated_inertia(node: Node) -> Result<SpatialInertia<f64>, MjcfError> {
    let position = parse_vector3(node, "pos")?.unwrap_or([0.0; 3]);
    let mass = parse_scalar(node, "mass")?.ok_or_else(|| required(node, "mass"))?;

    let diagonal = parse_vector3(node, "diaginertia")?;
    let full = parse_vector6(node, "fullinertia")?;

    let tensor = match (diagonal, full) {
        (Some(principal), None) => {
            // The three numbers run along the axes of a frame turned by `quat` from the body's
            // own, so turn them back to read the whole tensor in the body's axes.
            let quat = parse_vector4(node, "quat")?.unwrap_or([1.0, 0.0, 0.0, 0.0]);
            let rotation = unit_quaternion(node, quat, "quat")?.to_rotation_matrix();
            rotation * Matrix::from_diagonal(principal) * rotation.transpose()
        }
        // `quat` says nothing about a full tensor; MuJoCo does not read it there either.
        (None, Some([ixx, iyy, izz, ixy, ixz, iyz])) => {
            Matrix::from([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        }
        _ => {
            return Err(bad_attribute(
                node,
                "fullinertia",
                node.attribute("fullinertia").unwrap_or_default(),
            ));
        }
    };

    SpatialInertia::new(mass, Vector::new(position), tensor).map_err(MjcfError::Inertia)
}

/// The mass properties worked out from the shapes a body is built from, for a body with no stated
/// inertia of its own. Only the body's own `<geom>` children are measured, never a descendant
/// body's.
fn synthesized_inertia(
    body: Node,
    table: &DefaultTable,
    class_chain: Option<&str>,
    name: &str,
) -> Result<SpatialInertia<f64>, MjcfError> {
    let mut shapes: Vec<GeomMass> = Vec::new();
    for geom in elements(body, "geom") {
        if let Some(shape) = read_geom(geom, table, class_chain, name)? {
            shapes.push(shape);
        }
    }

    // No shape carrying mass means nothing to work the body out from. Refusing here is what keeps
    // a massless body from ever loading.
    let total_mass: f64 = shapes.iter().map(|shape| shape.mass).sum();
    if total_mass == 0.0 {
        return Err(MjcfError::NoInertiaSource {
            body: name.to_owned(),
        });
    }

    // The body balances at the average of the shape centres, each weighted by its mass.
    let weighted = shapes.iter().fold(Vector::zeros(), |running, shape| {
        running + shape.center.scale(shape.mass)
    });
    let center_of_mass = weighted.scale(1.0 / total_mass);

    // Each shape resists being spun about the body's balance point by more than about its own
    // centre, by an amount set by its mass and how far apart the two points are.
    let summed = shapes.iter().fold(Matrix::zeros(), |running, shape| {
        running + shape.inertia + shifted(shape.mass, shape.center - center_of_mass)
    });

    SpatialInertia::new(total_mass, center_of_mass, summed).map_err(MjcfError::Inertia)
}

/// How much harder a mass is to spin about a point `offset` away from where it balances.
fn shifted(mass: f64, offset: Vector3D) -> Matrix3D {
    let distance_squared = offset.dot(offset);
    Matrix::from_fn(|row, column| {
        let along_diagonal = if row == column { distance_squared } else { 0.0 };
        mass * (along_diagonal - offset[row] * offset[column])
    })
}

/// The error for an attribute the file has to carry and does not.
#[must_use]
fn required(node: Node, attribute: &str) -> MjcfError {
    bad_attribute(
        node,
        attribute,
        node.attribute(attribute).unwrap_or_default(),
    )
}
