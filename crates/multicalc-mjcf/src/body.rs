//! Reading the one body a model file describes.
//!
//! Where the file states the body's mass properties they are used as written. Where it does not,
//! they are worked out from the shapes the body is built from.

use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use multicalc::spatial::{SE3, SO3, SpatialInertia};
use roxmltree::{Document, Node};

use crate::MjcfError;
use crate::defaults::{
    DefaultTable, bad_attribute, element, elements, parse_scalar, parse_vector3, parse_vector4,
    unit_quaternion,
};
use crate::geometry::{GeomMass, read_geom};

/// MuJoCo reads a joint that names no type as a hinge.
const ASSUMED_JOINT: &str = "hinge";

/// Everything one model file says about its body.
pub(crate) struct BodyModel {
    pub name: String,
    pub pose: SE3<f64>,
    pub has_free_joint: bool,
    pub inertia: SpatialInertia<f64>,
}

/// Reads the single body a file describes, refusing anything outside that by name.
pub(crate) fn read(document: &Document) -> Result<BodyModel, MjcfError> {
    let root = document.root_element();

    // A file that pulls in another file is refused before anything is read out of it, so a model
    // is never built from only the half that is understood.
    if root
        .descendants()
        .any(|node| node.is_element() && node.tag_name().name() == "include")
    {
        return Err(MjcfError::IncludeUnsupported);
    }

    // Read and ignore: nothing in this subset is written in angle units, so whether the file works
    // in degrees or radians cannot change what is loaded.
    let _angle = element(root, "compiler").and_then(|node| node.attribute("angle"));

    let table = DefaultTable::build(root)?;
    let worldbody = element(root, "worldbody").ok_or(MjcfError::MissingWorldbody)?;

    let bodies: Vec<Node> = worldbody
        .descendants()
        .filter(|node| node.is_element() && node.tag_name().name() == "body")
        .collect();
    let body = match bodies.as_slice() {
        [] => return Err(MjcfError::NoBodies),
        [only] => *only,
        several => {
            return Err(MjcfError::MultipleBodies {
                count: several.len(),
            });
        }
    };

    let name = body.attribute("name").unwrap_or("body").to_owned();
    let position = parse_vector3(body, "pos")?.unwrap_or([0.0; 3]);
    let quat = parse_vector4(body, "quat")?.unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let pose = SE3::from_parts(
        SO3::from_quaternion(unit_quaternion(body, quat, "quat")?),
        Vector::new(position),
    );

    let has_free_joint = read_free_joint(body, &name)?;
    let childclass = body.attribute("childclass");
    let inertia = match element(body, "inertial") {
        Some(inertial) => stated_inertia(inertial)?,
        None => synthesized_inertia(body, &table, childclass, &name)?,
    };

    Ok(BodyModel {
        name,
        pose,
        has_free_joint,
        inertia,
    })
}

/// Checks that the body hangs off the world by a free joint and nothing else.
fn read_free_joint(body: Node, name: &str) -> Result<bool, MjcfError> {
    let mut free = false;
    for child in body.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "freejoint" => free = true,
            "joint" => {
                let joint_type = child.attribute("type").unwrap_or(ASSUMED_JOINT);
                if joint_type != "free" {
                    return Err(MjcfError::UnsupportedJoint {
                        body: name.to_owned(),
                        joint_type: joint_type.to_owned(),
                    });
                }
                free = true;
            }
            _ => {}
        }
    }
    if free {
        Ok(true)
    } else {
        Err(MjcfError::MissingFreeJoint {
            body: name.to_owned(),
        })
    }
}

/// The mass properties a file states outright.
fn stated_inertia(node: Node) -> Result<SpatialInertia<f64>, MjcfError> {
    let position = parse_vector3(node, "pos")?.unwrap_or([0.0; 3]);
    let quat = parse_vector4(node, "quat")?.unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let mass = parse_scalar(node, "mass")?.ok_or_else(|| required(node, "mass"))?;
    let principal =
        parse_vector3(node, "diaginertia")?.ok_or_else(|| required(node, "diaginertia"))?;

    // The three numbers run along the axes of a frame turned by `quat` from the body's own, so
    // turn them back to read the whole tensor in the body's axes.
    let rotation = unit_quaternion(node, quat, "quat")?.to_rotation_matrix();
    let rotated = rotation * Matrix::from_diagonal(principal) * rotation.transpose();

    SpatialInertia::new(mass, Vector::new(position), rotated).map_err(MjcfError::Inertia)
}

/// The mass properties worked out from the shapes a body is built from, for a file that states
/// none of its own.
fn synthesized_inertia(
    body: Node,
    table: &DefaultTable,
    childclass: Option<&str>,
    name: &str,
) -> Result<SpatialInertia<f64>, MjcfError> {
    let mut shapes: Vec<GeomMass> = Vec::new();
    for geom in elements(body, "geom") {
        if let Some(shape) = read_geom(geom, table, childclass, name)? {
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
