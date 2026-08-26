//! `<link>` parsing: name and spatial inertia.
//!
//! `<inertial>` is optional and never derived from geometry. A link without one, or one stating
//! zero mass, is massless: the usual encoding for tool and sensor frames.

use std::collections::HashMap;

use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::spatial::{Quaternion, SE3, SO3, SpatialInertia};
use roxmltree::Node;

use crate::urdf::visual::read_shapes;
use crate::xml::{bad_attribute, element, elements, parse_scalar, parse_vector3};
use crate::{ModelError, VisualGeometry};

/// A `<link>` as stated, before tree resolution.
pub(crate) struct ParsedLink {
    pub name: String,
    pub inertia: Option<SpatialInertia<f64>>,
    pub visual_geometry: Vec<VisualGeometry>,
}

/// Every `<link>` child of `<robot>`, in document order.
pub(crate) fn read_links(
    root: Node,
    materials: &HashMap<String, [f64; 4]>,
) -> Result<Vec<ParsedLink>, ModelError> {
    let mut links = Vec::new();
    for node in elements(root, "link") {
        let name = node.attribute("name").unwrap_or("link").to_owned();
        let inertia = match element(node, "inertial") {
            Some(inertial) => read_inertial(inertial)?,
            None => None,
        };
        links.push(ParsedLink {
            name,
            inertia,
            visual_geometry: read_shapes(node, materials)?,
        });
    }
    Ok(links)
}

/// Spatial inertia from one `<inertial>` block, or `None` where it states zero mass.
///
/// `<origin>` gives the COM and the frame the tensor is expressed in; the tensor is rotated into
/// link axes as `R I Rᵀ`.
fn read_inertial(node: Node) -> Result<Option<SpatialInertia<f64>>, ModelError> {
    let mass_node = element(node, "mass").ok_or_else(|| bad_attribute(node, "mass", ""))?;
    let mass = required(mass_node, "value")?;

    // Zero mass is a frame, as an absent `<inertial>` is. Nothing else in the block is read: the
    // mass is what says the link carries no inertia, and a file writing one often omits the tensor
    // or leaves it at zero. A negative mass is still refused.
    if mass == 0.0 {
        return Ok(None);
    }

    let origin = read_origin(node)?;
    let inertia_node =
        element(node, "inertia").ok_or_else(|| bad_attribute(node, "inertia", ""))?;
    let stated = Matrix::from([
        [
            required(inertia_node, "ixx")?,
            required(inertia_node, "ixy")?,
            required(inertia_node, "ixz")?,
        ],
        [
            required(inertia_node, "ixy")?,
            required(inertia_node, "iyy")?,
            required(inertia_node, "iyz")?,
        ],
        [
            required(inertia_node, "ixz")?,
            required(inertia_node, "iyz")?,
            required(inertia_node, "izz")?,
        ],
    ]);

    let rotation = origin.rotation().to_matrix();
    let tensor = rotation * stated * rotation.transpose();

    SpatialInertia::new(mass, origin.translation(), tensor)
        .map(Some)
        .map_err(ModelError::Inertia)
}

/// The transform an `<origin xyz rpy>` child gives, or identity if absent.
///
/// `rpy` is fixed-axis roll-pitch-yaw: `R = Rz(yaw)·Ry(pitch)·Rx(roll)`.
pub(crate) fn read_origin(parent: Node) -> Result<SE3<f64>, ModelError> {
    let Some(node) = element(parent, "origin") else {
        return Ok(SE3::identity());
    };
    let translation = parse_vector3(node, "xyz")?.unwrap_or([0.0; 3]);
    let [roll, pitch, yaw] = parse_vector3(node, "rpy")?.unwrap_or([0.0; 3]);
    Ok(SE3::from_parts(
        SO3::from_quaternion(Quaternion::from_euler_zyx(roll, pitch, yaw)),
        Vector::new(translation),
    ))
}

/// A required scalar attribute.
fn required(node: Node, attribute: &'static str) -> Result<f64, ModelError> {
    parse_scalar(node, attribute)?.ok_or_else(|| bad_attribute(node, attribute, ""))
}
