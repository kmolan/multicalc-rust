//! Reading the links a URDF file describes: name and mass properties.
//!
//! URDF states a link's mass outright or not at all — there is no working it out from shapes, and
//! a link with none is a real link that simply carries no mass, which is how tool and sensor
//! frames are written.

use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::spatial::{Quaternion, SE3, SO3, SpatialInertia};
use roxmltree::Node;

use crate::ModelError;
use crate::xml::{bad_attribute, element, elements, parse_scalar, parse_vector3};

/// One link as the file states it, before the tree is worked out.
pub(crate) struct ParsedLink {
    pub name: String,
    pub inertia: Option<SpatialInertia<f64>>,
}

/// Reads every `<link>` child of `<robot>`, in document order.
pub(crate) fn read_links(root: Node) -> Result<Vec<ParsedLink>, ModelError> {
    let mut links = Vec::new();
    for node in elements(root, "link") {
        let name = node.attribute("name").unwrap_or("link").to_owned();
        let inertia = match element(node, "inertial") {
            Some(inertial) => Some(read_inertial(inertial)?),
            None => None,
        };
        links.push(ParsedLink { name, inertia });
    }
    Ok(links)
}

/// The mass properties one `<inertial>` block states.
///
/// Its `<origin>` says where the link balances and which way the stated numbers are turned, so the
/// six figures are turned back into the link's own axes before they are stored.
fn read_inertial(node: Node) -> Result<SpatialInertia<f64>, ModelError> {
    let origin = read_origin(node)?;

    let mass_node = element(node, "mass").ok_or_else(|| bad_attribute(node, "mass", ""))?;
    let mass = required(mass_node, "value")?;

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

    SpatialInertia::new(mass, origin.translation(), tensor).map_err(ModelError::Inertia)
}

/// The pose an `<origin xyz rpy>` child describes, or the identity where there is none.
///
/// `rpy` is a roll, then a pitch, then a yaw, each about a fixed axis.
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

/// The one number an attribute the file has to carry holds.
fn required(node: Node, attribute: &'static str) -> Result<f64, ModelError> {
    parse_scalar(node, attribute)?.ok_or_else(|| bad_attribute(node, attribute, ""))
}
