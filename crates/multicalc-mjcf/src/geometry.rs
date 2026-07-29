//! How much mass one shape carries, and how hard it is to spin.
//!
//! A shape's settings can come from the shape itself or from any default block it inherits, so
//! they are gathered in precedence order first and measured afterwards.

use std::f64::consts::PI;

use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use roxmltree::Node;

use crate::MjcfError;
use crate::defaults::{DefaultTable, GeomDefaults, bad_attribute, unit_quaternion};

/// What MuJoCo assumes for a shape that states nothing. There is no assumed size.
const ASSUMED_TYPE: &str = "sphere";
const ASSUMED_DENSITY: f64 = 1000.0;

/// The shapes this loader can measure.
enum Shape {
    Sphere,
    Ellipsoid,
    Box,
}

/// One shape's contribution to the body it belongs to.
pub(crate) struct GeomMass {
    /// How much mass the shape carries.
    pub mass: f64,
    /// Where the shape's own centre sits, in the body's axes.
    pub center: Vector3D,
    /// How the shape resists being spun about its own centre, in the body's axes.
    pub inertia: Matrix3D,
}

/// Measures one shape, or reports that it carries no mass and so contributes nothing.
pub(crate) fn read_geom(
    node: Node,
    table: &DefaultTable,
    childclass: Option<&str>,
    body: &str,
) -> Result<Option<GeomMass>, MjcfError> {
    let settings = effective(node, table, childclass)?;

    // A shape stated to carry no mass is dropped before its form is looked at, so a model can name
    // a shape this loader cannot measure as long as none of its mass rests there.
    if settings.mass == Some(0.0) {
        return Ok(None);
    }

    let shape = match settings.geom_type.as_deref().unwrap_or(ASSUMED_TYPE) {
        "sphere" => Shape::Sphere,
        "ellipsoid" => Shape::Ellipsoid,
        "box" => Shape::Box,
        "mesh" => {
            return Err(MjcfError::MeshInertiaUnsupported {
                body: body.to_owned(),
            });
        }
        other => {
            return Err(MjcfError::UnsupportedGeomType {
                body: body.to_owned(),
                geom_type: other.to_owned(),
            });
        }
    };

    // Every shape here resists spinning by the same pattern — the sum of the squares of the two
    // extents across from each axis — and they differ only in the share of the mass that counts.
    // A sphere is an ellipsoid that reaches equally far in every direction, so the two share a
    // measurement.
    let [first, second, third] = extents(node, &settings, &shape)?;
    let (volume, share) = match shape {
        Shape::Sphere | Shape::Ellipsoid => (4.0 / 3.0 * PI * first * second * third, 5.0),
        Shape::Box => (8.0 * first * second * third, 3.0),
    };

    let mass = match settings.mass {
        Some(stated) => stated,
        None => settings.density.unwrap_or(ASSUMED_DENSITY) * volume,
    };
    if mass == 0.0 {
        return Ok(None);
    }

    let part = mass / share;
    let principal = [
        part * (second * second + third * third),
        part * (first * first + third * third),
        part * (first * first + second * second),
    ];

    // Those three numbers are along the shape's own axes, so turn them into the body's.
    let quat = settings.quat.unwrap_or([1.0, 0.0, 0.0, 0.0]);
    let rotation = unit_quaternion(node, quat, "quat")?.to_rotation_matrix();
    let inertia = rotation * Matrix::from_diagonal(principal) * rotation.transpose();

    Ok(Some(GeomMass {
        mass,
        center: Vector::new(settings.pos.unwrap_or([0.0; 3])),
        inertia,
    }))
}

/// A shape's settings, with the shape's own winning over the class it names, that over the class
/// its body names, and that over the unnamed block.
fn effective(
    node: Node,
    table: &DefaultTable,
    childclass: Option<&str>,
) -> Result<GeomDefaults, MjcfError> {
    let mut settings = table.resolve(None)?.clone();
    if let Some(name) = childclass {
        settings = settings.overridden_by(table.resolve(Some(name))?);
    }
    if let Some(name) = node.attribute("class") {
        settings = settings.overridden_by(table.resolve(Some(name))?);
    }
    Ok(settings.overridden_by(&GeomDefaults::read(node)?))
}

/// How far a shape reaches along each of its own axes: the semi-axes of an ellipsoid, or the
/// half-widths of a box. A sphere states only its radius, and reaches that far every way.
fn extents(node: Node, settings: &GeomDefaults, shape: &Shape) -> Result<[f64; 3], MjcfError> {
    let size = settings.size.as_deref().unwrap_or_default();
    let reach = match (shape, size) {
        (Shape::Sphere, [radius]) => Some([*radius; 3]),
        (Shape::Sphere, _) => None,
        (_, _) => <[f64; 3]>::try_from(size).ok(),
    };
    reach.ok_or_else(|| bad_attribute(node, "size", node.attribute("size").unwrap_or_default()))
}
