//! How much mass one shape carries, and how hard it is to spin.
//!
//! A shape's settings can come from the shape itself or from any default block it inherits, so
//! they are gathered in precedence order first and measured afterwards.

use std::f64::consts::PI;

use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use multicalc::spatial::Quaternion;
use roxmltree::Node;

use crate::MjcfError;
use crate::defaults::{
    DefaultTable, GeomDefaults, bad_attribute, reject_orientation_attributes, unit_quaternion,
};

/// What MuJoCo assumes for a shape that states nothing. There is no assumed size.
const ASSUMED_TYPE: &str = "sphere";
const ASSUMED_DENSITY: f64 = 1000.0;

/// The shapes this loader can measure.
enum Shape {
    Sphere,
    Ellipsoid,
    Box,
    Cylinder,
    Capsule,
}

/// What a shape's two stated ends work out to: how long it is, where it sits, and which way it
/// faces. These stand in for the half-length its size would otherwise give and for `pos` and
/// `quat`, so once they are worked out the measurement proceeds as for any other shape.
struct Axis {
    half_length: f64,
    center: [f64; 3],
    turn: Quaternion<f64>,
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
    class_chain: Option<&str>,
    body: &str,
) -> Result<Option<GeomMass>, MjcfError> {
    reject_orientation_attributes(node, "geom")?;
    let settings = effective(node, table, class_chain)?;

    // A shape stated to carry no mass is dropped before its form is looked at, so a model can name
    // a shape this loader cannot measure as long as none of its mass rests there.
    if settings.mass == Some(0.0) {
        return Ok(None);
    }

    let shape = match settings.geom_type.as_deref().unwrap_or(ASSUMED_TYPE) {
        "sphere" => Shape::Sphere,
        "ellipsoid" => Shape::Ellipsoid,
        "box" => Shape::Box,
        "cylinder" => Shape::Cylinder,
        "capsule" => Shape::Capsule,
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

    // A shape can say where the two ends of its axis are rather than how long it is and which way
    // it faces, which is how a model built around where its joints sit is usually written.
    let axis = axis(node, &settings, &shape, body)?;
    let [first, second, third] = extents(node, &settings, &shape, axis.as_ref())?;

    // Every shape here resists spinning by the same pattern — the mass, times how far that mass
    // sits from the axis, summed over the two axes across from it. So a form contributes only how
    // much room it takes up and how widely it spreads its mass along each of its own three axes,
    // and each form is measured into those four numbers once.
    let (volume, spread) = match shape {
        // A ball stretched by a different amount along each axis, so every axis spreads its reach
        // the same way: a fifth of the square of it.
        Shape::Sphere | Shape::Ellipsoid => (
            4.0 / 3.0 * PI * first * second * third,
            [
                first * first / 5.0,
                second * second / 5.0,
                third * third / 5.0,
            ],
        ),
        // A cube stretched the same way, which spreads its mass further out than a ball does: a
        // third of the square of each half-width rather than a fifth.
        Shape::Box => (
            8.0 * first * second * third,
            [
                first * first / 3.0,
                second * second / 3.0,
                third * third / 3.0,
            ],
        ),
        // A disc across and a bar along, which do not spread alike — this is where one fraction
        // for the whole shape stops being enough. Either way across is a quarter of the square of
        // the radius; along the axis it is a third of the square of the half-length.
        Shape::Cylinder => (
            2.0 * PI * first * first * third,
            [
                first * first / 4.0,
                first * first / 4.0,
                third * third / 3.0,
            ],
        ),
        Shape::Capsule => capsule(first, third),
    };

    let mass = match settings.mass {
        Some(stated) => stated,
        None => settings.density.unwrap_or(ASSUMED_DENSITY) * volume,
    };
    if mass == 0.0 {
        return Ok(None);
    }

    let principal = [
        mass * (spread[1] + spread[2]),
        mass * (spread[0] + spread[2]),
        mass * (spread[0] + spread[1]),
    ];

    // Those three numbers are along the shape's own axes, so turn them into the body's. Stated
    // ends carry their own facing, and MuJoCo lets that beat a `quat` written alongside them
    // rather than refusing the pair, so the same is done here.
    let turn = match &axis {
        Some(axis) => axis.turn,
        None => unit_quaternion(node, settings.quat.unwrap_or([1.0, 0.0, 0.0, 0.0]), "quat")?,
    };
    let rotation = turn.to_rotation_matrix();
    let inertia = rotation * Matrix::from_diagonal(principal) * rotation.transpose();

    let center = match &axis {
        Some(axis) => axis.center,
        None => settings.pos.unwrap_or([0.0; 3]),
    };

    Ok(Some(GeomMass {
        mass,
        center: Vector::new(center),
        inertia,
    }))
}

/// A shape's settings, with the shape's own winning over the class it names, that over the class
/// its body names, and that over the unnamed block.
fn effective(
    node: Node,
    table: &DefaultTable,
    class_chain: Option<&str>,
) -> Result<GeomDefaults, MjcfError> {
    let mut settings = table.resolve(None)?.geom.clone();
    if let Some(name) = class_chain {
        settings = settings.overridden_by(&table.resolve(Some(name))?.geom);
    }
    if let Some(name) = node.attribute("class") {
        settings = settings.overridden_by(&table.resolve(Some(name))?.geom);
    }
    Ok(settings.overridden_by(&GeomDefaults::read(node)?))
}

/// A capsule is a cylinder with a hemisphere capping each end, so it is measured as the two bodies
/// it is built from and then blended: each part spreads its own mass its own way, and counts for
/// the share of the whole it carries.
///
/// The caps are the reason a capsule needs more than a cylinder does. Their mass does not sit on
/// the plane through the middle but a half-length off it, and moving mass away from a plane adds
/// the square of how far it moved — that is the parallel-axis shift, and it is what the two terms
/// beyond the cap's own spread account for.
fn capsule(radius: f64, half_length: f64) -> (f64, [f64; 3]) {
    let barrel = 2.0 * PI * radius * radius * half_length;
    let caps = 4.0 / 3.0 * PI * radius * radius * radius;
    let volume = barrel + caps;

    // A shape that takes up no room has no mass to share out between its parts, and a mass stated
    // for one has nowhere to sit, so there is nothing left that resists being spun.
    if volume == 0.0 {
        return (0.0, [0.0; 3]);
    }
    let (barrel_share, caps_share) = (barrel / volume, caps / volume);

    // A hemisphere spreads its mass across an axis exactly as the whole ball it is half of does,
    // so the caps count a fifth of the square of the radius either way across.
    let across = radius * radius / 5.0;
    // Along the axis they carry that same fifth, about where each cap balances — three eighths of
    // the radius out from its flat face — plus the square of how far that point is from the middle.
    let along = half_length * half_length + 0.75 * radius * half_length + across;

    (
        volume,
        [
            barrel_share * radius * radius / 4.0 + caps_share * across,
            barrel_share * radius * radius / 4.0 + caps_share * across,
            barrel_share * half_length * half_length / 3.0 + caps_share * along,
        ],
    )
}

/// Reads the two ends of a shape's axis, when it gives them.
///
/// A `fromto` says where a shape starts and where it stops, and everything the measurement needs
/// follows from that: the shape is half as long as the two ends are apart, sits at their middle,
/// and faces along the line between them. MuJoCo allows this on boxes and ellipsoids too, but
/// reads their remaining size numbers in a way this loader has not pinned down against the
/// compiler, so only the two round forms are taken and the rest are refused by name.
fn axis(
    node: Node,
    settings: &GeomDefaults,
    shape: &Shape,
    body: &str,
) -> Result<Option<Axis>, MjcfError> {
    let Some(ends) = settings.fromto else {
        return Ok(None);
    };
    if !matches!(shape, Shape::Cylinder | Shape::Capsule) {
        return Err(MjcfError::UnsupportedFromTo {
            body: body.to_owned(),
            geom_type: settings
                .geom_type
                .clone()
                .unwrap_or_else(|| ASSUMED_TYPE.to_owned()),
        });
    }
    // The two ends already say where the shape sits, so a `pos` alongside them is a second answer
    // to the same question. MuJoCo refuses that pair outright, and so does this.
    if settings.pos.is_some() {
        return Err(MjcfError::ConflictingPlacement {
            body: body.to_owned(),
        });
    }

    let start = Vector::new([ends[0], ends[1], ends[2]]);
    let end = Vector::new([ends[3], ends[4], ends[5]]);
    let along = end - start;

    // Two ends in the same place pin down no direction, so there is no shape to measure.
    let length = along.norm();
    if length == 0.0 {
        return Err(bad_attribute(
            node,
            "fromto",
            node.attribute("fromto").unwrap_or_default(),
        ));
    }

    Ok(Some(Axis {
        half_length: length / 2.0,
        center: (start + end).scale(0.5).into_array(),
        // Which end is which does not matter: both forms this is read for look the same either
        // way along their axis, so a turn onto the line is as good as a turn onto the direction.
        turn: Quaternion::from_two_vectors(Vector::new([0.0, 0.0, 1.0]), along),
    }))
}

/// How far a shape reaches along each of its own axes: the semi-axes of an ellipsoid, or the
/// half-widths of a box. A sphere states only its radius, and reaches that far every way. A
/// cylinder and a capsule state a radius and the half-length of the barrel between their ends,
/// and are round about the third axis, so the radius stands for the first two.
fn extents(
    node: Node,
    settings: &GeomDefaults,
    shape: &Shape,
    axis: Option<&Axis>,
) -> Result<[f64; 3], MjcfError> {
    let size = settings.size.as_deref().unwrap_or_default();
    let reach = match (shape, size, axis) {
        // A shape whose ends are stated has already been measured along its axis, and MuJoCo
        // discards whatever the size says about that, so only the radius is read from it here.
        (Shape::Cylinder | Shape::Capsule, [radius, ..], Some(axis)) => {
            Some([*radius, *radius, axis.half_length])
        }
        (Shape::Sphere, [radius], None) => Some([*radius; 3]),
        (Shape::Cylinder | Shape::Capsule, [radius, half_length], None) => {
            Some([*radius, *radius, *half_length])
        }
        (Shape::Sphere | Shape::Cylinder | Shape::Capsule, _, _) => None,
        (_, _, _) => <[f64; 3]>::try_from(size).ok(),
    };
    reach.ok_or_else(|| bad_attribute(node, "size", node.attribute("size").unwrap_or_default()))
}
