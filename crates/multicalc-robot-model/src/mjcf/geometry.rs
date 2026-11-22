//! Geom mass, rotational inertia, and drawable shape.
//!
//! Settings resolve through the default-class chain first; integration happens afterwards.

use std::f64::consts::PI;

use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use multicalc::spatial::{Quaternion, SE3, SO3};
use roxmltree::Node;

use crate::mjcf::asset::AssetTable;
use crate::mjcf::compiler::CompilerSettings;
use crate::mjcf::defaults::{DefaultTable, GeomDefaults};
use crate::xml::bad_attribute;
use crate::{GeometryShape, ModelError, VisualGeometry};

/// MuJoCo's defaults for an unspecified geom. Size has no default.
const ASSUMED_TYPE: &str = "sphere";
const ASSUMED_DENSITY: f64 = 1000.0;
const ASSUMED_RGBA: [f64; 4] = [0.5, 0.5, 0.5, 1.0];

/// Geom types this reader can integrate.
enum Shape {
    Sphere,
    Ellipsoid,
    Box,
    Cylinder,
    Capsule,
}

/// A `fromto` resolved to half-length, centre and turn, standing in for `size`, `pos` and `quat`.
struct Axis {
    half_length: f64,
    center: [f64; 3],
    turn: Quaternion<f64>,
}

/// One geom's contribution to its body, in body axes.
pub(crate) struct GeomMass {
    pub mass: f64,
    /// About the geom's own centre.
    pub center: Vector3D,
    pub inertia: Matrix3D,
}

/// Integrates one geom, or reports that it is massless and contributes nothing.
pub(crate) fn read_geom(
    node: Node,
    table: &DefaultTable,
    class_chain: Option<&str>,
    compiler: &CompilerSettings,
    body: &str,
) -> Result<Option<GeomMass>, ModelError> {
    let settings = effective(node, table, class_chain)?;

    // Dropped before the type is checked, so an unintegrable type is allowed while massless.
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
            return Err(ModelError::MeshInertiaUnsupported {
                body: body.to_owned(),
            });
        }
        other => {
            return Err(ModelError::UnsupportedGeomType {
                body: body.to_owned(),
                geom_type: other.to_owned(),
            });
        }
    };

    // `fromto` gives the axis endpoints in place of a length and orientation.
    let axis = axis(node, &settings, &shape, body)?;
    let [first, second, third] = extents(node, &settings, &shape, axis.as_ref())?;

    // `I_kk = m · (g_i + g_j)` for the two axes across from `k`, so each type reduces to its
    // volume and its three squared radii of gyration `g`.
    let (volume, spread) = match shape {
        // `g_i = a_i² / 5` per semi-axis.
        Shape::Sphere | Shape::Ellipsoid => (
            4.0 / 3.0 * PI * first * second * third,
            [
                first * first / 5.0,
                second * second / 5.0,
                third * third / 5.0,
            ],
        ),
        // `g_i = h_i² / 3` per half-width.
        Shape::Box => (
            8.0 * first * second * third,
            [
                first * first / 3.0,
                second * second / 3.0,
                third * third / 3.0,
            ],
        ),
        // `g = [r²/4, r²/4, l²/3]`: a disc across, a bar along.
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

    // The principal moments stand in the shape's own axes: `R I Rᵀ` puts them in the body's.
    // A `fromto` carries its own facing, and MuJoCo lets that beat any form written alongside it.
    let turn = match &axis {
        Some(axis) => axis.turn,
        None => settings.orientation.resolve(node, compiler)?,
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

/// One geom as a drawable shape, or `None` for a type or mesh reference this cannot draw.
///
/// Unlike [`read_geom`] it rejects nothing by type: an undrawable shape carries no mass.
pub(crate) fn read_visual_geom(
    node: Node,
    table: &DefaultTable,
    class_chain: Option<&str>,
    assets: &AssetTable,
    compiler: &CompilerSettings,
    body: &str,
) -> Result<Option<VisualGeometry>, ModelError> {
    let settings = effective(node, table, class_chain)?;
    let group = settings.group.unwrap_or(0.0).max(0.0) as u32;

    // Off the resolved chain, never off `node`: `unitree_go1` states its material only in
    // `<default class="visual">`, and its trunk geom names none.
    let color = match settings.rgba {
        Some(stated) => stated,
        None => settings
            .material
            .as_deref()
            .and_then(|name| assets.material(name))
            .unwrap_or(ASSUMED_RGBA),
    };

    let shape = match settings.geom_type.as_deref().unwrap_or(ASSUMED_TYPE) {
        "sphere" => Shape::Sphere,
        "ellipsoid" => Shape::Ellipsoid,
        "box" => Shape::Box,
        "cylinder" => Shape::Cylinder,
        "capsule" => Shape::Capsule,
        "mesh" => {
            let Some(asset) = settings.mesh.as_deref().and_then(|name| assets.mesh(name)) else {
                return Ok(None);
            };
            let shape = GeometryShape::Mesh {
                file: asset.file.clone(),
                scale: Vector::new(asset.scale),
            };
            let turn = settings.orientation.resolve(node, compiler)?;
            let pose = SE3::from_parts(
                SO3::from_quaternion(turn),
                Vector::new(settings.pos.unwrap_or([0.0; 3])),
            );
            return Ok(Some(VisualGeometry::new(shape, pose, color, group)));
        }
        _ => return Ok(None),
    };

    // Guarded, so `axis` never rejects a `fromto` it cannot measure: an integration limit is not a
    // reason to leave the geom undrawn.
    let axis = match shape {
        Shape::Cylinder | Shape::Capsule => axis(node, &settings, &shape, body)?,
        _ => None,
    };
    let [first, second, third] = extents(node, &settings, &shape, axis.as_ref())?;

    let (center, turn) = match &axis {
        Some(axis) => (axis.center, axis.turn),
        None => (
            settings.pos.unwrap_or([0.0; 3]),
            settings.orientation.resolve(node, compiler)?,
        ),
    };
    let pose = SE3::from_parts(SO3::from_quaternion(turn), Vector::new(center));

    let shape = match shape {
        Shape::Sphere => GeometryShape::Sphere { radius: first },
        Shape::Ellipsoid => GeometryShape::Ellipsoid {
            semi_axes: Vector::new([first, second, third]),
        },
        Shape::Box => GeometryShape::Box {
            half_extents: Vector::new([first, second, third]),
        },
        Shape::Cylinder => GeometryShape::Cylinder {
            radius: first,
            half_length: third,
        },
        Shape::Capsule => GeometryShape::Capsule {
            radius: first,
            half_length: third,
        },
    };
    Ok(Some(VisualGeometry::new(shape, pose, color, group)))
}

/// A geom's settings: its own beat the class it names, that beats its body's `childclass`, that
/// beats the unnamed block.
fn effective(
    node: Node,
    table: &DefaultTable,
    class_chain: Option<&str>,
) -> Result<GeomDefaults, ModelError> {
    let mut settings = table.resolve(None)?.geom.clone();
    if let Some(name) = class_chain {
        settings = settings.overridden_by(&table.resolve(Some(name))?.geom);
    }
    if let Some(name) = node.attribute("class") {
        settings = settings.overridden_by(&table.resolve(Some(name))?.geom);
    }
    Ok(settings.overridden_by(&GeomDefaults::read(node)?))
}

/// A capsule's volume and squared radii of gyration: barrel and caps, weighted by volume share.
///
/// The caps sit a half-length off the mid-plane, so their axial term carries the parallel-axis
/// shift `l² + ¾ r l` on top of their own spread.
fn capsule(radius: f64, half_length: f64) -> (f64, [f64; 3]) {
    let barrel = 2.0 * PI * radius * radius * half_length;
    let caps = 4.0 / 3.0 * PI * radius * radius * radius;
    let volume = barrel + caps;

    // No volume, no shares to weight by.
    if volume == 0.0 {
        return (0.0, [0.0; 3]);
    }
    let (barrel_share, caps_share) = (barrel / volume, caps / volume);

    // A hemisphere spreads across an axis as the whole ball does: `r² / 5`.
    let across = radius * radius / 5.0;
    // Axially, that same fifth about each cap's own centroid, plus the parallel-axis shift.
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

/// A `fromto`'s half-length, centre and facing: half the span, the midpoint, the line through them.
///
/// MuJoCo allows `fromto` on boxes and ellipsoids too, reading their remaining size numbers in a way
/// this loader has not pinned down against the compiler, so only the round forms are taken.
fn axis(
    node: Node,
    settings: &GeomDefaults,
    shape: &Shape,
    body: &str,
) -> Result<Option<Axis>, ModelError> {
    let Some(ends) = settings.fromto else {
        return Ok(None);
    };
    if !matches!(shape, Shape::Cylinder | Shape::Capsule) {
        return Err(ModelError::UnsupportedFromTo {
            body: body.to_owned(),
            geom_type: settings
                .geom_type
                .clone()
                .unwrap_or_else(|| ASSUMED_TYPE.to_owned()),
        });
    }
    // The ends already place the shape, so a `pos` beside them is a second answer. MuJoCo refuses
    // the pair.
    if settings.pos.is_some() {
        return Err(ModelError::ConflictingPlacement {
            body: body.to_owned(),
        });
    }

    let start = Vector::new([ends[0], ends[1], ends[2]]);
    let end = Vector::new([ends[3], ends[4], ends[5]]);
    let along = end - start;

    // Coincident ends pin down no direction.
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
        // Both round forms are symmetric about their axis, so end order does not matter.
        turn: Quaternion::from_two_vectors(Vector::new([0.0, 0.0, 1.0]), along),
    }))
}

/// Reach along each of the shape's own axes: ellipsoid semi-axes, box half-widths, a sphere's
/// radius three times over, `[r, r, half_length]` for the round forms.
fn extents(
    node: Node,
    settings: &GeomDefaults,
    shape: &Shape,
    axis: Option<&Axis>,
) -> Result<[f64; 3], ModelError> {
    let size = settings.size.as_deref().unwrap_or_default();
    let reach = match (shape, size, axis) {
        // A `fromto` already gives the half-length, and MuJoCo discards what `size` says about it.
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
