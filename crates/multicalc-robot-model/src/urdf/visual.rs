//! `<visual>` and `<collision>` shapes.
//!
//! Read for viewing only. Both are kept and tagged by group, 0 for visual and 3 for collision as
//! MJCF numbers them, so one group filter serves both formats.

use std::collections::HashMap;

use multicalc::linear_algebra::Vector;
use roxmltree::Node;

use crate::urdf::link::read_origin;
use crate::xml::{bad_attribute, element, elements, parse_scalar, parse_vector3, parse_vector4};
use crate::{GeometryShape, ModelError, VisualGeometry};

/// Colour for a shape naming no material.
const ASSUMED_RGBA: [f64; 4] = [0.5, 0.5, 0.5, 1.0];

/// Group numbers, matching MJCF's convention.
const VISUAL_GROUP: u32 = 0;
const COLLISION_GROUP: u32 = 3;

/// A mesh stating no `scale`.
const ASSUMED_MESH_SCALE: [f64; 3] = [1.0, 1.0, 1.0];

/// Every top-level `<material>`, by name. One stating no `<color>` is skipped.
pub(crate) fn read_materials(root: Node) -> Result<HashMap<String, [f64; 4]>, ModelError> {
    let mut materials = HashMap::new();
    for node in elements(root, "material") {
        let Some(name) = node.attribute("name") else {
            continue;
        };
        let Some(color) = element(node, "color") else {
            continue;
        };
        materials.insert(name.to_owned(), read_rgba(color)?);
    }
    Ok(materials)
}

/// One link's `<visual>` and `<collision>` shapes, visuals first.
pub(crate) fn read_shapes(
    link: Node,
    materials: &HashMap<String, [f64; 4]>,
) -> Result<Vec<VisualGeometry>, ModelError> {
    let mut shapes = Vec::new();
    for (tag, group) in [("visual", VISUAL_GROUP), ("collision", COLLISION_GROUP)] {
        for node in elements(link, tag) {
            let Some(geometry) = element(node, "geometry") else {
                continue;
            };
            let Some(shape) = read_shape(geometry)? else {
                continue;
            };
            shapes.push(VisualGeometry::new(
                shape,
                read_origin(node)?,
                color(node, materials)?,
                group,
            ));
        }
    }
    Ok(shapes)
}

/// The `<geometry>`'s own child as a shape, or `None` for a form this cannot draw.
fn read_shape(geometry: Node) -> Result<Option<GeometryShape>, ModelError> {
    let Some(node) = geometry.children().find(Node::is_element) else {
        return Ok(None);
    };
    match node.tag_name().name() {
        // URDF states full size, MJCF half-widths.
        "box" => {
            let size =
                parse_vector3(node, "size")?.ok_or_else(|| bad_attribute(node, "size", ""))?;
            Ok(Some(GeometryShape::Box {
                half_extents: Vector::new(size.map(|value| value / 2.0)),
            }))
        }
        "cylinder" => Ok(Some(GeometryShape::Cylinder {
            radius: required(node, "radius")?,
            half_length: required(node, "length")? / 2.0,
        })),
        "sphere" => Ok(Some(GeometryShape::Sphere {
            radius: required(node, "radius")?,
        })),
        // Kept as written, `package://` and all. `RobotModel::mesh_path` resolves it.
        "mesh" => {
            let file = node
                .attribute("filename")
                .ok_or_else(|| bad_attribute(node, "filename", ""))?;
            let scale = parse_vector3(node, "scale")?.unwrap_or(ASSUMED_MESH_SCALE);
            Ok(Some(GeometryShape::Mesh {
                file: file.to_owned(),
                scale: Vector::new(scale),
            }))
        }
        _ => Ok(None),
    }
}

/// A shape's colour: its `<material>`'s own `<color>`, else the top-level material it names.
fn color(node: Node, materials: &HashMap<String, [f64; 4]>) -> Result<[f64; 4], ModelError> {
    let Some(material) = element(node, "material") else {
        return Ok(ASSUMED_RGBA);
    };
    if let Some(stated) = element(material, "color") {
        return read_rgba(stated);
    }
    Ok(material
        .attribute("name")
        .and_then(|name| materials.get(name).copied())
        .unwrap_or(ASSUMED_RGBA))
}

/// A `<color>`'s `rgba`, which it must state.
fn read_rgba(color: Node) -> Result<[f64; 4], ModelError> {
    parse_vector4(color, "rgba")?.ok_or_else(|| bad_attribute(color, "rgba", ""))
}

/// A required scalar attribute.
fn required(node: Node, attribute: &'static str) -> Result<f64, ModelError> {
    parse_scalar(node, attribute)?.ok_or_else(|| bad_attribute(node, attribute, ""))
}
