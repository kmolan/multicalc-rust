//! `<worldbody>` tree parsing.
//!
//! Spatial inertia comes from `<inertial>` where stated, otherwise it is integrated over the
//! body's geoms.

use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use multicalc::spatial::{SE3, SO3, SpatialInertia};
use roxmltree::{Document, Node};

use crate::mjcf::asset::AssetTable;
use crate::mjcf::compiler::{CompilerSettings, InertiaFromGeom};
use crate::mjcf::defaults::DefaultTable;
use crate::mjcf::geometry::{GeomMass, read_geom, read_visual_geom};
use crate::mjcf::joint::read_joint;
use crate::mjcf::orientation::Orientation;
use crate::xml::{
    bad_attribute, element, elements, ignored_sections, parse_scalar, parse_vector3, parse_vector6,
};
use crate::{JointDescription, ModelError, VisualGeometry};

/// Top-level elements this reader consumes. Every other one is listed in `ignored`.
const READ_SECTIONS: [&str; 4] = ["asset", "compiler", "default", "worldbody"];

/// One parsed file: bodies in document order, plus what was skipped.
pub(crate) struct ParsedModel {
    pub name: String,
    pub bodies: Vec<ParsedBody>,
    pub floating_base: bool,
    pub ignored: Vec<String>,
}

/// One `<body>` as parsed: topology, transform, inertia, joint.
pub(crate) struct ParsedBody {
    pub name: String,
    pub parent: Option<usize>,
    pub pose: SE3<f64>,
    /// Never empty here: MJCF always states or derives inertia. `Option` is for URDF's sake.
    pub inertia: Option<SpatialInertia<f64>>,
    pub joint: Option<JointDescription>,
    pub visual_geometry: Vec<VisualGeometry>,
}

/// Parses the body tree, rejecting anything outside the supported subset by name.
pub(crate) fn read(document: &Document) -> Result<ParsedModel, ModelError> {
    let root = document.root_element();
    let name = root.attribute("model").unwrap_or("model").to_owned();
    let settings = CompilerSettings::read(root)?;
    let table = DefaultTable::build(root)?;
    let assets = AssetTable::build(root, &table, &settings)?;
    let worldbody = element(root, "worldbody").ok_or(ModelError::MissingWorldbody)?;

    let mut bodies = Vec::new();
    let mut floating_base = false;
    for node in top_level_bodies(worldbody) {
        walk_body(
            node,
            None,
            None,
            &table,
            &settings,
            &assets,
            &mut bodies,
            &mut floating_base,
        )?;
    }

    if bodies.is_empty() {
        return Err(ModelError::NoBodies);
    }

    Ok(ParsedModel {
        name,
        bodies,
        floating_base,
        ignored: ignored_sections(root, &READ_SECTIONS),
    })
}

/// The `<body>` children of `<worldbody>`, seeing through a nested `<worldbody>` an `<include>`
/// spliced in: splicing reinserts the included `<mujoco>` verbatim, so its body list arrives
/// wrapped. Nested `<body>` elements are never wrapped, so deeper recursion is a plain child scan.
fn top_level_bodies<'doc, 'input>(worldbody: Node<'doc, 'input>) -> Vec<Node<'doc, 'input>> {
    let mut found = Vec::new();
    for child in worldbody.children().filter(Node::is_element) {
        match child.tag_name().name() {
            "body" => found.push(child),
            "worldbody" => found.extend(top_level_bodies(child)),
            _ => {}
        }
    }
    found
}

/// Parses one body, emits it, then recurses into its `<body>` children.
///
/// `inherited_class` is the nearest enclosing `childclass`, carried down until overridden.
#[expect(
    clippy::too_many_arguments,
    reason = "the recursion threads the class chain, both lookup tables, and the output"
)]
fn walk_body(
    node: Node,
    parent: Option<usize>,
    inherited_class: Option<&str>,
    table: &DefaultTable,
    settings: &CompilerSettings,
    assets: &AssetTable,
    bodies: &mut Vec<ParsedBody>,
    floating_base: &mut bool,
) -> Result<(), ModelError> {
    let name = node.attribute("name").unwrap_or("body").to_owned();

    let position = parse_vector3(node, "pos")?.unwrap_or([0.0; 3]);
    let turn = Orientation::read(node)?.resolve(node, settings)?;
    let pose = SE3::from_parts(SO3::from_quaternion(turn), Vector::new(position));

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
                return Err(ModelError::FreeJointNotAtRoot { body: name });
            }
            *floating_base = true;
            Some(JointDescription::floating(name.clone()))
        }
        _ => read_joint(node, table, class_chain, settings, &name)?,
    };

    let inertia = match settings.inertia_from_geom {
        InertiaFromGeom::Always => Some(synthesized_inertia(
            node,
            table,
            class_chain,
            settings,
            &name,
        )?),
        InertiaFromGeom::Never => {
            let inertial = element(node, "inertial")
                .ok_or_else(|| ModelError::NoInertiaSource { body: name.clone() })?;
            Some(stated_inertia(inertial, settings, &name)?)
        }
        InertiaFromGeom::Auto => match element(node, "inertial") {
            Some(inertial) => Some(stated_inertia(inertial, settings, &name)?),
            None => Some(synthesized_inertia(
                node,
                table,
                class_chain,
                settings,
                &name,
            )?),
        },
    };

    let mut visual_geometry = Vec::new();
    for geom in elements(node, "geom") {
        if let Some(shape) = read_visual_geom(geom, table, class_chain, assets, settings, &name)? {
            visual_geometry.push(shape);
        }
    }

    let index = bodies.len();
    bodies.push(ParsedBody {
        name,
        parent,
        pose,
        inertia,
        joint,
        visual_geometry,
    });

    for child in elements(node, "body") {
        walk_body(
            child,
            Some(index),
            class_chain,
            table,
            settings,
            assets,
            bodies,
            floating_base,
        )?;
    }
    Ok(())
}

/// Whether an element is a free joint: `<freejoint/>` or `<joint type="free">`.
fn is_free_joint(node: Node) -> bool {
    match node.tag_name().name() {
        "freejoint" => true,
        "joint" => node.attribute("type") == Some("free"),
        _ => false,
    }
}

/// Spatial inertia stated outright, via `diaginertia` or `fullinertia`.
fn stated_inertia(
    node: Node,
    settings: &CompilerSettings,
    body: &str,
) -> Result<SpatialInertia<f64>, ModelError> {
    let position = parse_vector3(node, "pos")?.unwrap_or([0.0; 3]);
    let mass = parse_scalar(node, "mass")?.ok_or_else(|| required(node, "mass"))?;

    let diagonal = parse_vector3(node, "diaginertia")?;
    let full = parse_vector6(node, "fullinertia")?;
    let orientation = Orientation::read(node)?;

    let tensor = match (diagonal, full) {
        (Some(principal), None) => {
            // Principal moments stand in a turned frame: `R I Rᵀ` puts them in body axes.
            let rotation = orientation.resolve(node, settings)?.to_rotation_matrix();
            rotation * Matrix::from_diagonal(principal) * rotation.transpose()
        }
        // A full tensor already stands in body axes, so a turn beside it names no frame. MuJoCo
        // refuses the pair rather than dropping one, and so does this.
        (None, Some([ixx, iyy, izz, ixy, ixz, iyz])) => {
            if orientation.is_stated() {
                return Err(ModelError::FullInertiaWithOrientation {
                    body: body.to_owned(),
                });
            }
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

    SpatialInertia::new(mass, Vector::new(position), tensor).map_err(ModelError::Inertia)
}

/// Spatial inertia integrated over a body's geoms, for a body stating none. Only the body's own
/// `<geom>` children count, never a descendant body's.
fn synthesized_inertia(
    body: Node,
    table: &DefaultTable,
    class_chain: Option<&str>,
    settings: &CompilerSettings,
    name: &str,
) -> Result<SpatialInertia<f64>, ModelError> {
    let mut shapes: Vec<GeomMass> = Vec::new();
    for geom in elements(body, "geom") {
        if let Some(shape) = read_geom(geom, table, class_chain, settings, name)? {
            shapes.push(shape);
        }
    }

    // Nothing to integrate. This is what stops a massless MJCF body from loading.
    let total_mass: f64 = shapes.iter().map(|shape| shape.mass).sum();
    if total_mass == 0.0 {
        return Err(ModelError::NoInertiaSource {
            body: name.to_owned(),
        });
    }

    // Composite COM: the mass-weighted mean of the geom centres.
    let weighted = shapes.iter().fold(Vector::zeros(), |running, shape| {
        running + shape.center.scale(shape.mass)
    });
    let center_of_mass = weighted.scale(1.0 / total_mass);

    // Parallel-axis shift of each geom's inertia from its own centre to the composite COM.
    let summed = shapes.iter().fold(Matrix::zeros(), |running, shape| {
        running + shape.inertia + shifted(shape.mass, shape.center - center_of_mass)
    });

    SpatialInertia::new(total_mass, center_of_mass, summed).map_err(ModelError::Inertia)
}

/// The parallel-axis term for a mass displaced by `offset` from its COM.
fn shifted(mass: f64, offset: Vector3D) -> Matrix3D {
    let distance_squared = offset.dot(offset);
    Matrix::from_fn(|row, column| {
        let along_diagonal = if row == column { distance_squared } else { 0.0 };
        mass * (along_diagonal - offset[row] * offset[column])
    })
}

/// Error for a required attribute the file omits.
#[must_use]
fn required(node: Node, attribute: &str) -> ModelError {
    bad_attribute(
        node,
        attribute,
        node.attribute(attribute).unwrap_or_default(),
    )
}
