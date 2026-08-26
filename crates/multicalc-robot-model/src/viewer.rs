//! Rerun front end: logs a model's bodies and the shapes they draw.
//!
//! Bodies nest on the entity path, so Rerun composes each parent-relative pose down the tree and
//! the model stands at the pose the file states, every joint at its own reference value. No forward
//! kinematics, and no compile-time body count.

use std::collections::HashSet;
use std::fmt;
use std::path::PathBuf;

use multicalc::linear_algebra::Vector;

use crate::{GeometryShape, RobotModel, VisualGeometry};

/// What to draw and where meshes live.
#[derive(Debug, Clone)]
pub struct ViewerOptions {
    groups: Vec<u32>,
    package_paths: Vec<(String, PathBuf)>,
    frame_axis_length: f64,
}

impl ViewerOptions {
    /// Geom groups 0–2 shown, no package paths, 5 cm frame axes.
    #[must_use]
    pub fn new() -> Self {
        ViewerOptions {
            groups: vec![0, 1, 2],
            package_paths: Vec::new(),
            frame_axis_length: 0.05,
        }
    }

    /// The geom groups to draw.
    #[must_use]
    pub fn with_groups(mut self, groups: Vec<u32>) -> Self {
        self.groups = groups;
        self
    }

    /// Directory a `package://<name>/…` mesh reference resolves against.
    #[must_use]
    pub fn with_package_path(mut self, name: String, directory: PathBuf) -> Self {
        self.package_paths.push((name, directory));
        self
    }

    /// Frame gnomon length, in metres. 0 draws no frames.
    #[must_use]
    pub fn with_frame_axis_length(mut self, metres: f64) -> Self {
        self.frame_axis_length = metres;
        self
    }
}

impl Default for ViewerOptions {
    fn default() -> Self {
        ViewerOptions::new()
    }
}

/// What a log pass drew, and what it could not.
#[derive(Debug, Clone, Default)]
pub struct ViewerReport {
    shapes: usize,
    skipped_meshes: Vec<String>,
}

impl ViewerReport {
    /// Shapes logged, meshes included.
    #[inline]
    #[must_use]
    pub fn shapes(&self) -> usize {
        self.shapes
    }

    /// Mesh files whose path did not resolve or that could not be read, in the order met.
    ///
    /// A file the viewer cannot decode is not listed: decoding happens there, not here.
    #[inline]
    #[must_use]
    pub fn skipped_meshes(&self) -> &[String] {
        &self.skipped_meshes
    }
}

/// A Rerun SDK call failed.
#[derive(Debug)]
pub struct ViewerError(String);

impl fmt::Display for ViewerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "rerun error: {}", self.0)
    }
}

impl std::error::Error for ViewerError {}

/// Logs the whole model as a static scene: one entity per body, nested on the body tree.
pub fn log_model(
    stream: &rerun::RecordingStream,
    model: &RobotModel,
    options: &ViewerOptions,
) -> Result<ViewerReport, ViewerError> {
    let root = rerun::EntityPath::new(vec![rerun::EntityPathPart::new(model.name())]);
    log_static(stream, &root, &rerun::ViewCoordinates::RIGHT_HAND_Z_UP())?;

    let mut report = ViewerReport::default();
    let mut paths: Vec<rerun::EntityPath> = Vec::with_capacity(model.body_count());
    // Same-named siblings would collapse onto one entity, so a repeat takes an index suffix.
    let mut taken: HashSet<(Option<usize>, String)> = HashSet::new();

    // Bodies are in topological order, so a parent's path is always already built.
    for (index, body) in model.bodies().iter().enumerate() {
        let mut part = body.name().to_owned();
        if !taken.insert((body.parent(), part.clone())) {
            part = format!("{part}_{index}");
        }
        let parent = body.parent().map_or(&root, |parent| &paths[parent]);
        let path = child(parent, &part);

        log_static(stream, &path, &transform(body.pose()))?;
        if options.frame_axis_length > 0.0 {
            log_static(
                stream,
                &path,
                &rerun::TransformAxes3D::new(options.frame_axis_length as f32),
            )?;
        }

        let drawn: Vec<&VisualGeometry> = body
            .visual_geometry()
            .iter()
            .filter(|shape| options.groups.contains(&shape.group()))
            .collect();
        log_shapes(stream, model, options, &path, &drawn, &mut report)?;

        paths.push(path);
    }
    Ok(report)
}

/// One body's shapes: primitives batched one entity per kind, meshes one entity each.
fn log_shapes(
    stream: &rerun::RecordingStream,
    model: &RobotModel,
    options: &ViewerOptions,
    body: &rerun::EntityPath,
    shapes: &[&VisualGeometry],
    report: &mut ViewerReport,
) -> Result<(), ViewerError> {
    let mut boxes: Vec<(Placement, [f32; 3])> = Vec::new();
    let mut ellipsoids: Vec<(Placement, [f32; 3])> = Vec::new();
    let mut cylinders: Vec<Round> = Vec::new();
    let mut capsules: Vec<Round> = Vec::new();
    let mut meshes = 0usize;

    for shape in shapes {
        let placement = Placement::new(shape);
        match shape.shape() {
            GeometryShape::Sphere { radius } => {
                ellipsoids.push((placement, [*radius as f32; 3]));
            }
            GeometryShape::Ellipsoid { semi_axes } => {
                ellipsoids.push((placement, to_f32(semi_axes.into_array())));
            }
            GeometryShape::Box { half_extents } => {
                boxes.push((placement, to_f32(half_extents.into_array())));
            }
            GeometryShape::Cylinder {
                radius,
                half_length,
            } => cylinders.push(Round {
                placement,
                length: 2.0 * *half_length as f32,
                radius: *radius as f32,
            }),
            GeometryShape::Capsule {
                radius,
                half_length,
            } => {
                // Rerun runs a capsule from the entity origin to `(0, 0, length)`, so it is placed
                // by its lower cap centre: `centre - R·(0, 0, half_length)`.
                let offset = shape
                    .pose()
                    .rotation()
                    .act(Vector::new([0.0, 0.0, -half_length]));
                let start = shape.pose().translation() + offset;
                capsules.push(Round {
                    placement: Placement {
                        center: to_f32(start.into_array()),
                        ..placement
                    },
                    length: 2.0 * *half_length as f32,
                    radius: *radius as f32,
                });
            }
            GeometryShape::Mesh { file, scale } => {
                let path = child(body, &format!("mesh_{meshes}"));
                // Fails only for a file that cannot be read. The media type is guessed from the
                // extension and settled by the viewer, so a format it cannot decode fails there.
                let asset = model
                    .mesh_path(file, &options.package_paths)
                    .and_then(|resolved| rerun::Asset3D::from_file_path(&resolved).ok());
                let Some(asset) = asset else {
                    report.skipped_meshes.push(file.clone());
                    continue;
                };
                // An untextured mesh draws white without this.
                log_static(stream, &path, &asset.with_albedo_factor(placement.rgba))?;
                // `Asset3D` carries no scale of its own.
                log_static(
                    stream,
                    &path,
                    &rerun::Transform3D::from_translation_rotation_scale(
                        placement.center,
                        placement.rotation,
                        to_f32(scale.into_array()),
                    ),
                )?;
                meshes += 1;
                report.shapes += 1;
            }
        }
    }

    if !boxes.is_empty() {
        let archetype = rerun::Boxes3D::from_centers_and_half_sizes(
            boxes.iter().map(|(placement, _)| placement.center),
            boxes.iter().map(|(_, half_sizes)| *half_sizes),
        )
        .with_quaternions(boxes.iter().map(|(placement, _)| placement.rotation))
        .with_colors(boxes.iter().map(|(placement, _)| placement.color));
        log_static(stream, &child(body, "boxes"), &archetype)?;
        report.shapes += boxes.len();
    }

    if !ellipsoids.is_empty() {
        let archetype = rerun::Ellipsoids3D::from_centers_and_half_sizes(
            ellipsoids.iter().map(|(placement, _)| placement.center),
            ellipsoids.iter().map(|(_, half_sizes)| *half_sizes),
        )
        .with_quaternions(ellipsoids.iter().map(|(placement, _)| placement.rotation))
        .with_colors(ellipsoids.iter().map(|(placement, _)| placement.color));
        log_static(stream, &child(body, "ellipsoids"), &archetype)?;
        report.shapes += ellipsoids.len();
    }

    if !cylinders.is_empty() {
        let archetype = rerun::Cylinders3D::from_lengths_and_radii(
            cylinders.iter().map(|round| round.length),
            cylinders.iter().map(|round| round.radius),
        )
        .with_centers(cylinders.iter().map(|round| round.placement.center))
        .with_quaternions(cylinders.iter().map(|round| round.placement.rotation))
        .with_colors(cylinders.iter().map(|round| round.placement.color));
        log_static(stream, &child(body, "cylinders"), &archetype)?;
        report.shapes += cylinders.len();
    }

    if !capsules.is_empty() {
        let archetype = rerun::Capsules3D::from_lengths_and_radii(
            capsules.iter().map(|round| round.length),
            capsules.iter().map(|round| round.radius),
        )
        .with_translations(capsules.iter().map(|round| round.placement.center))
        .with_quaternions(capsules.iter().map(|round| round.placement.rotation))
        .with_colors(capsules.iter().map(|round| round.placement.color));
        log_static(stream, &child(body, "capsules"), &archetype)?;
        report.shapes += capsules.len();
    }

    Ok(())
}

/// Where a shape sits, how it is turned, and what colour it is drawn in.
///
/// A primitive takes a `Color` and a mesh an `AlbedoFactor`, so the bytes are kept alongside.
#[derive(Debug, Clone, Copy)]
struct Placement {
    center: [f32; 3],
    rotation: rerun::Quaternion,
    rgba: [u8; 4],
    color: rerun::Color,
}

impl Placement {
    fn new(shape: &VisualGeometry) -> Self {
        let rgba = shape.color().map(to_byte);
        let [red, green, blue, alpha] = rgba;
        Placement {
            center: to_f32(shape.pose().translation().into_array()),
            rotation: quaternion(shape.pose()),
            rgba,
            color: rerun::Color::from_unmultiplied_rgba(red, green, blue, alpha),
        }
    }
}

/// A cylinder or a capsule: a length along local z and a radius across it.
#[derive(Debug, Clone, Copy)]
struct Round {
    placement: Placement,
    length: f32,
    radius: f32,
}

/// `path` with one more part.
fn child(path: &rerun::EntityPath, part: &str) -> rerun::EntityPath {
    let mut parts = path.to_vec();
    parts.push(rerun::EntityPathPart::new(part));
    rerun::EntityPath::new(parts)
}

/// A pose as the transform Rerun composes down the entity path.
fn transform(pose: multicalc::spatial::SE3<f64>) -> rerun::Transform3D {
    rerun::Transform3D::from_translation_rotation(
        to_f32(pose.translation().into_array()),
        quaternion(pose),
    )
}

/// multicalc stores `[w, x, y, z]`, Rerun's `Quaternion` is xyzw. The one conversion point.
fn quaternion(pose: multicalc::spatial::SE3<f64>) -> rerun::Quaternion {
    let [w, x, y, z] = pose.rotation().quaternion().as_array();
    rerun::Quaternion::from_xyzw([x as f32, y as f32, z as f32, w as f32])
}

fn to_f32(values: [f64; 3]) -> [f32; 3] {
    values.map(|value| value as f32)
}

/// A colour channel in `0..=1` as a byte.
fn to_byte(component: f64) -> u8 {
    (component * 255.0).round().clamp(0.0, 255.0) as u8
}

/// One static scene, so nothing lands on a timeline.
fn log_static(
    stream: &rerun::RecordingStream,
    path: &rerun::EntityPath,
    archetype: &impl rerun::AsComponents,
) -> Result<(), ViewerError> {
    stream
        .log_static(path.clone(), archetype)
        .map_err(|err| ViewerError(err.to_string()))
}
