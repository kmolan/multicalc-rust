//! The Franka Panda's visual meshes, read from the Menagerie `panda.xml` assets.

use super::sink::Rgba;
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Body slots of the base-to-hand chain: `link0`, seven hinges, `hand`.
pub const FRANKA_PANDA_BODY_COUNT: usize = 9;

// panda.xml `<asset>` materials.
const MATERIAL_WHITE: Rgba = [255, 255, 255, 255];
const MATERIAL_OFF_WHITE: Rgba = [230, 235, 237, 255];
const MATERIAL_BLACK: Rgba = [64, 64, 64, 255];
const MATERIAL_GREEN: Rgba = [0, 255, 0, 255];
const MATERIAL_LIGHT_BLUE: Rgba = [10, 138, 199, 255];

/// One body's visual sub-meshes merged into one indexed triangle mesh.
#[derive(Debug, Clone, Default)]
pub struct BodyMesh {
    /// Vertex positions in the body frame.
    pub vertices: Vec<[f64; 3]>,
    /// One normal per vertex.
    pub normals: Vec<[f64; 3]>,
    /// Vertex index triples.
    pub triangles: Vec<[u32; 3]>,
    /// One material color per vertex.
    pub colors: Vec<Rgba>,
}

/// An error from reading a mesh file.
#[derive(Debug)]
pub enum MeshFileError {
    /// The file could not be read.
    IoError(std::io::Error),
    /// A line is not a `v`, `vn` or `v//vn` triangle face, at the given 1-based line number.
    Malformed {
        /// 1-based line number.
        line: usize,
    },
}

impl fmt::Display for MeshFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshFileError::IoError(error) => write!(f, "could not read the mesh file: {error}"),
            MeshFileError::Malformed { line } => write!(f, "mesh file line {line} is malformed"),
        }
    }
}

impl std::error::Error for MeshFileError {}

impl From<std::io::Error> for MeshFileError {
    fn from(error: std::io::Error) -> Self {
        MeshFileError::IoError(error)
    }
}

/// One body's visual sub-meshes: file stem (relative to the mesh directory, `.obj` implied) and
/// material color, from panda.xml's `<geom class="visual">` entries. No geom has its own
/// `pos`/`quat`. Empty past the last slot.
#[must_use]
pub fn body_meshes(slot: usize) -> &'static [(&'static str, Rgba)] {
    match slot {
        0 => &[
            ("link0_0", MATERIAL_OFF_WHITE),
            ("link0_1", MATERIAL_BLACK),
            ("link0_2", MATERIAL_OFF_WHITE),
            ("link0_3", MATERIAL_BLACK),
            ("link0_4", MATERIAL_OFF_WHITE),
            ("link0_5", MATERIAL_BLACK),
            ("link0_7", MATERIAL_WHITE),
            ("link0_8", MATERIAL_WHITE),
            ("link0_9", MATERIAL_BLACK),
            ("link0_10", MATERIAL_OFF_WHITE),
            ("link0_11", MATERIAL_WHITE),
        ],
        1 => &[("link1", MATERIAL_WHITE)],
        2 => &[("link2", MATERIAL_WHITE)],
        3 => &[
            ("link3_0", MATERIAL_WHITE),
            ("link3_1", MATERIAL_WHITE),
            ("link3_2", MATERIAL_WHITE),
            ("link3_3", MATERIAL_BLACK),
        ],
        4 => &[
            ("link4_0", MATERIAL_WHITE),
            ("link4_1", MATERIAL_WHITE),
            ("link4_2", MATERIAL_BLACK),
            ("link4_3", MATERIAL_WHITE),
        ],
        5 => &[
            ("link5_0", MATERIAL_BLACK),
            ("link5_1", MATERIAL_WHITE),
            ("link5_2", MATERIAL_WHITE),
        ],
        6 => &[
            ("link6_0", MATERIAL_OFF_WHITE),
            ("link6_1", MATERIAL_WHITE),
            ("link6_2", MATERIAL_BLACK),
            ("link6_3", MATERIAL_WHITE),
            ("link6_4", MATERIAL_WHITE),
            ("link6_5", MATERIAL_WHITE),
            ("link6_6", MATERIAL_WHITE),
            ("link6_7", MATERIAL_LIGHT_BLUE),
            ("link6_8", MATERIAL_LIGHT_BLUE),
            ("link6_9", MATERIAL_BLACK),
            ("link6_10", MATERIAL_BLACK),
            ("link6_11", MATERIAL_WHITE),
            ("link6_12", MATERIAL_GREEN),
            ("link6_13", MATERIAL_WHITE),
            ("link6_14", MATERIAL_BLACK),
            ("link6_15", MATERIAL_BLACK),
            ("link6_16", MATERIAL_WHITE),
        ],
        7 => &[
            ("link7_0", MATERIAL_WHITE),
            ("link7_1", MATERIAL_BLACK),
            ("link7_2", MATERIAL_BLACK),
            ("link7_3", MATERIAL_BLACK),
            ("link7_4", MATERIAL_BLACK),
            ("link7_5", MATERIAL_BLACK),
            ("link7_6", MATERIAL_BLACK),
            ("link7_7", MATERIAL_WHITE),
        ],
        8 => &[
            ("hand_0", MATERIAL_OFF_WHITE),
            ("hand_1", MATERIAL_BLACK),
            ("hand_2", MATERIAL_BLACK),
            ("hand_3", MATERIAL_WHITE),
            ("hand_4", MATERIAL_OFF_WHITE),
        ],
        _ => &[],
    }
}

/// Parses one `.obj` into an indexed triangle mesh, with every vertex colored `color`.
///
/// Faces are `v//vn` triangles; an output vertex is a (position, normal) index pair, so positions
/// can repeat. `mtllib`/`usemtl` and other lines are skipped.
pub fn read_obj(path: &Path, color: Rgba) -> Result<BodyMesh, MeshFileError> {
    let text = std::fs::read_to_string(path)?;

    let mut positions = Vec::new();
    let mut raw_normals = Vec::new();
    let mut mesh = BodyMesh::default();
    let mut seen: HashMap<(usize, usize), u32> = HashMap::new();

    for (index, line) in text.lines().enumerate() {
        let malformed = || MeshFileError::Malformed { line: index + 1 };
        let mut fields = line.split_ascii_whitespace();
        match fields.next() {
            Some(kind @ ("v" | "vn")) => {
                let coordinates: Vec<f64> = fields
                    .map(str::parse)
                    .collect::<Result<_, _>>()
                    .map_err(|_| malformed())?;
                let [x, y, z] = coordinates[..] else {
                    return Err(malformed());
                };
                if kind == "v" {
                    positions.push([x, y, z]);
                } else {
                    raw_normals.push([x, y, z]);
                }
            }
            Some("f") => {
                let mut triangle = Vec::with_capacity(3);
                for field in fields {
                    let mut parts = field.split('/');
                    let vertex_index = parse_index(parts.next()).ok_or_else(malformed)?;
                    let _texture_coordinate = parts.next();
                    let normal_index = parse_index(parts.next()).ok_or_else(malformed)?;
                    let key = (vertex_index, normal_index);
                    let output_index = match seen.get(&key) {
                        Some(&known) => known,
                        None => {
                            let position = *positions.get(vertex_index).ok_or_else(malformed)?;
                            let normal = *raw_normals.get(normal_index).ok_or_else(malformed)?;
                            mesh.vertices.push(position);
                            mesh.normals.push(normal);
                            mesh.colors.push(color);
                            let fresh = (mesh.vertices.len() - 1) as u32;
                            seen.insert(key, fresh);
                            fresh
                        }
                    };
                    triangle.push(output_index);
                }
                let [first, second, third] = triangle[..] else {
                    return Err(malformed());
                };
                mesh.triangles.push([first, second, third]);
            }
            _ => {}
        }
    }
    Ok(mesh)
}

/// A 1-based `.obj` index as a 0-based one.
fn parse_index(field: Option<&str>) -> Option<usize> {
    field?.parse::<usize>().ok()?.checked_sub(1)
}

/// Merges body `slot`'s visual sub-meshes from `mesh_dir`.
pub fn load_body_mesh(mesh_dir: &Path, slot: usize) -> Result<BodyMesh, MeshFileError> {
    let mut merged = BodyMesh::default();
    for &(file_stem, color) in body_meshes(slot) {
        let part = read_obj(&mesh_dir.join(format!("{file_stem}.obj")), color)?;
        let offset = merged.vertices.len() as u32;
        merged.vertices.extend(part.vertices);
        merged.normals.extend(part.normals);
        merged.colors.extend(part.colors);
        merged.triangles.extend(
            part.triangles
                .into_iter()
                .map(|[first, second, third]| [first + offset, second + offset, third + offset]),
        );
    }
    Ok(merged)
}
