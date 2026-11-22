//! `<asset>` mesh and material declarations.
//!
//! Mesh files are stored with `<compiler meshdir>` (or `assetdir`) already prepended, leaving a
//! path relative to the model file's own directory.

use std::collections::HashMap;
use std::path::Path;

use roxmltree::Node;

use crate::ModelError;
use crate::mjcf::compiler::CompilerSettings;
use crate::mjcf::defaults::DefaultTable;
use crate::xml::{elements, parse_vector3, parse_vector4};

/// MuJoCo's default material colour.
const ASSUMED_MATERIAL_RGBA: [f64; 4] = [1.0, 1.0, 1.0, 1.0];

/// A mesh with no `scale` anywhere on its class chain.
const ASSUMED_MESH_SCALE: [f64; 3] = [1.0, 1.0, 1.0];

/// One `<mesh>`: where its file sits and how far it is scaled.
pub(crate) struct MeshAsset {
    pub file: String,
    pub scale: [f64; 3],
}

/// Every `<mesh>` and `<material>`, by name.
pub(crate) struct AssetTable {
    meshes: HashMap<String, MeshAsset>,
    materials: HashMap<String, [f64; 4]>,
}

impl AssetTable {
    /// Reads every `<asset>` block. A `<mesh>` with no `file`, or a `<material>` with no `name`,
    /// is unreferenceable and skipped.
    pub(crate) fn build(
        root: Node,
        table: &DefaultTable,
        compiler: &CompilerSettings,
    ) -> Result<Self, ModelError> {
        let mut meshes = HashMap::new();
        let mut materials = HashMap::new();

        for asset in elements(root, "asset") {
            for node in elements(asset, "mesh") {
                let Some(file) = node.attribute("file") else {
                    continue;
                };
                // MuJoCo names an unnamed mesh after its file stem, as `skydio_x2` and
                // `unitree_go1` reference theirs.
                let name = match node.attribute("name") {
                    Some(stated) => stated.to_owned(),
                    None => Path::new(file)
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                };
                let scale = match parse_vector3(node, "scale")? {
                    Some(stated) => stated,
                    None => class_scale(node, table)?.unwrap_or(ASSUMED_MESH_SCALE),
                };
                let file = if compiler.mesh_directory.is_empty() {
                    file.to_owned()
                } else {
                    format!("{}/{file}", compiler.mesh_directory)
                };
                meshes.insert(name, MeshAsset { file, scale });
            }

            for node in elements(asset, "material") {
                let Some(name) = node.attribute("name") else {
                    continue;
                };
                let rgba = parse_vector4(node, "rgba")?.unwrap_or(ASSUMED_MATERIAL_RGBA);
                materials.insert(name.to_owned(), rgba);
            }
        }

        Ok(AssetTable { meshes, materials })
    }

    /// The mesh a geom's `mesh` attribute names.
    pub(crate) fn mesh(&self, name: &str) -> Option<&MeshAsset> {
        self.meshes.get(name)
    }

    /// The colour a geom's `material` attribute names.
    pub(crate) fn material(&self, name: &str) -> Option<[f64; 4]> {
        self.materials.get(name).copied()
    }
}

/// A mesh's scale off the default-class chain: the unnamed block, then the class it names.
fn class_scale(node: Node, table: &DefaultTable) -> Result<Option<[f64; 3]>, ModelError> {
    let mut settings = table.resolve(None)?.mesh.clone();
    if let Some(name) = node.attribute("class") {
        settings = settings.overridden_by(&table.resolve(Some(name))?.mesh);
    }
    Ok(settings.scale)
}
