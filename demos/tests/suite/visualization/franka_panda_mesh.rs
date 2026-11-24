use multicalc_demos::visualization::franka_panda_mesh::{
    FRANKA_PANDA_BODY_COUNT, MeshFileError, body_meshes, load_body_mesh, read_obj,
};
use std::path::Path;

const MESH_DIR: &str = "../third_party/menagerie/franka_emika_panda/assets";

// Every body merges into one well-formed mesh: one normal and one color per vertex, every index in
// range.
#[test]
fn every_body_mesh_loads_well_formed() {
    let mesh_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(MESH_DIR);
    for slot in 0..FRANKA_PANDA_BODY_COUNT {
        let mesh = load_body_mesh(&mesh_dir, slot).unwrap();
        assert!(!mesh.triangles.is_empty(), "slot {slot} has no triangles");
        assert_eq!(mesh.normals.len(), mesh.vertices.len());
        assert_eq!(mesh.colors.len(), mesh.vertices.len());
        let vertex_count = mesh.vertices.len() as u32;
        assert!(
            mesh.triangles
                .iter()
                .flatten()
                .all(|&index| index < vertex_count),
            "slot {slot} has an index out of range"
        );
    }
    assert!(body_meshes(FRANKA_PANDA_BODY_COUNT).is_empty());
}

// A face that names a vertex the file never defined is rejected with its line number.
#[test]
fn face_past_the_vertices_is_malformed() {
    let path = std::env::temp_dir().join("multicalc_demos_malformed_mesh.obj");
    std::fs::write(&path, "v 0 0 0\nvn 0 0 1\nf 1//1 2//1 3//1\n").unwrap();
    let result = read_obj(&path, [0, 0, 0, 255]);
    let _ = std::fs::remove_file(&path);
    assert!(matches!(result, Err(MeshFileError::Malformed { line: 3 })));
}
