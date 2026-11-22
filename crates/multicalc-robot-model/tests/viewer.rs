#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! The viewer against every vendored model, recorded to a file rather than streamed. No viewer
//! process is launched, so this runs in CI.

use std::path::{Path, PathBuf};

use multicalc_robot_model::RobotModel;
use multicalc_robot_model::viewer::{self, ViewerOptions, ViewerReport};
use rerun::RecordingStreamBuilder;

fn model(relative: &str) -> RobotModel {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party")
        .join(relative);
    multicalc_robot_model::load_path(&path)
        .unwrap_or_else(|err| panic!("loading {relative}: {err}"))
}

/// Logs a model into a recording named after the case, returning what was drawn.
///
/// The file is checked for size and removed, so nothing is left behind.
fn record(case: &str, model: &RobotModel, options: &ViewerOptions) -> ViewerReport {
    let path: PathBuf = std::env::temp_dir().join(format!("multicalc_model_viewer_{case}.rrd"));
    let stream = RecordingStreamBuilder::new("multicalc_model_viewer_test")
        .save(&path)
        .unwrap();
    let report = viewer::log_model(&stream, model, options).unwrap();
    stream.flush_blocking().unwrap();
    drop(stream);

    let written = std::fs::metadata(&path).unwrap_or_else(|err| panic!("{path:?}: {err}"));
    assert!(written.len() > 0, "{path:?} is empty");
    std::fs::remove_file(&path).unwrap();
    report
}

#[test]
fn every_menagerie_model_draws_its_shapes() {
    for (case, relative) in [
        ("panda", "menagerie/franka_emika_panda/panda.xml"),
        ("go1", "menagerie/unitree_go1/go1.xml"),
        ("x2", "menagerie/skydio_x2/x2.xml"),
    ] {
        let report = record(case, &model(relative), &ViewerOptions::new());
        assert!(report.shapes() > 0, "{case} drew nothing");
    }
}

#[test]
fn the_moveit_panda_draws_frames_alone() {
    // Eleven `<visual>` `package://` meshes, which resolve to nothing without a package path, and
    // eleven `<collision>` meshes in group 3, outside the default filter. Nothing here turns on the
    // mesh format: the files are simply not vendored.
    let report = record(
        "moveit_panda",
        &model("moveit_resources_panda/panda.urdf"),
        &ViewerOptions::new(),
    );
    assert_eq!(report.shapes(), 0);
    assert_eq!(report.skipped_meshes().len(), 11);
}

#[test]
fn the_group_filter_selects_what_is_drawn() {
    // Group-2 visual meshes against group-3 collision primitives. Neither count bounds the other:
    // the trunk alone carries eight collision geoms to one visual mesh.
    let go1 = model("menagerie/unitree_go1/go1.xml");
    let visual = record("go1_visual", &go1, &ViewerOptions::new());
    let collision = record(
        "go1_collision",
        &go1,
        &ViewerOptions::new().with_groups(vec![3]),
    );

    assert!(visual.shapes() > 0);
    assert!(collision.shapes() > 0);
    assert_ne!(visual.shapes(), collision.shapes());
}
