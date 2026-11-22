#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! `<include>` resolution: only `load_path` follows it, relative to the including file's own
//! directory, and a chain of includes that never bottoms out is refused rather than hung on.

use std::path::{Path, PathBuf};

use multicalc_robot_model::ModelError;
use multicalc_robot_model::mjcf::load_path;

/// A fresh directory under the system temp directory, unique to this process and this case.
#[must_use]
fn scratch_dir(case: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "multicalc-robot-model-include-{}-{case}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("parts")).unwrap();
    dir
}

fn write(path: &Path, contents: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

#[test]
fn resolves_an_include_relative_to_the_including_file() {
    let dir = scratch_dir("simple");
    write(
        &dir.join("parts/arm.xml"),
        r#"<mujoco><worldbody><body name="arm"><inertial mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>"#,
    );
    write(
        &dir.join("robot.xml"),
        r#"<mujoco><worldbody><include file="parts/arm.xml"/></worldbody></mujoco>"#,
    );

    let model = load_path(&dir.join("robot.xml")).unwrap();
    assert_eq!(model.body_count(), 1);
    assert_eq!(model.body(0).unwrap().name(), "arm");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn follows_an_include_inside_an_included_file() {
    let dir = scratch_dir("nested");
    write(
        &dir.join("parts/hand.xml"),
        r#"<mujoco><worldbody><body name="hand"><inertial mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>"#,
    );
    write(
        &dir.join("parts/arm.xml"),
        r#"<mujoco><worldbody><body name="arm"><inertial mass="1" diaginertia="1 1 1"/></body><include file="hand.xml"/></worldbody></mujoco>"#,
    );
    write(
        &dir.join("robot.xml"),
        r#"<mujoco><worldbody><include file="parts/arm.xml"/></worldbody></mujoco>"#,
    );

    let model = load_path(&dir.join("robot.xml")).unwrap();
    assert_eq!(model.body_count(), 2);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn refuses_two_files_that_include_each_other() {
    let dir = scratch_dir("cycle");
    write(
        &dir.join("a.xml"),
        r#"<mujoco><worldbody><include file="b.xml"/></worldbody></mujoco>"#,
    );
    write(
        &dir.join("b.xml"),
        r#"<mujoco><worldbody><include file="a.xml"/></worldbody></mujoco>"#,
    );

    assert_eq!(
        load_path(&dir.join("a.xml")).unwrap_err(),
        ModelError::IncludeTooDeep { depth: 9 }
    );

    std::fs::remove_dir_all(&dir).unwrap();
}
