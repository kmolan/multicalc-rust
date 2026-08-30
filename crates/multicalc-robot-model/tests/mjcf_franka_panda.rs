#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! The vendored Franka Emika Panda: body tree, per-joint settings, and where a default class
//! supplies what no body states. Numbers are hand-checked against the file, with no external
//! oracle; that comparison is `tools/qa/tests/mjcf.rs`.

use std::path::Path;

use multicalc::kinematics::JointKind;
use multicalc::linear_algebra::Vector;
use multicalc_robot_model::{GeometryShape, ModelError, RobotModel};

const BODY_NAMES: [&str; 11] = [
    "link0",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "hand",
    "left_finger",
    "right_finger",
];

#[must_use]
fn panda() -> RobotModel {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/menagerie/franka_emika_panda/panda.xml");
    multicalc_robot_model::mjcf::load_path(&path).unwrap()
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "{label}: {actual} is not {expected}"
    );
}

#[test]
fn reads_the_whole_arm() {
    let model = panda();

    assert_eq!(model.name(), "panda");
    assert_eq!(model.body_count(), 11);
    assert_eq!(model.movable_joint_count(), 9);
    assert!(!model.has_floating_base());

    for (index, name) in BODY_NAMES.into_iter().enumerate() {
        assert_eq!(model.body(index).unwrap().name(), name, "body {index}");
    }

    // link0 and hand are welds: a fixed base, and a frame carried by link7.
    assert!(model.body(0).unwrap().joint().is_none());
    assert!(model.body(8).unwrap().joint().is_none());
}

#[test]
fn the_seven_arm_joints_are_revolute_with_a_class_supplied_armature_and_damping() {
    let model = panda();

    for index in 1..=7 {
        let joint = model.body(index).unwrap().joint().unwrap();
        assert_eq!(joint.kind(), JointKind::Revolute, "body {index}");
        // Stated once, in a default class every link inherits. Read off the element alone both
        // would come back zero.
        assert_close(joint.armature(), 0.1, "armature");
        assert_close(joint.damping(), 1.0, "damping");
    }

    let limits = |index: usize| {
        model
            .body(index)
            .unwrap()
            .joint()
            .unwrap()
            .limits()
            .unwrap()
    };
    let (lower, upper) = limits(1);
    assert_close(lower, -2.8973, "joint 1 lower");
    assert_close(upper, 2.8973, "joint 1 upper");
    let (lower, upper) = limits(2);
    assert_close(lower, -1.7628, "joint 2 lower");
    assert_close(upper, 1.7628, "joint 2 upper");
    let (lower, upper) = limits(4);
    assert_close(lower, -3.0718, "joint 4 lower");
    assert_close(upper, -0.0698, "joint 4 upper");
    let (lower, upper) = limits(6);
    assert_close(lower, -0.0175, "joint 6 lower");
    assert_close(upper, 3.7525, "joint 6 upper");
}

#[test]
fn the_two_fingers_are_independent_sliding_joints() {
    let model = panda();

    for index in [9, 10] {
        let joint = model.body(index).unwrap().joint().unwrap();
        assert_eq!(joint.kind(), JointKind::Prismatic, "body {index}");
        assert_eq!(joint.limits(), Some((0.0, 0.04)), "body {index}");
    }
}

#[test]
fn reads_link1s_full_inertia_tensor() {
    let model = panda();
    let inertia = model.body_named("link1").unwrap().inertia().unwrap();

    assert_close(inertia.mass(), 4.970684, "link1 mass");
    assert_close(inertia.rotational_inertia()[(0, 1)], -0.000139, "link1 ixy");
}

#[test]
fn records_what_it_did_not_read() {
    let model = panda();
    let ignored: Vec<&str> = model.ignored().iter().map(String::as_str).collect();

    // `tendon` and `equality` carry the dropped coupling between the fingers: without them listed,
    // `movable_joint_count` alone would not say the fingers move independently rather than
    // mirrored.
    for section in [
        "tendon", "equality", "actuator", "keyframe", "contact", "option",
    ] {
        assert!(ignored.contains(&section), "missing {section}: {ignored:?}");
    }
}

#[test]
fn the_chain_down_to_the_hand_excludes_the_fingers() {
    let model = panda();
    assert_eq!(
        model.path_to("hand").unwrap(),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8]
    );

    let arm = model.kinematic_tree_to::<9, 9>("hand").unwrap();
    assert_eq!(arm.len(), 9);
    assert_eq!(arm.joint(8).unwrap().kind(), JointKind::Fixed);
    // The settings survive conversion, not just the parse.
    assert_close(arm.joint(1).unwrap().armature(), 0.1, "slot 1 armature");
}

#[test]
fn the_whole_model_fits_a_tree_of_eleven_but_not_ten() {
    let model = panda();

    assert!(model.kinematic_tree::<11, 11>().is_ok());
    assert_eq!(
        model.kinematic_tree::<10, 10>().unwrap_err(),
        ModelError::TreeCapacityExceeded {
            needed: 11,
            capacity: 10,
        }
    );
}

#[test]
fn articulated_body_carries_every_body_inertia() {
    let model = panda();
    let body = model
        .articulated_body::<16, 16>(Vector::new([0.0, 0.0, -9.81]))
        .unwrap();

    assert_eq!(body.len(), model.body_count());
    assert_eq!(model.body_count(), BODY_NAMES.len());
    for (index, name) in BODY_NAMES.into_iter().enumerate() {
        let stated = model.body(index).unwrap().inertia();
        match (stated, body.inertia(index)) {
            (Some(stated), Some(carried)) => {
                assert_close(carried.mass(), stated.mass(), name);
                assert!(
                    (carried.center_of_mass() - stated.center_of_mass()).norm() < 1e-12,
                    "{name} centre of mass"
                );
                let difference = carried.rotational_inertia() - stated.rotational_inertia();
                for row in 0..3 {
                    for column in 0..3 {
                        assert_close(
                            difference[(row, column)],
                            0.0,
                            &format!("{name} rotational inertia ({row}, {column})"),
                        );
                    }
                }
            }
            (None, None) => {}
            _ => panic!("{name} disagrees on whether it has mass"),
        }
    }
}

#[test]
fn link0_draws_its_visual_and_collision_meshes() {
    let model = panda();
    let shapes = model.body(0).unwrap().visual_geometry();
    assert_eq!(shapes.len(), 12);

    // `<default class="visual">` supplies type and group; the geom names mesh and material.
    assert_eq!(
        shapes[0].shape(),
        &GeometryShape::Mesh {
            file: "assets/link0_0.obj".to_owned(),
            scale: Vector::new([1.0, 1.0, 1.0]),
        }
    );
    assert_eq!(shapes[0].group(), 2);
    // `<material name="off_white" rgba="0.901961 0.921569 0.929412 1"/>`
    let color = shapes[0].color();
    for (component, expected) in color.iter().zip([0.901961, 0.921569, 0.929412, 1.0]) {
        assert!((component - expected).abs() < 1e-12, "{color:?}");
    }

    // The collision mesh is the twelfth, in group 3.
    assert_eq!(shapes[11].group(), 3);
    assert_eq!(
        shapes[11].shape(),
        &GeometryShape::Mesh {
            file: "assets/link0.stl".to_owned(),
            scale: Vector::new([1.0, 1.0, 1.0]),
        }
    );
}

#[test]
fn mesh_paths_resolve_against_the_model_directory() {
    let model = panda();
    let GeometryShape::Mesh { file, .. } = model.body(0).unwrap().visual_geometry()[0].shape()
    else {
        panic!("link0's first geom is a mesh");
    };
    let path = model.mesh_path(file, &[]).unwrap();
    assert!(path.is_file(), "{path:?}");
}

#[test]
fn geometry_ingestion_left_the_inertias_alone() {
    let model = panda();

    // link0 carries twelve geoms and link1 two, and inertia is read in the same walk that collects
    // them, so this pins the two paths as independent.
    let link0 = model.body_named("link0").unwrap();
    assert_close(link0.inertia().unwrap().mass(), 0.629769, "link0 mass");
    assert_eq!(link0.visual_geometry().len(), 12);

    let link1 = model.body_named("link1").unwrap();
    assert_close(link1.inertia().unwrap().mass(), 4.970684, "link1 mass");
    assert_eq!(link1.visual_geometry().len(), 2);
}
