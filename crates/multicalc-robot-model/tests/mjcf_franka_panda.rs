#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Loads the vendored Franka Emika Panda and checks the structure the parse must produce: the
//! body tree, which joints carry which settings, and where a default class supplies data no body
//! states for itself. Numbers here are hand-checked against the model file directly, with no
//! external oracle involved — that comparison is `tools/qa/tests/mjcf.rs`.

use std::path::Path;

use multicalc::kinematics::JointKind;
use multicalc_robot_model::{MjcfError, RobotModel};

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
    multicalc_robot_model::load_path(&path).unwrap()
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

    // link0 and hand are welds: the base does not move, and the hand is a frame carried by link7.
    assert!(model.body(0).unwrap().joint().is_none());
    assert!(model.body(8).unwrap().joint().is_none());
}

#[test]
fn the_seven_arm_joints_are_revolute_with_a_class_supplied_armature_and_damping() {
    let model = panda();

    for index in 1..=7 {
        let joint = model.body(index).unwrap().joint().unwrap();
        assert_eq!(joint.kind(), JointKind::Revolute, "body {index}");
        // The Franka states these once, in a default class every link inherits — a reader that
        // took them from the element alone would find nothing and record zero for both.
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
    let inertia = model.body_named("link1").unwrap().inertia();

    assert_close(inertia.mass(), 4.970684, "link1 mass");
    assert_close(inertia.rotational_inertia()[(0, 1)], -0.000139, "link1 ixy");
}

#[test]
fn records_what_it_did_not_read() {
    let model = panda();
    let ignored: Vec<&str> = model.ignored().iter().map(String::as_str).collect();

    // The dropped coupling between the two fingers is what makes `tendon` and `equality` worth
    // seeing here: a caller who only checked `movable_joint_count` would not otherwise know they
    // move independently rather than mirrored.
    for section in [
        "tendon", "equality", "actuator", "asset", "keyframe", "contact", "option",
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
    // The data survived the conversion into a `KinematicTree`, not just the parse.
    assert_close(arm.joint(1).unwrap().armature(), 0.1, "slot 1 armature");
}

#[test]
fn the_whole_model_fits_a_tree_of_eleven_but_not_ten() {
    let model = panda();

    assert!(model.kinematic_tree::<11, 11>().is_ok());
    assert_eq!(
        model.kinematic_tree::<10, 10>().unwrap_err(),
        MjcfError::TreeCapacityExceeded {
            needed: 11,
            capacity: 10,
        }
    );
}
