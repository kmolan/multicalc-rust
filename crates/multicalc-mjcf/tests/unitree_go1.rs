#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Loads the vendored Unitree Go1 and checks the structure the parse must produce: a floating base
//! (the trunk, on a free joint) carrying twelve hinge joints across four branching legs, with the
//! per-joint dynamics settings a two-level default class supplies. Numbers here are hand-checked
//! against the model file directly, with no external oracle involved — that comparison is
//! `tools/qa/tests/{kinematics,inverse_kinematics,mjcf}.rs`.

use std::path::Path;

use multicalc::kinematics::JointKind;
use multicalc_mjcf::RobotModel;

const BODY_NAMES: [&str; 13] = [
    "trunk",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
];

/// Every body's parent, by index — the trunk at the root and four independent three-body legs
/// hanging off it.
const PARENTS: [Option<usize>; 13] = [
    None,
    Some(0),
    Some(1),
    Some(2),
    Some(0),
    Some(4),
    Some(5),
    Some(0),
    Some(7),
    Some(8),
    Some(0),
    Some(10),
    Some(11),
];

#[must_use]
fn go1() -> RobotModel {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/menagerie/unitree_go1/go1.xml");
    multicalc_mjcf::load_path(&path).unwrap()
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "{label}: {actual} is not {expected}"
    );
}

#[test]
fn reads_the_whole_body_and_leg_tree() {
    let model = go1();

    assert_eq!(model.name(), "go1");
    assert_eq!(model.body_count(), 13);
    assert_eq!(model.movable_joint_count(), 13);
    assert!(model.has_floating_base());

    for (index, name) in BODY_NAMES.into_iter().enumerate() {
        assert_eq!(model.body(index).unwrap().name(), name, "body {index}");
    }
    for (index, parent) in PARENTS.into_iter().enumerate() {
        assert_eq!(model.body(index).unwrap().parent(), parent, "body {index}");
    }
}

#[test]
fn the_trunk_carries_a_floating_joint() {
    let model = go1();
    let joint = model.body(0).unwrap().joint().unwrap();
    assert_eq!(joint.kind(), JointKind::Floating);
}

#[test]
fn the_twelve_leg_joints_are_revolute_with_class_supplied_dynamics() {
    let model = go1();

    for index in 1..13 {
        let joint = model.body(index).unwrap().joint().unwrap();
        assert_eq!(joint.kind(), JointKind::Revolute, "body {index}");
        // Every leg joint states `frictionloss` only once, in the `go1` class every joint
        // inherits — a reader that took it from the element alone would find nothing here.
        assert_close(joint.friction_loss(), 0.2, "body {index} friction_loss");
        assert_close(joint.armature(), 0.01, "body {index} armature");
    }

    // FR_hip_joint: the `abduction` class, nested one level under `go1`, overrides the axis,
    // damping, and range every abduction joint on the robot shares.
    let hip = model.body(1).unwrap().joint().unwrap();
    assert_eq!(hip.axis().into_array(), [1.0, 0.0, 0.0]);
    assert_close(hip.damping(), 1.0, "FR_hip_joint damping");
    assert_eq!(hip.limits(), Some((-0.863, 0.863)));

    // FR_thigh_joint: the `hip` class overrides only the range, so the axis and damping fall
    // through to the `go1` class underneath it.
    let thigh = model.body(2).unwrap().joint().unwrap();
    assert_eq!(thigh.axis().into_array(), [0.0, 1.0, 0.0]);
    assert_close(thigh.damping(), 2.0, "FR_thigh_joint damping");
    assert_eq!(thigh.limits(), Some((-0.686, 4.501)));

    // FR_calf_joint: the `knee` class, same inheritance shape as `hip`.
    let calf = model.body(3).unwrap().joint().unwrap();
    assert_eq!(calf.axis().into_array(), [0.0, 1.0, 0.0]);
    assert_close(calf.damping(), 2.0, "FR_calf_joint damping");
    assert_eq!(calf.limits(), Some((-2.818, -0.888)));
}

#[test]
fn reads_the_trunks_full_inertia_tensor() {
    let model = go1();
    let inertia = model.body_named("trunk").unwrap().inertia();

    assert_close(inertia.mass(), 5.204, "trunk mass");
}

#[test]
fn records_what_it_did_not_read() {
    let model = go1();
    let ignored: Vec<&str> = model.ignored().iter().map(String::as_str).collect();

    for section in ["actuator", "asset", "keyframe", "option"] {
        assert!(ignored.contains(&section), "missing {section}: {ignored:?}");
    }
}

#[test]
fn the_whole_model_builds_a_tree_whose_floating_base_widens_the_configuration() {
    let model = go1();

    // One joint slot per body, but the floating base takes seven configuration numbers where
    // every other joint takes one: 12 hinges + 7 = 19.
    let tree = model.kinematic_tree::<13, 19>().unwrap();
    assert_eq!(tree.len(), 13);
    assert_eq!(tree.joint(0).unwrap().kind(), JointKind::Floating);
    assert_eq!(tree.config_len(), 19);
    assert_eq!(tree.velocity_len(), 18);

    // A capacity sized for the joint count alone, with no room for the floating base's extra
    // six slots, is refused.
    assert!(model.kinematic_tree::<13, 13>().is_err());
}
