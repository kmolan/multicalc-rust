#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "urdf")]

//! The vendored MoveIt Panda, read as published. It covers two things no hand-written model here
//! does at full size: a gripper finger that follows the other one, and a description that states
//! no mass at all.

use std::path::Path;

use multicalc::kinematics::JointKind;
use multicalc::linear_algebra::Vector;
use multicalc_robot_model::{ModelError, ModelFormat, RobotModel};

fn panda() -> RobotModel {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/moveit_resources_panda/panda.urdf");
    multicalc_robot_model::urdf::load_path(&path).unwrap()
}

/// Every body in the file, in the order the links appear.
const BODIES: [&str; 12] = [
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_link8",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
];

#[test]
fn panda_reads_its_tree() {
    let model = panda();
    assert_eq!(model.format(), ModelFormat::Urdf);
    assert_eq!(model.name(), "panda");
    assert_eq!(model.body_count(), 12);

    let names: Vec<&str> = model.bodies().iter().map(|body| body.name()).collect();
    assert_eq!(names, BODIES);

    // A straight arm down to the hand, with both fingers hanging off the hand.
    let parents: Vec<Option<usize>> = model.bodies().iter().map(|body| body.parent()).collect();
    let want = [
        None,
        Some(0),
        Some(1),
        Some(2),
        Some(3),
        Some(4),
        Some(5),
        Some(6),
        Some(7),
        Some(8),
        Some(9),
        Some(9),
    ];
    assert_eq!(parents, want);
    assert!(!model.has_floating_base());
}

#[test]
fn panda_has_no_mass_anywhere() {
    // The file as published carries no `<inertial>` block on any link: it describes where the
    // robot's parts are and how they move, and nothing about what they weigh. This is exactly why
    // a body's mass properties are allowed to be absent.
    for body in panda().bodies() {
        assert_eq!(body.inertia(), None, "{}", body.name());
    }
}

#[test]
fn panda_joint_kinds() {
    let model = panda();
    let kind_of = |name: &str| {
        model
            .body_named(name)
            .unwrap()
            .joint()
            .map(|joint| joint.kind())
    };

    for index in 1..=7 {
        let link = format!("panda_link{index}");
        assert_eq!(kind_of(&link), Some(JointKind::Revolute), "{link}");
    }
    assert_eq!(kind_of("panda_leftfinger"), Some(JointKind::Prismatic));
    assert_eq!(kind_of("panda_rightfinger"), Some(JointKind::Prismatic));

    // `panda_joint8` and `panda_hand_joint` are welds, so the links they reach carry no joint.
    assert_eq!(kind_of("panda_link8"), None);
    assert_eq!(kind_of("panda_hand"), None);
    assert_eq!(model.movable_joint_count(), 9);
}

#[test]
fn panda_finger_mimics() {
    let model = panda();
    let left = model
        .body_named("panda_leftfinger")
        .unwrap()
        .joint()
        .unwrap();
    assert_eq!(left.name(), "panda_finger_joint1");
    assert_eq!(left.mimic(), None);

    let right = model
        .body_named("panda_rightfinger")
        .unwrap()
        .joint()
        .unwrap();
    let mimic = right.mimic().unwrap();
    assert_eq!(mimic.joint(), "panda_finger_joint1");
    // The file writes `<mimic joint="panda_finger_joint1" />` bare, so both take their defaults.
    assert_eq!(mimic.multiplier(), 1.0);
    assert_eq!(mimic.offset(), 0.0);
}

#[test]
fn panda_whole_tree_refuses() {
    assert_eq!(
        panda().kinematic_tree::<16, 16>().unwrap_err(),
        ModelError::MimicJointInTree {
            joint: "panda_finger_joint2".to_owned(),
            follows: "panda_finger_joint1".to_owned(),
        }
    );
}

#[test]
fn panda_arm_chain_builds() {
    // The chain down to the hand is the arm and the gripper mount, leaving out the finger that
    // follows the other one.
    let model = panda();
    let tree = model.kinematic_tree_to::<10, 10>("panda_hand").unwrap();
    assert_eq!(tree.len(), 10);

    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();
    assert_eq!(state.len(), 10);
    for slot in 0..10 {
        assert!(state.pose(slot).unwrap().translation().is_finite());
    }
}

#[test]
#[expect(
    clippy::approx_constant,
    reason = "-3.1416 is the number the file writes, not a rounded half turn"
)]
fn panda_safety_controller_is_not_read() {
    // The fourth joint states a soft range for a controller that sits inside its hard one. The
    // hard pair is what a model has to carry.
    let model = panda();
    let joint = model.body_named("panda_link4").unwrap().joint().unwrap();
    assert_eq!(joint.name(), "panda_joint4");
    assert_eq!(joint.limits(), Some((-3.1416, 0.0873)));
}

#[test]
fn panda_has_no_ignored_sections() {
    // The file carries only `<link>` and `<joint>` at the top level. Its meshes and its soft
    // limits sit inside those, so they are skipped without being named.
    assert!(panda().ignored().is_empty());
}
