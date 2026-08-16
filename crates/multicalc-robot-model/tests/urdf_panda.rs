#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(feature = "urdf")]

//! The vendored MoveIt Panda, as published: a mimic gripper joint and a wholly massless model, at
//! full size.

use std::path::Path;

use multicalc::kinematics::JointKind;
use multicalc::linear_algebra::Vector;
use multicalc_robot_model::{ModelError, ModelFormat, RobotModel};

fn panda() -> RobotModel {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/moveit_resources_panda/panda.urdf");
    multicalc_robot_model::urdf::load_path(&path).unwrap()
}

/// Bodies in document order.
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

    // A serial arm to the hand, with both fingers as children of the hand.
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
    // The published file has no `<inertial>` anywhere: kinematics only, no dynamics. This is why
    // spatial inertia is optional.
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

    // `panda_joint8` and `panda_hand_joint` are fixed, so their child bodies carry no joint.
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
    // Written bare as `<mimic joint="panda_finger_joint1" />`, so both take their defaults.
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
    // The chain to the hand is the arm plus the gripper mount, excluding the mimic finger.
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
    reason = "-3.1416 is the file's own literal, not an approximation of pi"
)]
fn panda_safety_controller_is_not_read() {
    // Joint 4's soft range sits inside its hard one; the hard pair is what is read.
    let model = panda();
    let joint = model.body_named("panda_link4").unwrap().joint().unwrap();
    assert_eq!(joint.name(), "panda_joint4");
    assert_eq!(joint.limits(), Some((-3.1416, 0.0873)));
}

#[test]
fn panda_has_no_ignored_sections() {
    // Only `<link>` and `<joint>` at the top level; meshes and soft limits sit inside those.
    assert!(panda().ignored().is_empty());
}
