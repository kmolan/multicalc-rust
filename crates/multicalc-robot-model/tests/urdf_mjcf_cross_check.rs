#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! MoveIt Panda URDF against Menagerie Panda MJCF: the same arm from two independent
//! descriptions, not conversions of each other. Measured agreement:
//!
//! - Arm joints 1-7: identical order, kind and axis.
//! - Link frames 0-7 at q = 0: 2e-12 in translation, 5e-12 in rotation.
//! - Gripper mount translation: 7e-13.
//! - Gripper mount rotation: 1.6e-8. The MJCF rounds the quaternion to seven digits
//!   (`0.9238795 0 0 -0.3826834`); the URDF gives the same pi/4 as an rpy angle to twelve.
//! - Topology differs: the URDF has a `panda_link8` frame between link7 and the hand, so the chain
//!   is 10 bodies against the MJCF's 9.
//! - Finger 2 differs: the URDF mirrors the axis and mimics finger 1; the MJCF gives both fingers
//!   the same axis and no coupling. Both are outside the chain compared here.

use std::path::{Path, PathBuf};

use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;
use multicalc_robot_model::RobotModel;

/// Agreement required on the arm frames.
const ARM_TOLERANCE: f64 = 1e-11;
/// Agreement required on the mount's rotation, set by the MJCF's rounded quaternion literal.
const MOUNT_TURN_TOLERANCE: f64 = 1e-7;

fn third_party(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party")
        .join(relative)
}

fn urdf_panda() -> RobotModel {
    multicalc_robot_model::urdf::load_path(&third_party("moveit_resources_panda/panda.urdf"))
        .unwrap()
}

fn mjcf_panda() -> RobotModel {
    multicalc_robot_model::mjcf::load_path(&third_party("menagerie/franka_emika_panda/panda.xml"))
        .unwrap()
}

/// Largest per-axis translation difference.
fn position_gap(left: SE3<f64>, right: SE3<f64>) -> f64 {
    (0..3)
        .map(|axis| (left.translation()[axis] - right.translation()[axis]).abs())
        .fold(0.0, f64::max)
}

/// Largest per-component quaternion difference, sign-aligned since q and -q are the same
/// rotation.
fn turn_gap(left: SE3<f64>, right: SE3<f64>) -> f64 {
    let (a, b) = (left.rotation().quaternion(), right.rotation().quaternion());
    let same_way = a.w() * b.w() + a.x() * b.x() + a.y() * b.y() + a.z() * b.z() >= 0.0;
    let sign = if same_way { 1.0 } else { -1.0 };
    [
        (sign * a.w() - b.w()).abs(),
        (sign * a.x() - b.x()).abs(),
        (sign * a.y() - b.y()).abs(),
        (sign * a.z() - b.z()).abs(),
    ]
    .into_iter()
    .fold(0.0, f64::max)
}

#[test]
fn the_two_files_agree_on_the_arm_joints() {
    let urdf = urdf_panda();
    let mjcf = mjcf_panda();

    for index in 1..=7 {
        let from_urdf = urdf
            .body_named(&format!("panda_link{index}"))
            .unwrap()
            .joint()
            .unwrap();
        let from_mjcf = mjcf
            .body_named(&format!("link{index}"))
            .unwrap()
            .joint()
            .unwrap();

        assert_eq!(from_urdf.kind(), from_mjcf.kind(), "joint {index} kind");
        assert_eq!(
            from_urdf.axis().into_array(),
            from_mjcf.axis().into_array(),
            "joint {index} axis"
        );
    }
}

#[test]
fn the_two_files_put_the_arm_in_the_same_place() {
    let urdf_chain = urdf_panda()
        .kinematic_tree_to::<10, 10>("panda_hand")
        .unwrap();
    let mjcf_chain = mjcf_panda().kinematic_tree_to::<10, 10>("hand").unwrap();

    // The URDF's extra `panda_link8` frame makes its chain one body longer.
    assert_eq!(urdf_chain.len(), 10);
    assert_eq!(mjcf_chain.len(), 9);

    let from_urdf = urdf_chain.forward_kinematics(&Vector::zeros()).unwrap();
    let from_mjcf = mjcf_chain.forward_kinematics(&Vector::zeros()).unwrap();

    // Slots 0-7 are link0-link7 in both.
    for slot in 0..=7 {
        let (left, right) = (from_urdf.pose(slot).unwrap(), from_mjcf.pose(slot).unwrap());
        assert!(
            position_gap(left, right) < ARM_TOLERANCE,
            "link {slot} sits {} apart",
            position_gap(left, right)
        );
        assert!(
            turn_gap(left, right) < ARM_TOLERANCE,
            "link {slot} is turned {} apart",
            turn_gap(left, right)
        );
    }

    let urdf_mount = from_urdf.pose(urdf_chain.len() - 1).unwrap();
    let mjcf_mount = from_mjcf.pose(mjcf_chain.len() - 1).unwrap();
    assert!(
        position_gap(urdf_mount, mjcf_mount) < ARM_TOLERANCE,
        "the gripper mount sits {} apart",
        position_gap(urdf_mount, mjcf_mount)
    );
    assert!(
        turn_gap(urdf_mount, mjcf_mount) < MOUNT_TURN_TOLERANCE,
        "the gripper mount is turned {} apart",
        turn_gap(urdf_mount, mjcf_mount)
    );
}

#[test]
fn the_two_files_disagree_about_the_second_finger() {
    // Recorded, not smoothed over: the URDF mirrors the axis and couples the joints, the MJCF
    // uses one axis for both and no coupling. Equivalent jaw motion, different encoding.
    let urdf = urdf_panda();
    let mjcf = mjcf_panda();

    let urdf_second = urdf
        .body_named("panda_rightfinger")
        .unwrap()
        .joint()
        .unwrap();
    let mjcf_second = mjcf.body_named("right_finger").unwrap().joint().unwrap();

    assert_eq!(urdf_second.axis().into_array(), [0.0, -1.0, 0.0]);
    assert_eq!(mjcf_second.axis().into_array(), [0.0, 1.0, 0.0]);
    assert_eq!(urdf_second.mimic().unwrap().joint(), "panda_finger_joint1");
    assert_eq!(mjcf_second.mimic(), None);
}
