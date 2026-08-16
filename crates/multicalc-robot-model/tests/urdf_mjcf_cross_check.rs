#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![cfg(all(feature = "mjcf", feature = "urdf"))]

//! The same robot read from both formats: the vendored MoveIt Panda URDF against the vendored
//! Menagerie Panda MJCF.
//!
//! The two files are separate descriptions of the same arm rather than conversions of each other,
//! so they are not expected to match everywhere. What they do and do not agree on, measured:
//!
//! - **The seven arm joints agree exactly.** Same order, same kind, same axis.
//! - **The eight arm link frames agree to about a millionth of a millimetre** at the pose where
//!   every joint reads zero — 2e-12 in position, 5e-12 in turn.
//! - **The gripper mount sits in the same place**, to 1e-12.
//! - **The gripper mount's turn agrees only to about 2e-8.** The MJCF writes it as a quaternion
//!   rounded to seven digits, `0.9238795 0 0 -0.3826834`, where the URDF writes the eighth of a
//!   turn it stands for as an angle to twelve. The gap is the MJCF file's own rounding.
//! - **The two files disagree about the tree.** The URDF carries a bare `panda_link8` frame
//!   between the last arm link and the hand, which the MJCF does not, so the same chain is ten
//!   bodies in one file and nine in the other.
//! - **The two files disagree about the second finger.** The URDF mirrors its axis and has it
//!   follow the first finger; the MJCF gives both fingers the same axis and no such link. The
//!   fingers are outside the chain compared here.

use std::path::{Path, PathBuf};

use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;
use multicalc_robot_model::RobotModel;

/// How far apart the two files' arm frames are allowed to sit.
const ARM_TOLERANCE: f64 = 1e-11;
/// How far apart the gripper mount's turn is allowed to be, which is set by the MJCF's own
/// rounding of the quaternion it writes rather than by anything either reader does.
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

/// The largest gap between two poses' positions.
fn position_gap(left: SE3<f64>, right: SE3<f64>) -> f64 {
    (0..3)
        .map(|axis| (left.translation()[axis] - right.translation()[axis]).abs())
        .fold(0.0, f64::max)
}

/// The largest gap between two poses' turns, comparing them the same way round — a quaternion and
/// its negative name the same turn.
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

    // Slots 0 to 7 are `link0` through `link7` in both files.
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
    // Recorded rather than smoothed over: the URDF closes the gripper by mirroring the second
    // finger's axis and having it follow the first, while the MJCF gives both fingers the same
    // axis and no such link. Either describes the same jaws closing.
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
