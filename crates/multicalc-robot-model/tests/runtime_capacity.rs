#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! The const generics of a tree are a capacity, not a model size: one
//! `KinematicTree<TREE_CAPACITY, CONFIGURATION_CAPACITY>` holds each model up to `TREE_CAPACITY`
//! bodies, and the spare slots change no result.

use std::path::{Path, PathBuf};

use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;
use multicalc_robot_model::{ModelError, RobotModel};

/// Body slots of the shared tree type.
const TREE_CAPACITY: usize = 32;
/// Configuration slots of the shared tree type: `TREE_CAPACITY` plus six for a floating base.
const CONFIGURATION_CAPACITY: usize = 38;

fn third_party(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party")
        .join(relative)
}

fn load(relative: &str) -> RobotModel {
    multicalc_robot_model::load_path(&third_party(relative)).unwrap()
}

fn skydio_x2() -> RobotModel {
    load("menagerie/skydio_x2/x2.xml")
}

fn franka_panda() -> RobotModel {
    load("menagerie/franka_emika_panda/panda.xml")
}

fn unitree_go1() -> RobotModel {
    load("menagerie/unitree_go1/go1.xml")
}

/// `step·(k + 1)` in slot `k < length`, zero past `length`. A floating base reads the identity
/// quaternion `[1, 0, 0, 0]` in slots 3–6.
fn ramp<const MAX_CONFIG: usize>(
    length: usize,
    step: f64,
    floating_base: bool,
) -> Vector<MAX_CONFIG, f64> {
    let mut values = Vector::from_fn(|index| {
        if index < length {
            step * (index + 1) as f64
        } else {
            0.0
        }
    });
    if floating_base {
        values.as_mut_slice()[3..7].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    }
    values
}

/// World pose of each body at the ramp configuration.
fn body_poses<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(
    model: &RobotModel,
) -> Vec<SE3<f64>> {
    let tree = model.kinematic_tree::<MAX_JOINTS, MAX_CONFIG>().unwrap();
    let configuration = ramp::<MAX_CONFIG>(tree.config_len(), 0.1, tree.has_floating_base());
    tree.forward_kinematics(&configuration)
        .unwrap()
        .poses()
        .to_vec()
}

/// RNEA torque at the ramp state, over the model's velocity slots. The spare slots stay zero.
fn joint_torques<const MAX_JOINTS: usize, const MAX_CONFIG: usize>(model: &RobotModel) -> Vec<f64> {
    let body = model
        .articulated_body::<MAX_JOINTS, MAX_CONFIG>(Vector::new([0.0, 0.0, -9.81]))
        .unwrap();
    let tree = body.tree();
    let configuration = ramp::<MAX_CONFIG>(tree.config_len(), 0.1, false);
    let velocity = ramp::<MAX_CONFIG>(tree.velocity_len(), 0.05, false);
    let acceleration = ramp::<MAX_CONFIG>(tree.velocity_len(), -0.02, false);
    let torque = body
        .inverse_dynamics_at(&configuration, &velocity, &acceleration)
        .unwrap();
    let (used, spare) = torque.as_slice().split_at(tree.velocity_len());
    assert!(spare.iter().all(|&entry| entry == 0.0));
    used.to_vec()
}

#[test]
fn spare_capacity_leaves_the_body_poses_unchanged() {
    let (drone, arm, quadruped) = (skydio_x2(), franka_panda(), unitree_go1());

    assert_eq!(
        body_poses::<TREE_CAPACITY, CONFIGURATION_CAPACITY>(&drone),
        body_poses::<1, 7>(&drone)
    );
    assert_eq!(
        body_poses::<TREE_CAPACITY, CONFIGURATION_CAPACITY>(&arm),
        body_poses::<11, 11>(&arm)
    );
    assert_eq!(
        body_poses::<TREE_CAPACITY, CONFIGURATION_CAPACITY>(&quadruped),
        body_poses::<13, 19>(&quadruped)
    );
}

#[test]
fn spare_capacity_leaves_the_joint_torques_unchanged() {
    // `ArticulatedBody` refuses a floating base, so the fixed-base arm carries this check.
    let arm = franka_panda();

    assert_eq!(
        joint_torques::<TREE_CAPACITY, CONFIGURATION_CAPACITY>(&arm),
        joint_torques::<11, 11>(&arm)
    );
}

#[test]
fn a_model_past_the_capacity_is_refused() {
    let arm = franka_panda();

    assert!(matches!(
        arm.kinematic_tree::<10, 10>(),
        Err(ModelError::TreeCapacityExceeded {
            needed: 11,
            capacity: 10
        })
    ));
}
