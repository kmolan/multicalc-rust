//! Kinematic-tree tests: model validation, per-joint-kind transforms, branching topology, slot
//! bookkeeping, and autodiff/f32 coverage.

use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use multicalc::error::KinematicsError;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::{SE3, SO3};

const TOL: f64 = 1e-12;

// ---- helpers ----------------------------------------------------------------

fn axis_z<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ZERO, T::ZERO, T::ONE])
}

fn axis_x<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ONE, T::ZERO, T::ZERO])
}

fn translation<T: Numeric>(x: T, y: T, z: T) -> SE3<T> {
    SE3::from_parts(SO3::identity(), Vector::new([x, y, z]))
}

/// Two revolute joints about z on unit links, plus a fixed tool one unit past the elbow.
fn planar_arm<T: Numeric>() -> KinematicTree<3, T> {
    let link = translation(T::ONE, T::ZERO, T::ZERO);
    KinematicTree::try_from_joints(
        &[
            Joint::revolute(axis_z(), SE3::identity()),
            Joint::revolute(axis_z(), link),
            Joint::fixed(link),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
        ],
    )
    .unwrap()
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < TOL, "{what}: got {got}, want {want}");
}

fn assert_vector_close(got: Vector3D<f64>, want: [f64; 3], what: &str) {
    let got = *got.as_array();
    for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(*got, *want, &format!("{what} component {index}"));
    }
}

// ---- model validation -------------------------------------------------------

#[test]
fn rejects_more_joints_than_capacity() {
    let joint = Joint::revolute(axis_z::<f64>(), SE3::identity());
    let built = KinematicTree::<2, f64>::try_from_joints(
        &[joint, joint, joint],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
        ],
    );
    assert_eq!(built, Err(KinematicsError::CapacityExceeded));
}

#[test]
fn rejects_mismatched_parent_count() {
    let joint = Joint::revolute(axis_z::<f64>(), SE3::identity());
    let built = KinematicTree::<4, f64>::try_from_joints(&[joint, joint], &[JointParent::World]);
    assert_eq!(built, Err(KinematicsError::JointCountMismatch));
}

#[test]
fn rejects_a_parent_that_is_not_earlier() {
    let joint = Joint::revolute(axis_z::<f64>(), SE3::identity());

    // A joint attached to itself.
    let attached_to_itself = KinematicTree::<4, f64>::try_from_joints(
        &[joint, joint],
        &[JointParent::Joint(0), JointParent::Joint(0)],
    );
    assert_eq!(attached_to_itself, Err(KinematicsError::ParentOutOfOrder));

    // A joint attached to one that comes after it.
    let attached_to_a_later_joint = KinematicTree::<4, f64>::try_from_joints(
        &[joint, joint],
        &[JointParent::Joint(1), JointParent::World],
    );
    assert_eq!(
        attached_to_a_later_joint,
        Err(KinematicsError::ParentOutOfOrder)
    );
}

#[test]
fn rejects_a_zero_axis() {
    let mut tree = KinematicTree::<4, f64>::new();
    assert_eq!(
        tree.push(
            Joint::revolute(Vector::zeros(), SE3::identity()),
            JointParent::World
        ),
        Err(KinematicsError::AxisHasNoDirection)
    );

    // A weld carries no axis of its own, so it is accepted.
    assert_eq!(
        tree.push(Joint::fixed(SE3::identity()), JointParent::World),
        Ok(())
    );
    assert_eq!(tree.len(), 1);
}

#[test]
fn rejects_reversed_limits() {
    let mut tree = KinematicTree::<4, f64>::new();
    let joint = Joint::revolute(axis_z::<f64>(), SE3::identity()).with_limits(1.0, -1.0);
    assert_eq!(
        tree.push(joint, JointParent::World),
        Err(KinematicsError::LimitsReversed)
    );
}

#[test]
fn rejects_non_finite_model_values() {
    let mut tree = KinematicTree::<4, f64>::new();

    let bad_offset = Joint::revolute(axis_z::<f64>(), SE3::identity()).with_zero_offset(f64::NAN);
    assert_eq!(
        tree.push(bad_offset, JointParent::World),
        Err(KinematicsError::NonFinite)
    );

    let bad_origin = Joint::revolute(axis_z::<f64>(), translation(f64::INFINITY, 0.0, 0.0));
    assert_eq!(
        tree.push(bad_origin, JointParent::World),
        Err(KinematicsError::NonFinite)
    );
}

#[test]
fn normalizes_a_non_unit_axis() {
    let mut tree = KinematicTree::<4, f64>::new();
    let joint = Joint::revolute(Vector::new([0.0, 0.0, 2.0]), SE3::identity());
    tree.push(joint, JointParent::World).unwrap();

    assert_vector_close(
        tree.joint(0).unwrap().axis(),
        [0.0, 0.0, 1.0],
        "stored axis",
    );
}

// ---- per-joint-kind transforms ----------------------------------------------

#[test]
fn revolute_joint_turns_about_its_axis() {
    let mut tree = KinematicTree::<1, f64>::new();
    tree.push(
        Joint::revolute(axis_z::<f64>(), SE3::identity()),
        JointParent::World,
    )
    .unwrap();

    let state = tree.forward_kinematics(&Vector::new([FRAC_PI_2])).unwrap();
    let pose = state.pose(0).unwrap();

    assert_vector_close(
        pose.rotation().act(axis_x()),
        [0.0, 1.0, 0.0],
        "rotated x axis",
    );
    assert_vector_close(pose.translation(), [0.0, 0.0, 0.0], "translation");
}

#[test]
fn revolute_joint_turns_about_its_anchor() {
    let mut tree = KinematicTree::<1, f64>::new();
    let joint =
        Joint::revolute(axis_z::<f64>(), SE3::identity()).with_anchor(Vector::new([1.0, 0.0, 0.0]));
    tree.push(joint, JointParent::World).unwrap();

    // Half a turn about the point one unit out on x carries the origin to two units out.
    let state = tree.forward_kinematics(&Vector::new([PI])).unwrap();
    let moved = state.pose(0).unwrap().act(Vector::zeros());

    assert_vector_close(moved, [2.0, 0.0, 0.0], "origin after half a turn");
}

#[test]
fn prismatic_joint_slides_along_its_axis() {
    let mut tree = KinematicTree::<1, f64>::new();
    tree.push(
        Joint::prismatic(axis_x::<f64>(), SE3::identity()),
        JointParent::World,
    )
    .unwrap();

    let state = tree.forward_kinematics(&Vector::new([0.75])).unwrap();
    let pose = state.pose(0).unwrap();

    assert_vector_close(pose.translation(), [0.75, 0.0, 0.0], "translation");
    assert_vector_close(
        pose.rotation().act(axis_x()),
        [1.0, 0.0, 0.0],
        "rotated x axis",
    );
}

#[test]
fn fixed_joint_ignores_its_reading() {
    let mut tree = KinematicTree::<1, f64>::new();
    let origin = translation(0.3, -0.2, 0.1);
    tree.push(Joint::fixed(origin), JointParent::World).unwrap();

    let at_zero = tree.forward_kinematics(&Vector::new([0.0])).unwrap();
    let at_five = tree.forward_kinematics(&Vector::new([5.0])).unwrap();

    assert_eq!(at_zero, at_five);
    assert_vector_close(
        at_zero.pose(0).unwrap().translation(),
        *origin.translation().as_array(),
        "welded pose",
    );
}

#[test]
fn zero_offset_shifts_where_zero_is() {
    let mut tree = KinematicTree::<1, f64>::new();
    let joint = Joint::revolute(axis_z::<f64>(), SE3::identity()).with_zero_offset(FRAC_PI_4);
    tree.push(joint, JointParent::World).unwrap();

    // At the reference reading the joint sits at its origin.
    let at_reference = tree.forward_kinematics(&Vector::new([FRAC_PI_4])).unwrap();
    assert_vector_close(
        at_reference.pose(0).unwrap().rotation().act(axis_x()),
        [1.0, 0.0, 0.0],
        "rotated x axis at the reference reading",
    );

    // A reading of zero is a quarter turn short of it.
    let at_zero = tree.forward_kinematics(&Vector::new([0.0])).unwrap();
    assert_vector_close(
        at_zero.pose(0).unwrap().rotation().act(axis_x()),
        [FRAC_PI_4.cos(), -FRAC_PI_4.sin(), 0.0],
        "rotated x axis at zero",
    );
}

// ---- topology ---------------------------------------------------------------

#[test]
fn two_joint_planar_arm_matches_the_closed_form() {
    let tree = planar_arm::<f64>();
    let (shoulder, elbow) = (0.3, -0.7);

    let state = tree
        .forward_kinematics(&Vector::new([shoulder, elbow, 0.0]))
        .unwrap();

    assert_vector_close(
        state.pose(2).unwrap().translation(),
        [
            shoulder.cos() + (shoulder + elbow).cos(),
            shoulder.sin() + (shoulder + elbow).sin(),
            0.0,
        ],
        "tool position",
    );
}

#[test]
fn branching_tree_settles_both_arms() {
    let first_branch = translation(1.0, 0.0, 0.0);
    let second_branch = translation(0.0, 1.0, 0.0);
    let tree = KinematicTree::<3, f64>::try_from_joints(
        &[
            Joint::revolute(axis_z::<f64>(), SE3::identity()),
            Joint::fixed(first_branch),
            Joint::fixed(second_branch),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(0),
        ],
    )
    .unwrap();

    let state = tree
        .forward_kinematics(&Vector::new([0.4, 0.0, 0.0]))
        .unwrap();
    let root = state.pose(0).unwrap();

    // Each branch hangs off the root alone, so neither sees the other's offset.
    assert_vector_close(
        state.pose(1).unwrap().translation(),
        *(root * first_branch).translation().as_array(),
        "first branch",
    );
    assert_vector_close(
        state.pose(2).unwrap().translation(),
        *(root * second_branch).translation().as_array(),
        "second branch",
    );
}

#[test]
fn unused_slots_are_not_reported() {
    let link = translation(1.0, 0.0, 0.0);
    let tree = KinematicTree::<8, f64>::try_from_joints(
        &[
            Joint::revolute(axis_z::<f64>(), SE3::identity()),
            Joint::revolute(axis_z::<f64>(), link),
            Joint::fixed(link),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
        ],
    )
    .unwrap();

    let state = tree.forward_kinematics(&Vector::zeros()).unwrap();

    assert_eq!(state.len(), 3);
    assert_eq!(state.poses().len(), 3);
    assert_eq!(state.pose(3), None);
    assert_eq!(tree.joint(3), None);
    assert_eq!(tree.parent(3), None);
}

#[test]
fn rejects_a_non_finite_reading() {
    let mut movable = KinematicTree::<1, f64>::new();
    movable
        .push(
            Joint::revolute(axis_z::<f64>(), SE3::identity()),
            JointParent::World,
        )
        .unwrap();
    assert_eq!(
        movable.forward_kinematics(&Vector::new([f64::NAN])),
        Err(KinematicsError::NonFinite)
    );

    // A weld never reads its slot, so the same reading is harmless.
    let mut welded = KinematicTree::<1, f64>::new();
    welded
        .push(Joint::fixed(SE3::identity()), JointParent::World)
        .unwrap();
    assert!(welded.forward_kinematics(&Vector::new([f64::NAN])).is_ok());
}

// ---- scalar coverage --------------------------------------------------------

#[test]
fn runs_under_dual_and_matches_finite_differences() {
    let (shoulder, elbow) = (0.3, -0.7);
    let tool_position = |shoulder: f64| {
        *planar_arm::<f64>()
            .forward_kinematics(&Vector::new([shoulder, elbow, 0.0]))
            .unwrap()
            .pose(2)
            .unwrap()
            .translation()
            .as_array()
    };

    let state = planar_arm::<Dual<f64>>()
        .forward_kinematics(&Vector::new([
            Dual::variable(shoulder),
            Dual::constant(elbow),
            Dual::constant(0.0),
        ]))
        .unwrap();
    let tool = *state.pose(2).unwrap().translation().as_array();

    // Central difference of the same arm in plain f64.
    let step = 1e-6;
    let ahead = tool_position(shoulder + step);
    let behind = tool_position(shoulder - step);

    for (index, derivative) in tool.iter().map(|value| value.deriv).enumerate() {
        let finite_difference = (ahead[index] - behind[index]) / (2.0 * step);
        assert!(
            (derivative - finite_difference).abs() < 1e-7,
            "component {index}: autodiff {derivative}, finite difference {finite_difference}"
        );
    }
}

#[test]
fn runs_in_f32() {
    let (shoulder, elbow) = (0.3_f32, -0.7_f32);
    let state = planar_arm::<f32>()
        .forward_kinematics(&Vector::new([shoulder, elbow, 0.0]))
        .unwrap();
    let [x, y, _] = *state.pose(2).unwrap().translation().as_array();

    assert!(
        (x - (shoulder.cos() + (shoulder + elbow).cos())).abs() < 1e-5,
        "tool x: {x}"
    );
    assert!(
        (y - (shoulder.sin() + (shoulder + elbow).sin())).abs() < 1e-5,
        "tool y: {y}"
    );
}
