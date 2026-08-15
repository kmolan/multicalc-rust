//! Continuous-joint tests: construction and limit refusal, forward-kinematics and Jacobian parity
//! with an equivalent unlimited `Revolute` joint at an angle past ±2π, wrap-aware
//! configuration-space distance, wrap-aware preferred-posture bias, and f32 coverage.

use core::f64::consts::PI;

use multicalc::error::KinematicsError;
use multicalc::kinematics::{
    InverseKinematics, JacobianFrame, Joint, JointKind, JointParent, KinematicTree,
    SecondaryObjective,
};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::scalar::Numeric;
use multicalc::spatial::{SE3, SO3};

const TOL: f64 = 1e-12;

// ---- helpers ----------------------------------------------------------------

fn axis_z<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ZERO, T::ZERO, T::ONE])
}

fn translation<T: Numeric>(x: T, y: T, z: T) -> SE3<T> {
    SE3::from_parts(SO3::identity(), Vector::new([x, y, z]))
}

/// One joint of the given kind about z, with a tool frame one unit out along x.
fn single_joint_arm<T: Numeric>(joint: Joint<T>) -> KinematicTree<2, 2, T> {
    KinematicTree::try_from_joints(
        &[joint, Joint::fixed(translation(T::ONE, T::ZERO, T::ZERO))],
        &[JointParent::World, JointParent::Joint(0)],
    )
    .unwrap()
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < TOL, "{what}: got {got}, want {want}");
}

/// Tool pose and Jacobian column agree between an unlimited revolute arm and a continuous one at
/// the same reading.
fn assert_matches_revolute_at(reading: f64) {
    let revolute = single_joint_arm(Joint::revolute(axis_z::<f64>(), SE3::identity()));
    let continuous = single_joint_arm(Joint::continuous(axis_z::<f64>(), SE3::identity()));
    let configuration = Vector::new([reading, 0.0]);

    let revolute_pose = revolute
        .forward_kinematics(&configuration)
        .unwrap()
        .pose(1)
        .unwrap();
    let continuous_pose = continuous
        .forward_kinematics(&configuration)
        .unwrap()
        .pose(1)
        .unwrap();

    let got = *continuous_pose.translation().as_array();
    let want = *revolute_pose.translation().as_array();
    for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(*got, *want, &format!("translation component {index}"));
    }
    let got = continuous_pose.rotation().quaternion().as_array();
    let want = revolute_pose.rotation().quaternion().as_array();
    for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        assert_close(*got, *want, &format!("quaternion component {index}"));
    }
}

// ---- construction -----------------------------------------------------------

#[test]
fn continuous_joint_has_no_limits() {
    let joint = Joint::continuous(axis_z::<f64>(), SE3::identity());
    assert_eq!(joint.kind(), JointKind::Continuous);
    assert_eq!(joint.limits(), None);
}

#[test]
fn rejects_limits_on_a_continuous_joint() {
    let mut tree = KinematicTree::<1, 1, f64>::new();
    assert_eq!(
        tree.push(
            Joint::continuous(axis_z(), SE3::identity()).with_limits(-1.0, 1.0),
            JointParent::World
        ),
        Err(KinematicsError::ContinuousJointHasLimits)
    );
}

#[test]
fn rejects_a_zero_axis_the_same_as_revolute() {
    let mut tree = KinematicTree::<1, 1, f64>::new();
    assert_eq!(
        tree.push(
            Joint::continuous(Vector::zeros(), SE3::identity()),
            JointParent::World
        ),
        Err(KinematicsError::AxisHasNoDirection)
    );
}

// ---- forward kinematics and Jacobian parity ---------------------------------

#[test]
fn matches_an_unlimited_revolute_at_a_wrapped_angle() {
    assert_matches_revolute_at(4.2);
    assert_matches_revolute_at(-4.2);
    assert_matches_revolute_at(4.2 + 2.0 * PI);
}

#[test]
fn turns_as_far_as_the_reading_says() {
    // Past a full turn the tool keeps going round rather than folding back: 7 rad is 7 - 2*pi past
    // the start.
    let tree = single_joint_arm(Joint::continuous(axis_z::<f64>(), SE3::identity()));
    let tool = tree
        .forward_kinematics(&Vector::new([7.0, 0.0]))
        .unwrap()
        .pose(1)
        .unwrap();
    let [x, y, _] = *tool.translation().as_array();
    assert_close(x, 7.0_f64.cos(), "tool x");
    assert_close(y, 7.0_f64.sin(), "tool y");
}

#[test]
fn jacobian_column_matches_an_unlimited_revolute() {
    let revolute = single_joint_arm(Joint::revolute(axis_z::<f64>(), SE3::identity()));
    let continuous = single_joint_arm(Joint::continuous(axis_z::<f64>(), SE3::identity()));
    let configuration = Vector::new([1.0, 0.0]);

    let revolute_jacobian = revolute
        .geometric_jacobian_at(&configuration, 1, JacobianFrame::World)
        .unwrap();
    let continuous_jacobian = continuous
        .geometric_jacobian_at(&configuration, 1, JacobianFrame::World)
        .unwrap();

    let revolute_column = revolute_jacobian.column(0).unwrap();
    let continuous_column = continuous_jacobian.column(0).unwrap();
    assert!((continuous_column.linear() - revolute_column.linear()).norm() < TOL);
    assert!((continuous_column.angular() - revolute_column.angular()).norm() < TOL);
}

// ---- configuration distance -------------------------------------------------

#[test]
fn wraps_the_shortest_way_around() {
    let tree = single_joint_arm(Joint::continuous(axis_z::<f64>(), SE3::identity()));
    let distance = tree.configuration_distance(&Vector::new([3.0, 0.0]), &Vector::new([-3.0, 0.0]));
    assert_close(distance, 2.0 * PI - 6.0, "wrapped distance");
}

#[test]
fn matches_plain_difference_for_a_revolute_joint() {
    let tree = single_joint_arm(Joint::revolute(axis_z::<f64>(), SE3::identity()));
    let distance = tree.configuration_distance(&Vector::new([3.0, 0.0]), &Vector::new([-3.0, 0.0]));
    assert_close(distance, 6.0, "plain distance");
}

#[test]
fn zero_for_a_fixed_joint_regardless_of_reading() {
    let tree = single_joint_arm(Joint::continuous(axis_z::<f64>(), SE3::identity()));

    // The weld's slot is never read, so changing it alone changes nothing.
    let weld_only = tree.configuration_distance(&Vector::new([1.0, 0.0]), &Vector::new([1.0, 5.0]));
    assert_close(weld_only, 0.0, "weld-only difference");

    let turn_only =
        tree.configuration_distance(&Vector::new([3.0, 0.0]), &Vector::new([-3.0, 0.0]));
    let both = tree.configuration_distance(&Vector::new([3.0, 0.0]), &Vector::new([-3.0, 5.0]));
    assert_close(both, turn_only, "weld contributes nothing");
}

// ---- secondary bias ---------------------------------------------------------

/// Two joints of the same kind about z, stacked at the same point, with a tool frame on the second
/// one. Only their sum moves the tool, so one direction of joint motion is free for the secondary
/// objective to use.
fn stacked_pair(kind: JointKind) -> KinematicTree<2, 2, f64> {
    let build = |origin| match kind {
        JointKind::Continuous => Joint::continuous(axis_z::<f64>(), origin),
        _ => Joint::revolute(axis_z::<f64>(), origin),
    };
    KinematicTree::try_from_joints(
        &[build(SE3::identity()), build(SE3::identity())],
        &[JointParent::World, JointParent::Joint(0)],
    )
    .unwrap()
}

/// Distance from `reading` to the preferred 3.0 rad, measured the short way round.
fn wrapped_gap_to_preference(reading: f64) -> f64 {
    (reading - 3.0).wrap_to_pi().abs()
}

/// One solve of a stacked pair seeded near -3.0 rad, preferring 3.0 rad, returning the first
/// joint's new reading.
fn first_reading_after_one_solve(kind: JointKind) -> f64 {
    let tree = stacked_pair(kind);
    let seed = Vector::new([-3.0, 0.0]);
    let target = SE3::from_parts(SO3::exp(axis_z::<f64>().scale(-2.9)), Vector::zeros());
    let solver = InverseKinematics::<2, f64>::new()
        .with_secondary_objective(SecondaryObjective::PreferredPosture(Vector::new([
            3.0, 0.0,
        ])))
        .with_secondary_gain(1.0)
        .with_maximum_iterations(1);
    let report = solver.solve(&tree, 1, target, &seed).unwrap();
    *report.joint_positions.get(0).unwrap()
}

#[test]
fn preferred_posture_wraps_for_a_continuous_joint() {
    // -3.0 rad is already close to a preferred 3.0 rad the short way round, so the joint should
    // close that small gap rather than travelling the long way.
    let before = wrapped_gap_to_preference(-3.0);
    let after = wrapped_gap_to_preference(first_reading_after_one_solve(JointKind::Continuous));
    assert!(after < before, "gap after {after} should be under {before}");
}

#[test]
fn preferred_posture_takes_the_long_way_for_a_revolute_joint() {
    // A limited joint has no periodicity to exploit, so the same preference drives it the other
    // way — the full 6 rad — and the short-way gap grows.
    let before = wrapped_gap_to_preference(-3.0);
    let after = wrapped_gap_to_preference(first_reading_after_one_solve(JointKind::Revolute));
    assert!(after > before, "gap after {after} should be over {before}");
}

// ---- errors -----------------------------------------------------------------

#[test]
fn reversed_limits_are_reported_before_the_continuous_refusal() {
    let mut tree = KinematicTree::<1, 1, f64>::new();
    assert_eq!(
        tree.push(
            Joint::continuous(axis_z(), SE3::identity()).with_limits(1.0, -1.0),
            JointParent::World
        ),
        Err(KinematicsError::LimitsReversed)
    );
    assert_eq!(
        tree.push(
            Joint::continuous(axis_z(), SE3::identity()).with_limits(-1.0, 1.0),
            JointParent::World
        ),
        Err(KinematicsError::ContinuousJointHasLimits)
    );
}

// ---- scalar coverage --------------------------------------------------------

#[test]
fn runs_in_f32() {
    let reading = 4.2_f32;
    let configuration = Vector::new([reading, 0.0]);
    let revolute = single_joint_arm(Joint::revolute(axis_z::<f32>(), SE3::identity()));
    let continuous = single_joint_arm(Joint::continuous(axis_z::<f32>(), SE3::identity()));

    let revolute_tool = revolute
        .forward_kinematics(&configuration)
        .unwrap()
        .pose(1)
        .unwrap();
    let continuous_tool = continuous
        .forward_kinematics(&configuration)
        .unwrap()
        .pose(1)
        .unwrap();
    assert!((continuous_tool.translation() - revolute_tool.translation()).norm() < 1e-5);

    let distance =
        continuous.configuration_distance(&Vector::new([3.0_f32, 0.0]), &Vector::new([-3.0, 0.0]));
    assert!(
        (distance - (2.0 * core::f32::consts::PI - 6.0)).abs() < 1e-5,
        "wrapped distance: {distance}"
    );
}
