//! Continuous-joint tests: construction and limit refusal, FK and Jacobian parity with an
//! unlimited `Revolute` joint past ±2π, wrapped configuration distance, shortest-arc
//! preferred-posture bias, and f32 coverage.

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

/// One joint about z, tool frame welded 1 m along x.
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

/// Tool pose matches between an unlimited revolute chain and a continuous one at the same reading.
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
    // No folding back past 2*pi: 7 rad is 7 rad.
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

    // The weld's slot is never read.
    let weld_only = tree.configuration_distance(&Vector::new([1.0, 0.0]), &Vector::new([1.0, 5.0]));
    assert_close(weld_only, 0.0, "weld-only difference");

    let turn_only =
        tree.configuration_distance(&Vector::new([3.0, 0.0]), &Vector::new([-3.0, 0.0]));
    let both = tree.configuration_distance(&Vector::new([3.0, 0.0]), &Vector::new([-3.0, 5.0]));
    assert_close(both, turn_only, "weld contributes nothing");
}

// ---- secondary bias ---------------------------------------------------------

/// Two coaxial joints about z at the same origin, tool on the second. Rank 1 against 2 DOF, so the
/// null space carries the secondary objective.
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

/// Shortest-arc error from `reading` to the preferred 3.0 rad.
fn wrapped_gap_to_preference(reading: f64) -> f64 {
    (reading - 3.0).wrap_to_pi().abs()
}

/// One iteration of a stacked pair seeded at -3.0 rad, preferring 3.0 rad; returns joint 0.
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
    // -3.0 to 3.0 is -0.283 rad the short way, so the null-space term drives negative.
    let before = wrapped_gap_to_preference(-3.0);
    let after = wrapped_gap_to_preference(first_reading_after_one_solve(JointKind::Continuous));
    assert!(after < before, "gap after {after} should be under {before}");
}

#[test]
fn preferred_posture_takes_the_long_way_for_a_revolute_joint() {
    // Aperiodic: the same preference reads as +6 rad, driving positive and widening the arc.
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
