//! Inverse-kinematics tests: convergence, the forward-kinematics round trip, termination reasons,
//! joint limits, step limiting, singular seeds, error cases, and f32 coverage.

use multicalc::error::KinematicsError;
use multicalc::kinematics::{
    InverseKinematics, InverseKinematicsTermination, JacobianFrame, Joint, JointParent,
    KinematicTree, SecondaryObjective,
};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::scalar::Numeric;
use multicalc::spatial::{SE3, SO3};

/// Slots in the spatial arm: six revolute joints plus the welded tool.
const SPATIAL_JOINTS: usize = 7;

// ---- helpers ----------------------------------------------------------------

fn axis_x<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ONE, T::ZERO, T::ZERO])
}

fn axis_y<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ZERO, T::ONE, T::ZERO])
}

fn axis_z<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ZERO, T::ZERO, T::ONE])
}

fn translation<T: Numeric>(x: f64, y: f64, z: f64) -> SE3<T> {
    SE3::from_parts(
        SO3::identity(),
        Vector::new([T::from_f64(x), T::from_f64(y), T::from_f64(z)]),
    )
}

/// Two revolute joints about z on unit links, plus a fixed tool one unit past the elbow.
fn planar_arm<T: Numeric>() -> KinematicTree<3, T> {
    let link = translation(1.0, 0.0, 0.0);
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

/// Six revolute joints alternating about x and y, each a 0.25 link up the parent's z, with the
/// tool welded 0.25 further. Reach is 1.5 from the base.
fn spatial_arm<T: Numeric>() -> KinematicTree<SPATIAL_JOINTS, T> {
    build_spatial_arm(None)
}

/// The same arm with every revolute joint held to the same travel.
fn limited_spatial_arm<T: Numeric>(lower: f64, upper: f64) -> KinematicTree<SPATIAL_JOINTS, T> {
    build_spatial_arm(Some((lower, upper)))
}

fn build_spatial_arm<T: Numeric>(limits: Option<(f64, f64)>) -> KinematicTree<SPATIAL_JOINTS, T> {
    let link = translation::<T>(0.0, 0.0, 0.25);
    let mut tree = KinematicTree::new();
    for index in 0..6 {
        let axis = if index % 2 == 0 { axis_x() } else { axis_y() };
        let origin = if index == 0 { SE3::identity() } else { link };
        let mut joint = Joint::revolute(axis, origin);
        if let Some((lower, upper)) = limits {
            joint = joint.with_limits(T::from_f64(lower), T::from_f64(upper));
        }
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(joint, parent).unwrap();
    }
    tree.push(Joint::fixed(link), JointParent::Joint(5))
        .unwrap();
    tree
}

/// Slots in the redundant arm: seven revolute joints plus the welded tool.
const REDUNDANT_JOINTS: usize = 8;

/// Seven revolute joints alternating about z and y on 0.25 links, with the tool welded 0.25
/// further.
///
/// Seven joints against a six-dimensional task, so one degree of freedom is spare — which the
/// six-joint arm above does not have, and without which a null-space projector is just zero.
fn redundant_arm<T: Numeric>(limits: Option<(f64, f64)>) -> KinematicTree<REDUNDANT_JOINTS, T> {
    let link = translation::<T>(0.0, 0.0, 0.25);
    let mut tree = KinematicTree::new();
    for index in 0..7 {
        let axis = if index % 2 == 0 { axis_z() } else { axis_y() };
        let origin = if index == 0 { SE3::identity() } else { link };
        let mut joint = Joint::revolute(axis, origin);
        if let Some((lower, upper)) = limits {
            joint = joint.with_limits(T::from_f64(lower), T::from_f64(upper));
        }
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(joint, parent).unwrap();
    }
    tree.push(Joint::fixed(link), JointParent::Joint(6))
        .unwrap();
    tree
}

/// A configuration of the redundant arm that leaves no joint at a special angle.
fn redundant_readings() -> Vector<REDUNDANT_JOINTS, f64> {
    Vector::new([0.3, -0.5, 0.4, 0.7, -0.3, 0.5, 0.2, 0.0])
}

fn redundant_tool_pose(
    tree: &KinematicTree<REDUNDANT_JOINTS, f64>,
    readings: &Vector<REDUNDANT_JOINTS, f64>,
) -> SE3<f64> {
    tree.forward_kinematics(readings).unwrap().pose(7).unwrap()
}

fn redundant_perturbed(
    readings: &Vector<REDUNDANT_JOINTS, f64>,
    offset: f64,
) -> Vector<REDUNDANT_JOINTS, f64> {
    Vector::from_fn(|index| {
        let reading = readings.get(index).copied().unwrap_or(0.0);
        if index < 7 { reading + offset } else { reading }
    })
}

/// A configuration that leaves no joint at a special angle.
fn generic_readings() -> Vector<SPATIAL_JOINTS, f64> {
    Vector::new([0.2, -0.4, 0.5, 0.3, -0.2, 0.6, 0.0])
}

/// The same readings pushed 0.3 rad off, so the solver has real work to do.
fn perturbed(readings: &Vector<SPATIAL_JOINTS, f64>, offset: f64) -> Vector<SPATIAL_JOINTS, f64> {
    Vector::from_fn(|index| {
        let reading = readings.get(index).copied().unwrap_or(0.0);
        if index < 6 { reading + offset } else { reading }
    })
}

fn tool_pose(
    tree: &KinematicTree<SPATIAL_JOINTS, f64>,
    readings: &Vector<SPATIAL_JOINTS, f64>,
) -> SE3<f64> {
    tree.forward_kinematics(readings).unwrap().pose(6).unwrap()
}

/// How far apart two poses are, as a distance and a turn.
fn pose_gap(got: SE3<f64>, want: SE3<f64>) -> (f64, f64) {
    (
        (got.translation() - want.translation()).norm(),
        (want.rotation().inverse() * got.rotation()).log().norm(),
    )
}

// ---- convergence ------------------------------------------------------------

#[test]
fn reaches_a_reachable_pose() {
    let tree = spatial_arm::<f64>();
    let known = generic_readings();
    let target = tool_pose(&tree, &known);

    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .solve(&tree, 6, target, &perturbed(&known, 0.3))
        .unwrap();

    assert_eq!(report.termination, InverseKinematicsTermination::Converged);
    assert!(report.position_error < 1e-6, "{}", report.position_error);
    assert!(
        report.orientation_error < 1e-6,
        "{}",
        report.orientation_error
    );
}

#[test]
fn round_trips_through_forward_kinematics() {
    let tree = spatial_arm::<f64>();
    let known = generic_readings();
    let target = tool_pose(&tree, &known);

    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .solve(&tree, 6, target, &perturbed(&known, 0.3))
        .unwrap();

    // The answer is not unique, so the readings themselves prove nothing — putting them back
    // through forward kinematics has to land on the target.
    let (distance, turn) = pose_gap(tool_pose(&tree, &report.joint_positions), target);

    assert!(distance < 1e-6, "tool position off by {distance}");
    assert!(turn < 1e-6, "tool orientation off by {turn}");
}

#[test]
fn resolves_redundancy_without_losing_the_pose() {
    let tree = spatial_arm::<f64>();
    let known = generic_readings();
    let target = tool_pose(&tree, &known);

    // A secondary objective uses the spare freedom, so it must not cost any tracking accuracy.
    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .with_secondary_objective(SecondaryObjective::PreferredPosture(Vector::zeros()))
        .solve(&tree, 6, target, &perturbed(&known, 0.3))
        .unwrap();

    let (distance, turn) = pose_gap(tool_pose(&tree, &report.joint_positions), target);

    assert_eq!(report.termination, InverseKinematicsTermination::Converged);
    assert!(distance < 1e-6, "tool position off by {distance}");
    assert!(turn < 1e-6, "tool orientation off by {turn}");
}

// ---- termination ------------------------------------------------------------

#[test]
fn reports_the_budget_running_out() {
    let tree = spatial_arm::<f64>();
    let seed = Vector::new([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]);

    // Twice the arm's reach, so it can never arrive.
    let target = translation::<f64>(3.0, 0.0, 0.0);
    let start_gap = (tool_pose(&tree, &seed).translation() - target.translation()).norm();

    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .with_maximum_iterations(5)
        .solve(&tree, 6, target, &seed)
        .unwrap();

    assert_eq!(
        report.termination,
        InverseKinematicsTermination::IterationBudget
    );
    assert_eq!(report.iterations, 5);
    assert!(report.position_error.is_finite());
    assert!(
        report.position_error < start_gap,
        "made no progress: {} against {start_gap}",
        report.position_error
    );
}

#[test]
fn stalls_rather_than_spinning() {
    let tree = spatial_arm::<f64>();
    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .solve(
            &tree,
            6,
            translation(3.0, 0.0, 0.0),
            &Vector::new([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]),
        )
        .unwrap();

    assert!(
        matches!(
            report.termination,
            InverseKinematicsTermination::Stalled | InverseKinematicsTermination::IterationBudget
        ),
        "{:?}",
        report.termination
    );
    assert!(report.joint_positions.is_finite());
}

// ---- joint limits -----------------------------------------------------------

#[test]
fn keeps_every_reading_inside_its_travel() {
    let tree = limited_spatial_arm::<f64>(-0.5, 0.5);
    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .solve(&tree, 6, translation(3.0, 0.0, 0.0), &Vector::zeros())
        .unwrap();

    for index in 0..6 {
        let reading = report.joint_positions.get(index).copied().unwrap();
        assert!(
            (-0.5 - 1e-12..=0.5 + 1e-12).contains(&reading),
            "joint {index} left its travel at {reading}"
        );
    }
}

#[test]
fn ignores_limits_when_told_to() {
    let tree = limited_spatial_arm::<f64>(-0.5, 0.5);
    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .with_respect_limits(false)
        .solve(&tree, 6, translation(3.0, 0.0, 0.0), &Vector::zeros())
        .unwrap();

    let left_its_travel = (0..6).any(|index| {
        report
            .joint_positions
            .get(index)
            .copied()
            .is_some_and(|reading| reading.abs() > 0.5)
    });
    assert!(
        left_its_travel,
        "every reading stayed inside its travel, so the flag did nothing: {:?}",
        report.joint_positions
    );
}

// ---- step limiting ----------------------------------------------------------

#[test]
fn no_step_exceeds_the_cap() {
    let tree = spatial_arm::<f64>();
    let seed = Vector::new([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]);

    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .with_maximum_step_norm(0.01)
        .with_maximum_iterations(1)
        .solve(&tree, 6, translation(3.0, 0.0, 0.0), &seed)
        .unwrap();

    let moved = (report.joint_positions - seed).norm();
    assert!(moved <= 0.01 + 1e-12, "one step moved {moved}");
}

// ---- singular seeds ---------------------------------------------------------

#[test]
fn survives_a_stretched_out_arm() {
    let tree = spatial_arm::<f64>();

    // Every joint at zero leaves the arm straight up z with every axis in the x-y plane, so no
    // joint can turn the tool about z at all. The target asks for exactly that turn, which is the
    // case damping exists for: undamped, the readings run away to five figures.
    let target = SE3::from_parts(
        SO3::exp(Vector::new([0.0, 0.0, 0.6])),
        Vector::new([0.05, 0.0, 1.4]),
    );
    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .solve(&tree, 6, target, &Vector::zeros())
        .unwrap();

    assert!(report.joint_positions.is_finite());
    for index in 0..6 {
        let reading = report.joint_positions.get(index).copied().unwrap();
        assert!(reading.abs() < 100.0, "joint {index} ran off to {reading}");
    }
}

#[test]
fn stalls_when_steps_stop_making_progress() {
    let tree = limited_spatial_arm::<f64>(-0.5, 0.5);

    // Every joint pinned against its stop with the target still far off: what the limits leave of
    // each step falls under the floor, so the solve gives up rather than grinding out its budget.
    let report = InverseKinematics::<SPATIAL_JOINTS, f64>::new()
        .with_minimum_step_norm(1e-2)
        .solve(&tree, 6, translation(3.0, 0.0, 0.0), &Vector::zeros())
        .unwrap();

    assert_eq!(report.termination, InverseKinematicsTermination::Stalled);
    assert!(report.iterations < 100, "{}", report.iterations);
    assert!(report.joint_positions.is_finite());
}

// ---- errors -----------------------------------------------------------------

#[test]
fn errors_on_a_tool_index_past_the_tree() {
    let tree = planar_arm::<f64>();
    let solved = InverseKinematics::<3, f64>::new().solve(
        &tree,
        3,
        translation(1.0, 1.0, 0.0),
        &Vector::zeros(),
    );

    assert_eq!(solved.err(), Some(KinematicsError::ToolIndexOutOfRange));
}

#[test]
fn errors_on_a_non_positive_joint_weight() {
    let tree = planar_arm::<f64>();
    let solved = InverseKinematics::<3, f64>::new()
        .with_joint_weights(Vector::new([0.0, 1.0, 1.0]))
        .solve(&tree, 2, translation(1.0, 1.0, 0.0), &Vector::zeros());

    assert_eq!(solved.err(), Some(KinematicsError::NonPositiveWeight));
}

#[test]
fn errors_on_a_non_finite_seed() {
    let tree = planar_arm::<f64>();
    let solved = InverseKinematics::<3, f64>::new().solve(
        &tree,
        2,
        translation(1.0, 1.0, 0.0),
        &Vector::new([f64::NAN, 0.0, 0.0]),
    );

    assert_eq!(solved.err(), Some(KinematicsError::NonFinite));
}

// ---- scalar coverage --------------------------------------------------------

#[test]
fn runs_in_f32() {
    let tree = planar_arm::<f32>();
    let report = InverseKinematics::<3, f32>::new()
        .with_position_tolerance(1e-4)
        .with_orientation_tolerance(1e-4)
        .solve(
            &tree,
            2,
            translation(1.0, 1.0, 0.0),
            &Vector::new([0.3, 0.3, 0.0]),
        )
        .unwrap();

    assert_eq!(report.termination, InverseKinematicsTermination::Converged);
}

// ---- null-space --------------------------------------------------------------

/// How far the readings sit from a reference configuration.
fn distance_to(
    readings: &Vector<REDUNDANT_JOINTS, f64>,
    reference: &Vector<REDUNDANT_JOINTS, f64>,
) -> f64 {
    (*readings - *reference).norm()
}

/// The tightest any joint sits to a limit of its travel.
fn smallest_limit_margin(readings: &Vector<REDUNDANT_JOINTS, f64>, lower: f64, upper: f64) -> f64 {
    (0..7)
        .map(|index| {
            let reading = readings.get(index).copied().unwrap_or(0.0);
            (reading - lower).min(upper - reading)
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn secondary_objective_does_not_disturb_the_task() {
    let tree = redundant_arm::<f64>(None);
    let known = redundant_readings();
    let target = redundant_tool_pose(&tree, &known);
    let seed = redundant_perturbed(&known, 0.3);

    // The spare freedom is spare: using it must not cost any tracking accuracy.
    for objective in [
        SecondaryObjective::None,
        SecondaryObjective::PreferredPosture(Vector::zeros()),
    ] {
        let report = InverseKinematics::<REDUNDANT_JOINTS, f64>::new()
            .with_secondary_objective(objective)
            .solve(&tree, 7, target, &seed)
            .unwrap();
        let (distance, turn) =
            pose_gap(redundant_tool_pose(&tree, &report.joint_positions), target);

        assert_eq!(report.termination, InverseKinematicsTermination::Converged);
        assert!(distance < 1e-6, "{objective:?}: position off by {distance}");
        assert!(turn < 1e-6, "{objective:?}: orientation off by {turn}");
    }
}

#[test]
fn preferred_posture_pulls_the_arm_toward_it() {
    let tree = redundant_arm::<f64>(None);
    let known = redundant_readings();
    let target = redundant_tool_pose(&tree, &known);
    let seed = redundant_perturbed(&known, 0.3);
    let reference = Vector::zeros();

    let plain = InverseKinematics::<REDUNDANT_JOINTS, f64>::new()
        .solve(&tree, 7, target, &seed)
        .unwrap();
    let posture = InverseKinematics::<REDUNDANT_JOINTS, f64>::new()
        .with_secondary_objective(SecondaryObjective::PreferredPosture(reference))
        .solve(&tree, 7, target, &seed)
        .unwrap();

    let plain_distance = distance_to(&plain.joint_positions, &reference);
    let posture_distance = distance_to(&posture.joint_positions, &reference);

    assert!(
        posture_distance < plain_distance,
        "posture run ended {posture_distance} from the reference, plain run {plain_distance}"
    );
}

#[test]
fn joint_limit_margin_moves_off_the_limits() {
    let (lower, upper) = (-1.2, 1.2);
    let tree = redundant_arm::<f64>(Some((lower, upper)));

    // A pose whose natural answer leaves the base joint hard against its stop, approached from a
    // seed well away from it so the solve actually iterates — the spare freedom is only ever spent
    // on the way to the target, never after the task is already met.
    let known = Vector::new([1.15, 0.3, 0.2, 0.5, 0.2, 0.3, 0.2, 0.0]);
    let target = redundant_tool_pose(&tree, &known);
    let seed = redundant_perturbed(&known, -0.35);

    let plain = InverseKinematics::<REDUNDANT_JOINTS, f64>::new()
        .solve(&tree, 7, target, &seed)
        .unwrap();
    let margin_seeking = InverseKinematics::<REDUNDANT_JOINTS, f64>::new()
        .with_secondary_objective(SecondaryObjective::JointLimitMargin)
        .with_secondary_gain(0.5)
        .solve(&tree, 7, target, &seed)
        .unwrap();

    for report in [&plain, &margin_seeking] {
        let (distance, turn) =
            pose_gap(redundant_tool_pose(&tree, &report.joint_positions), target);
        assert!(distance < 1e-6, "position off by {distance}");
        assert!(turn < 1e-6, "orientation off by {turn}");
    }

    let plain_margin = smallest_limit_margin(&plain.joint_positions, lower, upper);
    let sought_margin = smallest_limit_margin(&margin_seeking.joint_positions, lower, upper);

    assert!(
        sought_margin > plain_margin,
        "margin-seeking run left {sought_margin} of travel, plain run {plain_margin}"
    );
}

#[test]
fn projector_kills_task_motion() {
    let tree = redundant_arm::<f64>(None);
    let jacobian = tree
        .geometric_jacobian_at(&redundant_readings(), 7, JacobianFrame::World)
        .unwrap();
    let weights = Vector::from_fn(|_| 1.0);

    // The projector's defining property, tested without going through the solver. It only has
    // anything to keep because this arm has a joint to spare.
    let wanted = Vector::new([1.0, -0.5, 0.25, 0.75, -0.3, 0.6, 0.4, 0.0]);
    let projected = jacobian.null_space_projector(&weights, 0.0).unwrap() * wanted;
    let twist = jacobian.tool_twist(&projected);

    assert!(
        projected.norm() > 0.1,
        "nothing survived the projection, so this proves nothing"
    );
    assert!(twist.linear().norm() < 1e-9, "{}", twist.linear().norm());
    assert!(twist.angular().norm() < 1e-9, "{}", twist.angular().norm());
}

#[test]
fn weighted_pseudo_inverse_produces_the_wanted_twist() {
    let tree = spatial_arm::<f64>();
    let jacobian = tree
        .geometric_jacobian_at(&generic_readings(), 6, JacobianFrame::World)
        .unwrap();
    let weights = Vector::from_fn(|_| 1.0);

    // A twist the arm can actually produce, so the inverse has an exact answer to find.
    let rates = Vector::new([0.4, -0.2, 0.7, 0.1, -0.5, 0.3, 0.0]);
    let wanted = jacobian.tool_twist(&rates).to_vector();
    let recovered = jacobian.damped_pseudo_inverse(&weights, 0.0).unwrap() * wanted;
    let produced = jacobian.tool_twist(&recovered).to_vector();

    assert!(wanted.norm() > 0.1, "the wanted twist is nearly nothing");
    assert!(
        (produced - wanted).norm() < 1e-9,
        "{}",
        (produced - wanted).norm()
    );
}

#[test]
fn heavier_joints_move_less() {
    let tree = spatial_arm::<f64>();
    let jacobian = tree
        .geometric_jacobian_at(&generic_readings(), 6, JacobianFrame::World)
        .unwrap();

    let rates = Vector::new([0.4, -0.2, 0.7, 0.1, -0.5, 0.3, 0.0]);
    let wanted = jacobian.tool_twist(&rates).to_vector();

    let uniform = Vector::from_fn(|_| 1.0);
    let heavy_base = Vector::from_fn(|index| if index == 0 { 100.0 } else { 1.0 });

    let even = jacobian.damped_pseudo_inverse(&uniform, 0.0).unwrap() * wanted;
    let sparing = jacobian.damped_pseudo_inverse(&heavy_base, 0.0).unwrap() * wanted;

    let even_base = even.get(0).copied().unwrap().abs();
    let sparing_base = sparing.get(0).copied().unwrap().abs();

    assert!(
        sparing_base < even_base,
        "weighting joint 0 heavily moved it {sparing_base}, evenly weighted {even_base}"
    );

    // Both still have to produce the twist that was asked for.
    for velocities in [even, sparing] {
        let produced = jacobian.tool_twist(&velocities).to_vector();
        assert!(
            (produced - wanted).norm() < 1e-9,
            "{}",
            (produced - wanted).norm()
        );
    }
}
