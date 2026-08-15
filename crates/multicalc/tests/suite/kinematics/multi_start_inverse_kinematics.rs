//! Multi-start IK tests: branch enumeration from a seed array and from PRNG-jittered draws,
//! deduplication by configuration distance, nearest-branch selection, capacity errors, and
//! PRNG reproducibility.

use core::f64::consts::PI;

use multicalc::error::KinematicsError;
use multicalc::kinematics::{
    InverseKinematicsTermination, Joint, JointParent, KinematicTree, MultiStartInverseKinematics,
};
use multicalc::linear_algebra::{Vector, Vector3D};
use multicalc::random::Pcg32;
use multicalc::scalar::Numeric;
use multicalc::spatial::{SE3, SO3};

/// Slots in the test arm: six hinges plus the welded tool.
const JOINTS: usize = 7;

/// Tool frame index.
const TOOL: usize = 6;

// ---- helpers ----------------------------------------------------------------

fn axis_x<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ONE, T::ZERO, T::ZERO])
}

fn axis_y<T: Numeric>() -> Vector3D<T> {
    Vector::new([T::ZERO, T::ONE, T::ZERO])
}

fn translation<T: Numeric>(x: f64, y: f64, z: f64) -> SE3<T> {
    SE3::from_parts(
        SO3::identity(),
        Vector::new([T::from_f64(x), T::from_f64(y), T::from_f64(z)]),
    )
}

/// Six hinges alternating about x and y on 0.25 m links, tool welded 0.25 m past the last. Six
/// actuated DOF against a six-DOF task, so its solutions are discrete branches. 1.5 m of reach.
fn spatial_arm(unbounded: bool) -> KinematicTree<JOINTS, JOINTS, f64> {
    let link = translation::<f64>(0.0, 0.0, 0.25);
    let mut tree = KinematicTree::new();
    for index in 0..6 {
        let axis = if index % 2 == 0 { axis_x() } else { axis_y() };
        let origin = if index == 0 { SE3::identity() } else { link };
        let joint = if unbounded {
            Joint::continuous(axis, origin)
        } else {
            Joint::revolute(axis, origin)
        };
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

/// A configuration off every joint's zero, used as the pose to solve back to.
fn posture() -> Vector<JOINTS, f64> {
    Vector::new([0.3, 0.6, -0.4, 0.9, 0.2, -0.5, 0.0])
}

fn target_at(
    tree: &KinematicTree<JOINTS, JOINTS, f64>,
    configuration: &Vector<JOINTS, f64>,
) -> SE3<f64> {
    tree.forward_kinematics(configuration)
        .unwrap()
        .pose(TOOL)
        .unwrap()
}

// ---- solve_from_seeds -------------------------------------------------------

#[test]
fn separated_seeds_find_more_than_one_branch() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());
    let seeds = [
        posture(),
        Vector::new([-0.3, -0.6, 0.4, -0.9, -0.2, 0.5, 0.0]),
        Vector::new([1.0, -1.2, 0.8, -1.5, 0.6, 1.1, 0.0]),
    ];

    let report = MultiStartInverseKinematics::<4, JOINTS, f64>::new()
        .solve_from_seeds(&tree, TOOL, target, &seeds)
        .unwrap();

    assert_eq!(report.attempts(), 3);
    assert!(report.len() >= 2, "distinct branches: {}", report.len());
    for solution in report.solutions() {
        assert_eq!(
            solution.termination,
            InverseKinematicsTermination::Converged
        );
        // Every branch reaches the same pose, whatever readings it got there with.
        let reached = target_at(&tree, &solution.joint_positions);
        assert!((reached.translation() - target.translation()).norm() < 1e-6);
    }
    // Distinctness is the stored threshold, not an accident of ordering.
    for (index, first) in report.solutions().iter().enumerate() {
        for second in report.solutions().iter().skip(index + 1) {
            let apart =
                tree.configuration_distance(&first.joint_positions, &second.joint_positions);
            assert!(apart >= 1e-3, "branches {apart} apart");
        }
    }
}

#[test]
fn seeds_landing_on_one_branch_are_deduplicated() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());
    let mut nearby = posture();
    if let Some(slot) = nearby.get_mut(0) {
        *slot += 1e-2;
    }

    let report = MultiStartInverseKinematics::<4, JOINTS, f64>::new()
        .solve_from_seeds(&tree, TOOL, target, &[posture(), nearby])
        .unwrap();

    assert_eq!(report.attempts(), 2);
    assert_eq!(report.len(), 1);
}

#[test]
fn the_distinct_threshold_decides_what_counts_as_one_branch() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());
    let seeds = [
        posture(),
        Vector::new([-0.3, -0.6, 0.4, -0.9, -0.2, 0.5, 0.0]),
    ];

    let separated = MultiStartInverseKinematics::<4, JOINTS, f64>::new()
        .solve_from_seeds(&tree, TOOL, target, &seeds)
        .unwrap();
    // A threshold wider than the whole configuration space collapses everything to one.
    let collapsed = MultiStartInverseKinematics::<4, JOINTS, f64>::new()
        .with_distinct_threshold(1e3)
        .solve_from_seeds(&tree, TOOL, target, &seeds)
        .unwrap();

    assert_eq!(separated.len(), 2);
    assert_eq!(collapsed.len(), 1);
}

#[test]
fn unbounded_joints_fold_a_2pi_apart_seed_onto_one_branch() {
    // The same readings 2*pi apart: a plain difference on revolute joints, nothing on continuous
    // ones, so only the unbounded arm treats them as one branch.
    let mut wrapped = posture();
    if let Some(slot) = wrapped.get_mut(0) {
        *slot += 2.0 * PI;
    }

    let bounded = spatial_arm(false);
    let bounded_report = MultiStartInverseKinematics::<4, JOINTS, f64>::new()
        .solve_from_seeds(
            &bounded,
            TOOL,
            target_at(&bounded, &posture()),
            &[posture(), wrapped],
        )
        .unwrap();

    let unbounded = spatial_arm(true);
    let unbounded_report = MultiStartInverseKinematics::<4, JOINTS, f64>::new()
        .solve_from_seeds(
            &unbounded,
            TOOL,
            target_at(&unbounded, &posture()),
            &[posture(), wrapped],
        )
        .unwrap();

    assert_eq!(bounded_report.len(), 2);
    assert_eq!(unbounded_report.len(), 1);
}

#[test]
fn an_unreachable_target_leaves_the_report_empty() {
    let tree = spatial_arm(false);
    // 5 m out against 1.5 m of reach.
    let target = translation::<f64>(5.0, 0.0, 0.0);

    let report = MultiStartInverseKinematics::<4, JOINTS, f64>::new()
        .solve_from_seeds(&tree, TOOL, target, &[posture()])
        .unwrap();

    assert!(report.is_empty());
    assert_eq!(report.len(), 0);
    assert_eq!(report.attempts(), 1);
    assert!(report.closest_to(&tree, &posture()).is_none());
}

#[test]
fn errors_past_seed_capacity() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());
    let seeds = [posture(); 3];

    assert_eq!(
        MultiStartInverseKinematics::<2, JOINTS, f64>::new()
            .solve_from_seeds(&tree, TOOL, target, &seeds)
            .err(),
        Some(KinematicsError::StartCapacityExceeded)
    );
}

#[test]
fn passes_the_inner_solver_errors_through() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());

    assert_eq!(
        MultiStartInverseKinematics::<2, JOINTS, f64>::new()
            .solve_from_seeds(&tree, JOINTS, target, &[posture()])
            .err(),
        Some(KinematicsError::ToolIndexOutOfRange)
    );
}

// ---- solve_seeded -----------------------------------------------------------

#[test]
fn the_base_seed_runs_unperturbed() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());
    let mut source = Pcg32::<f64>::new(7);

    // One start is the base seed alone: already at the answer, so it converges where it sits.
    let report = MultiStartInverseKinematics::<4, JOINTS, f64>::new()
        .solve_seeded(&tree, TOOL, target, &posture(), &mut source, 1)
        .unwrap();

    assert_eq!(report.attempts(), 1);
    assert_eq!(report.len(), 1);
    let solution = report.solutions().first().unwrap();
    assert!(tree.configuration_distance(&solution.joint_positions, &posture()) < 1e-6);
}

#[test]
fn jittered_starts_reach_branches_the_base_seed_does_not() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());
    let mut source = Pcg32::<f64>::new(11);

    let report = MultiStartInverseKinematics::<8, JOINTS, f64>::new()
        .solve_seeded(&tree, TOOL, target, &posture(), &mut source, 6)
        .unwrap();

    assert_eq!(report.attempts(), 6);
    assert!(report.len() >= 2, "distinct branches: {}", report.len());
}

#[test]
fn the_same_seed_reproduces_the_same_report() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());

    let mut first_source = Pcg32::<f64>::new(11);
    let mut second_source = Pcg32::<f64>::new(11);
    let solver = MultiStartInverseKinematics::<8, JOINTS, f64>::new();
    let first = solver
        .solve_seeded(&tree, TOOL, target, &posture(), &mut first_source, 5)
        .unwrap();
    let second = solver
        .solve_seeded(&tree, TOOL, target, &posture(), &mut second_source, 5)
        .unwrap();

    assert_eq!(first.len(), second.len());
    assert_eq!(first.attempts(), second.attempts());
    for (first, second) in first.solutions().iter().zip(second.solutions().iter()) {
        assert_eq!(first.joint_positions, second.joint_positions);
        assert_eq!(first.termination, second.termination);
        assert_eq!(first.iterations, second.iterations);
    }
}

#[test]
fn a_different_seed_draws_different_starts() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());
    let solver = MultiStartInverseKinematics::<8, JOINTS, f64>::new();

    let mut first_source = Pcg32::<f64>::new(11);
    let mut second_source = Pcg32::<f64>::new(12);
    let first = solver
        .solve_seeded(&tree, TOOL, target, &posture(), &mut first_source, 5)
        .unwrap();
    let second = solver
        .solve_seeded(&tree, TOOL, target, &posture(), &mut second_source, 5)
        .unwrap();

    // Both hold the unperturbed base-seed solution; past that the branch sets differ.
    let same = first.len() == second.len()
        && first
            .solutions()
            .iter()
            .zip(second.solutions().iter())
            .all(|(first, second)| first.joint_positions == second.joint_positions);
    assert!(!same, "two generator seeds produced identical branch sets");
}

#[test]
fn jitter_stays_inside_a_bounded_joint() {
    // Every hinge held to a narrow range: a jittered draw is redrawn inside it, so no start can
    // seed outside the travel and the solver's clamping never has to rescue it.
    let link = translation::<f64>(0.0, 0.0, 0.25);
    let mut tree = KinematicTree::<JOINTS, JOINTS, f64>::new();
    for index in 0..6 {
        let axis = if index % 2 == 0 { axis_x() } else { axis_y() };
        let origin = if index == 0 { SE3::identity() } else { link };
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(Joint::revolute(axis, origin).with_limits(-0.2, 0.2), parent)
            .unwrap();
    }
    tree.push(Joint::fixed(link), JointParent::Joint(5))
        .unwrap();

    let inside = Vector::new([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]);
    let target = target_at(&tree, &inside);
    let mut source = Pcg32::<f64>::new(3);

    let report = MultiStartInverseKinematics::<8, JOINTS, f64>::new()
        .solve_seeded(&tree, TOOL, target, &inside, &mut source, 6)
        .unwrap();

    for solution in report.solutions() {
        for index in 0..6 {
            let reading = *solution.joint_positions.get(index).unwrap();
            assert!((-0.2..=0.2).contains(&reading), "joint {index}: {reading}");
        }
    }
}

#[test]
fn errors_past_seed_capacity_when_jittering() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());
    let mut source = Pcg32::<f64>::new(1);

    assert_eq!(
        MultiStartInverseKinematics::<2, JOINTS, f64>::new()
            .solve_seeded(&tree, TOOL, target, &posture(), &mut source, 3)
            .err(),
        Some(KinematicsError::StartCapacityExceeded)
    );
}

// ---- closest_to -------------------------------------------------------------

#[test]
fn closest_to_picks_the_nearest_branch() {
    let tree = spatial_arm(false);
    let target = target_at(&tree, &posture());
    let seeds = [
        posture(),
        Vector::new([-0.3, -0.6, 0.4, -0.9, -0.2, 0.5, 0.0]),
        Vector::new([1.0, -1.2, 0.8, -1.5, 0.6, 1.1, 0.0]),
    ];

    let report = MultiStartInverseKinematics::<4, JOINTS, f64>::new()
        .solve_from_seeds(&tree, TOOL, target, &seeds)
        .unwrap();

    // Asked from the posture itself, the branch standing on it wins.
    let nearest = report.closest_to(&tree, &posture()).unwrap();
    assert!(tree.configuration_distance(&nearest.joint_positions, &posture()) < 1e-6);

    // Asked from any reference, no other kept branch is nearer.
    let reference = Vector::new([0.0, 1.4, 0.0, -0.9, 0.0, 0.4, 0.0]);
    let nearest = report.closest_to(&tree, &reference).unwrap();
    let best = tree.configuration_distance(&nearest.joint_positions, &reference);
    for solution in report.solutions() {
        assert!(tree.configuration_distance(&solution.joint_positions, &reference) >= best);
    }
}
