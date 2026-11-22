//! Articulated-body tests: the closed forms a single pendulum has, RNEA and CRBA read against each
//! other, what armature and joint friction each contribute, forward dynamics inverting inverse
//! dynamics, energy along a free swing, the `Dual` derivatives, and the constructor's refusals.

use core::f64::consts::FRAC_PI_2;

use multicalc::dynamics::{ArticulatedBody, DynamicsWorkspace};
use multicalc::error::DynamicsError;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::ode::Rk4;
use multicalc::scalar::{Dual, Numeric};
use multicalc::spatial::{SE3, SO3, SpatialInertia};

const GRAVITY: f64 = -9.81;

/// Eight configurations covering both signs and a fold-back, paired so no entry repeats its partner.
const SAMPLE_CONFIGURATIONS: [[f64; 2]; 8] = [
    [-2.0, -1.0],
    [-1.0, -0.4],
    [-0.4, 0.0],
    [0.0, 0.3],
    [0.3, 0.9],
    [0.9, 1.7],
    [1.7, 2.6],
    [2.6, -2.0],
];

/// One revolute joint about `+y` at the origin, a 2 kg link balancing 0.5 m out along `x`.
fn single_pendulum<T: Numeric>() -> ArticulatedBody<1, 1, T> {
    single_pendulum_with(T::ZERO, T::ZERO, T::ZERO)
}

fn single_pendulum_with<T: Numeric>(
    armature: T,
    damping: T,
    friction_loss: T,
) -> ArticulatedBody<1, 1, T> {
    let axis = Vector::new([T::ZERO, T::ONE, T::ZERO]);
    let hinge = Joint::revolute(axis, SE3::identity())
        .with_armature(armature)
        .with_damping(damping)
        .with_friction_loss(friction_loss);
    let tree = KinematicTree::<1, 1, T>::try_from_joints(&[hinge], &[JointParent::World]).unwrap();
    ArticulatedBody::new(tree, &[Some(link_inertia(2.0, 0.01))], earth_gravity()).unwrap()
}

/// Two revolute joints about `+y`, a unit link between them, both links 1.5 kg balancing 0.5 m out.
fn double_pendulum<T: Numeric>() -> ArticulatedBody<2, 2, T> {
    double_pendulum_with([0.05, 0.03], [0.7, 0.4], [0.2, 0.1])
}

/// The same geometry and masses with armature, damping and friction loss all zero.
fn double_pendulum_plain<T: Numeric>() -> ArticulatedBody<2, 2, T> {
    double_pendulum_with([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
}

fn double_pendulum_with<T: Numeric>(
    armature: [f64; 2],
    damping: [f64; 2],
    friction_loss: [f64; 2],
) -> ArticulatedBody<2, 2, T> {
    let axis = Vector::new([T::ZERO, T::ONE, T::ZERO]);
    let link = SE3::from_parts(SO3::identity(), Vector::new([T::ONE, T::ZERO, T::ZERO]));
    let origins = [SE3::identity(), link];

    let mut tree = KinematicTree::<2, 2, T>::new();
    for index in 0..2 {
        let joint = Joint::revolute(axis, origins[index])
            .with_armature(T::from_f64(armature[index]))
            .with_damping(T::from_f64(damping[index]))
            .with_friction_loss(T::from_f64(friction_loss[index]));
        let parent = if index == 0 {
            JointParent::World
        } else {
            JointParent::Joint(index - 1)
        };
        tree.push(joint, parent).unwrap();
    }

    let inertia = link_inertia(1.5, 0.02);
    ArticulatedBody::new(tree, &[Some(inertia), Some(inertia)], earth_gravity()).unwrap()
}

/// A link of `mass` balancing 0.5 m out along `x`, isotropic about its own balance point.
fn link_inertia<T: Numeric>(mass: f64, rotational: f64) -> SpatialInertia<T> {
    SpatialInertia::new(
        T::from_f64(mass),
        Vector::new([T::HALF, T::ZERO, T::ZERO]),
        Matrix::from_diagonal([
            T::from_f64(rotational),
            T::from_f64(rotational),
            T::from_f64(rotational),
        ]),
    )
    .unwrap()
}

fn earth_gravity<T: Numeric>() -> Vector<3, T> {
    Vector::new([T::ZERO, T::ZERO, T::from_f64(GRAVITY)])
}

fn pair(values: [f64; 2]) -> Vector<2, f64> {
    Vector::new(values)
}

#[test]
fn single_pendulum_gravity_torque_matches_closed_form() {
    let body = single_pendulum::<f64>();
    for angle in [0.0, 0.3, -1.1, FRAC_PI_2, 2.5] {
        let torque = body.gravity_torque_at(&Vector::new([angle])).unwrap();
        let want = 2.0 * GRAVITY * 0.5 * angle.cos();
        assert!(
            (torque[0] - want).abs() < 1e-12,
            "at {angle}: {} vs {want}",
            torque[0]
        );
    }
}

#[test]
fn single_pendulum_joint_space_inertia_matches_parallel_axis() {
    let body = single_pendulum::<f64>();
    for angle in [0.0, 0.3, -1.1, FRAC_PI_2, 2.5] {
        let inertia = body.joint_space_inertia_at(&Vector::new([angle])).unwrap();
        assert!((inertia[(0, 0)] - 0.51).abs() < 1e-12, "at {angle}");
    }
}

#[test]
fn single_pendulum_inertial_torque_matches_closed_form() {
    let body = single_pendulum::<f64>();
    for angle in [0.0, 0.3, -1.1, FRAC_PI_2, 2.5] {
        let torque = body
            .inverse_dynamics_at(&Vector::new([angle]), &Vector::zeros(), &Vector::new([1.3]))
            .unwrap();
        let want = 0.51 * 1.3 + 2.0 * GRAVITY * 0.5 * angle.cos();
        assert!((torque[0] - want).abs() < 1e-12, "at {angle}");
    }
}

#[test]
fn inverse_dynamics_and_joint_space_inertia_agree() {
    let body = double_pendulum_plain::<f64>();
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = pair(configuration);
        let inertia = body.joint_space_inertia_at(&position).unwrap();
        let at_rest = body
            .inverse_dynamics_at(&position, &Vector::zeros(), &Vector::zeros())
            .unwrap();

        for change in SAMPLE_CONFIGURATIONS {
            let acceleration = pair(change);
            let torque = body
                .inverse_dynamics_at(&position, &Vector::zeros(), &acceleration)
                .unwrap();
            let inertial = inertia * acceleration;
            for index in 0..2 {
                let got = torque[index] - at_rest[index];
                assert!(
                    (got - inertial[index]).abs() <= 1e-10 * inertial[index].abs().max(1.0),
                    "{configuration:?} {change:?} entry {index}: {got} vs {}",
                    inertial[index]
                );
            }
        }
    }
}

#[test]
fn joint_space_inertia_is_symmetric_and_positive_definite() {
    let body = double_pendulum::<f64>();
    for configuration in SAMPLE_CONFIGURATIONS {
        let inertia = body.joint_space_inertia_at(&pair(configuration)).unwrap();
        assert!((inertia[(0, 1)] - inertia[(1, 0)]).abs() < 1e-12);
        assert!(inertia[(0, 0)] > 0.0 && inertia[(1, 1)] > 0.0);
        assert!(inertia.cholesky().is_ok(), "{configuration:?}");
    }
}

#[test]
fn armature_lands_only_on_the_diagonal() {
    let with_rotors = double_pendulum::<f64>();
    let without = double_pendulum_plain::<f64>();
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = pair(configuration);
        let difference = with_rotors.joint_space_inertia_at(&position).unwrap()
            - without.joint_space_inertia_at(&position).unwrap();
        assert!((difference[(0, 0)] - 0.05).abs() < 1e-12);
        assert!((difference[(1, 1)] - 0.03).abs() < 1e-12);
        assert!(difference[(0, 1)].abs() < 1e-12 && difference[(1, 0)].abs() < 1e-12);
    }
}

#[test]
fn armature_scales_the_acceleration_term() {
    let with_rotors = double_pendulum::<f64>();
    let without = double_pendulum_plain::<f64>();
    let acceleration = pair([1.3, -0.8]);
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = pair(configuration);
        let difference = with_rotors
            .inverse_dynamics_at(&position, &Vector::zeros(), &acceleration)
            .unwrap()
            - without
                .inverse_dynamics_at(&position, &Vector::zeros(), &acceleration)
                .unwrap();
        assert!((difference[0] - 0.05 * 1.3).abs() < 1e-12);
        assert!((difference[1] - 0.03 * -0.8).abs() < 1e-12);
    }
}

#[test]
fn friction_superposes_on_the_rigid_body_torque() {
    let with_friction = double_pendulum::<f64>();
    let without = double_pendulum_plain::<f64>();
    let velocity = pair([1.4, -0.6]);
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = pair(configuration);
        let difference = with_friction.bias_torque_at(&position, &velocity).unwrap()
            - without.bias_torque_at(&position, &velocity).unwrap();
        assert!((difference[0] - (0.7 * 1.4 + 0.2)).abs() < 1e-12);
        assert!((difference[1] - (0.4 * -0.6 - 0.1)).abs() < 1e-12);
    }
}

/// `Numeric::signum` is overridden for `f64` to the primitive, where `0.0.signum()` is `1.0`;
/// using it here would break a standing joint away under its full friction loss.
#[test]
fn friction_vanishes_at_zero_rate() {
    let with_friction = double_pendulum::<f64>();
    let without = double_pendulum_plain::<f64>();
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = pair(configuration);
        let loaded = with_friction
            .inverse_dynamics_at(&position, &Vector::zeros(), &Vector::zeros())
            .unwrap();
        let plain = without
            .inverse_dynamics_at(&position, &Vector::zeros(), &Vector::zeros())
            .unwrap();
        assert!((loaded[0] - plain[0]).abs() < 1e-12);
        assert!((loaded[1] - plain[1]).abs() < 1e-12);
    }
}

/// A two-joint arm with a welded tool frame, and the two-joint arm carrying the tool's mass folded
/// into its own link. `MAX_CONFIG` is 3 in both so the vectors line up.
fn welded_tool_arm(tool: Option<SpatialInertia<f64>>) -> ArticulatedBody<3, 3, f64> {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let link = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
    let mut tree = KinematicTree::<3, 3, f64>::new();
    tree.push(Joint::revolute(axis, SE3::identity()), JointParent::World)
        .unwrap();
    tree.push(Joint::revolute(axis, link), JointParent::Joint(0))
        .unwrap();
    tree.push(Joint::fixed(SE3::identity()), JointParent::Joint(1))
        .unwrap();

    let inertia = link_inertia::<f64>(1.5, 0.02);
    ArticulatedBody::new(tree, &[Some(inertia), Some(inertia), tool], earth_gravity()).unwrap()
}

/// The same two joints with no third slot, the second link carrying `second` directly.
fn plain_two_joint_arm(second: SpatialInertia<f64>) -> ArticulatedBody<3, 3, f64> {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let link = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
    let mut tree = KinematicTree::<3, 3, f64>::new();
    tree.push(Joint::revolute(axis, SE3::identity()), JointParent::World)
        .unwrap();
    tree.push(Joint::revolute(axis, link), JointParent::Joint(0))
        .unwrap();

    ArticulatedBody::new(
        tree,
        &[Some(link_inertia::<f64>(1.5, 0.02)), Some(second)],
        earth_gravity(),
    )
    .unwrap()
}

#[test]
fn welded_slot_carries_its_mass() {
    // The weld sits at the identity relative to joint 1, so the tool's inertia needs no
    // re-referencing before it is folded in.
    let tool = SpatialInertia::new(
        0.4,
        Vector::new([0.2, 0.0, 0.0]),
        Matrix::from_diagonal([0.001, 0.001, 0.001]),
    )
    .unwrap();
    let welded = welded_tool_arm(Some(tool));
    let folded = plain_two_joint_arm(link_inertia::<f64>(1.5, 0.02).combined(tool));

    let velocity = Vector::new([0.9, -0.5, 0.0]);
    let acceleration = Vector::new([0.4, 1.1, 0.0]);
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = Vector::new([configuration[0], configuration[1], 0.0]);
        let carried = welded
            .inverse_dynamics_at(&position, &velocity, &acceleration)
            .unwrap();
        let want = folded
            .inverse_dynamics_at(&position, &velocity, &acceleration)
            .unwrap();
        assert!((carried[0] - want[0]).abs() < 1e-12, "{configuration:?}");
        assert!((carried[1] - want[1]).abs() < 1e-12, "{configuration:?}");
    }
}

#[test]
fn massless_slot_contributes_nothing() {
    let welded = welded_tool_arm(None);
    let plain = plain_two_joint_arm(link_inertia::<f64>(1.5, 0.02));

    let velocity = Vector::new([0.9, -0.5, 0.0]);
    let acceleration = Vector::new([0.4, 1.1, 0.0]);
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = Vector::new([configuration[0], configuration[1], 0.0]);
        let carried = welded
            .inverse_dynamics_at(&position, &velocity, &acceleration)
            .unwrap();
        let want = plain
            .inverse_dynamics_at(&position, &velocity, &acceleration)
            .unwrap();
        assert!((carried[0] - want[0]).abs() < 1e-12, "{configuration:?}");
        assert!((carried[1] - want[1]).abs() < 1e-12, "{configuration:?}");
    }
}

#[test]
fn dual_derivative_matches_joint_space_inertia() {
    let body = single_pendulum::<Dual<f64>>();
    let torque = body
        .inverse_dynamics_at(
            &Vector::new([Dual::constant(0.4)]),
            &Vector::new([Dual::constant(0.0)]),
            &Vector::new([Dual::variable(0.0)]),
        )
        .unwrap();
    assert!((torque[0].deriv - 0.51).abs() < 1e-12);
}

#[test]
fn dual_derivative_of_damping_matches_the_coefficient() {
    let body = single_pendulum_with::<Dual<f64>>(
        Dual::constant(0.0),
        Dual::constant(0.9),
        Dual::constant(0.0),
    );
    let torque = body
        .inverse_dynamics_at(
            &Vector::new([Dual::constant(0.4)]),
            &Vector::new([Dual::variable(2.0)]),
            &Vector::new([Dual::constant(0.0)]),
        )
        .unwrap();
    assert!((torque[0].deriv - 0.9).abs() < 1e-12);
}

#[test]
fn rejects_a_floating_base() {
    let mut tree = KinematicTree::<2, 8, f64>::new();
    tree.push(Joint::floating(SE3::identity()), JointParent::World)
        .unwrap();
    tree.push(
        Joint::revolute(Vector::new([0.0, 1.0, 0.0]), SE3::identity()),
        JointParent::Joint(0),
    )
    .unwrap();

    let inertia = link_inertia::<f64>(1.0, 0.01);
    assert_eq!(
        ArticulatedBody::<2, 8, f64>::new(tree, &[Some(inertia), Some(inertia)], earth_gravity()),
        Err(DynamicsError::FloatingBaseUnsupported)
    );
}

#[test]
fn rejects_an_inertia_count_mismatch() {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let tree = KinematicTree::<2, 2, f64>::try_from_joints(
        &[
            Joint::revolute(axis, SE3::identity()),
            Joint::revolute(axis, SE3::identity()),
        ],
        &[JointParent::World, JointParent::Joint(0)],
    )
    .unwrap();

    assert_eq!(
        ArticulatedBody::<2, 2, f64>::new(
            tree,
            &[Some(link_inertia::<f64>(1.0, 0.01))],
            earth_gravity(),
        ),
        Err(DynamicsError::InertiaCountMismatch)
    );
}

#[test]
fn rejects_non_finite_gravity() {
    let tree = KinematicTree::<1, 1, f64>::try_from_joints(
        &[Joint::revolute(
            Vector::new([0.0, 1.0, 0.0]),
            SE3::identity(),
        )],
        &[JointParent::World],
    )
    .unwrap();

    assert_eq!(
        ArticulatedBody::<1, 1, f64>::new(
            tree,
            &[Some(link_inertia::<f64>(1.0, 0.01))],
            Vector::new([0.0, 0.0, f64::NAN]),
        ),
        Err(DynamicsError::NonFinite)
    );
}

#[test]
fn rejects_a_non_finite_rate() {
    let body = single_pendulum::<f64>();
    assert_eq!(
        body.inverse_dynamics_at(
            &Vector::zeros(),
            &Vector::new([f64::INFINITY]),
            &Vector::zeros(),
        ),
        Err(DynamicsError::NonFinite)
    );
}

#[test]
fn rejects_poses_from_another_model() {
    let one_joint = KinematicTree::<2, 2, f64>::try_from_joints(
        &[Joint::revolute(
            Vector::new([0.0, 1.0, 0.0]),
            SE3::identity(),
        )],
        &[JointParent::World],
    )
    .unwrap();
    let state = one_joint.forward_kinematics(&Vector::zeros()).unwrap();

    let body = double_pendulum::<f64>();
    assert_eq!(
        body.inverse_dynamics(&state, &Vector::zeros(), &Vector::zeros()),
        Err(DynamicsError::StateShapeMismatch)
    );
}

#[test]
fn forward_dynamics_inverts_inverse_dynamics() {
    let body = double_pendulum::<f64>();
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = pair(configuration);
        for rates in SAMPLE_CONFIGURATIONS {
            let velocity = pair(rates);
            for changes in SAMPLE_CONFIGURATIONS {
                let acceleration = pair(changes);
                let torque = body
                    .inverse_dynamics_at(&position, &velocity, &acceleration)
                    .unwrap();
                let recovered = body
                    .forward_dynamics_at(&position, &velocity, &torque)
                    .unwrap();
                for index in 0..2 {
                    let want = acceleration[index];
                    assert!(
                        (recovered[index] - want).abs() <= 1e-10 * want.abs().max(1.0),
                        "{configuration:?} {rates:?} {changes:?} entry {index}: \
                         {} vs {want}",
                        recovered[index]
                    );
                }
            }
        }
    }
}

#[test]
fn forward_dynamics_with_workspace_matches_the_stack_local_form() {
    let body = double_pendulum::<f64>();
    let mut workspace = DynamicsWorkspace::<2, 2, f64>::new();
    let torque = pair([3.0, -1.2]);
    for configuration in SAMPLE_CONFIGURATIONS {
        let state = body
            .tree()
            .forward_kinematics(&pair(configuration))
            .unwrap();
        let velocity = pair([0.8, -0.3]);
        let stack_local = body.forward_dynamics(&state, &velocity, &torque).unwrap();
        let borrowed = body
            .forward_dynamics_with_workspace(&state, &velocity, &torque, &mut workspace)
            .unwrap();
        assert_eq!(stack_local[0], borrowed[0]);
        assert_eq!(stack_local[1], borrowed[1]);
    }
}

#[test]
fn forward_dynamics_matches_the_inertia_matrix_route() {
    let body = double_pendulum::<f64>();
    let velocity = pair([1.1, -0.7]);
    let torque = pair([2.4, -0.9]);
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = pair(configuration);
        let recursive = body
            .forward_dynamics_at(&position, &velocity, &torque)
            .unwrap();

        let bias = body.bias_torque_at(&position, &velocity).unwrap();
        let solved = body
            .joint_space_inertia_at(&position)
            .unwrap()
            .cholesky()
            .unwrap()
            .solve(torque - bias);

        for index in 0..2 {
            assert!(
                (recursive[index] - solved[index]).abs() <= 1e-10 * solved[index].abs().max(1.0),
                "{configuration:?} entry {index}: {} vs {}",
                recursive[index],
                solved[index]
            );
        }
    }
}

/// `[q; q̇]` as the four numbers `Rk4` carries, and back.
fn swing_rate(body: &ArticulatedBody<2, 2, f64>, state: &Vector<4, f64>) -> Vector<4, f64> {
    let position = pair([state[0], state[1]]);
    let velocity = pair([state[2], state[3]]);
    let acceleration = body
        .forward_dynamics_at(&position, &velocity, &Vector::zeros())
        .unwrap();
    Vector::new([velocity[0], velocity[1], acceleration[0], acceleration[1]])
}

fn total_energy(body: &ArticulatedBody<2, 2, f64>, state: &Vector<4, f64>) -> f64 {
    let solved = body
        .tree()
        .forward_kinematics(&pair([state[0], state[1]]))
        .unwrap();
    let velocity = pair([state[2], state[3]]);
    body.kinetic_energy(&solved, &velocity).unwrap() + body.potential_energy(&solved).unwrap()
}

#[test]
fn energy_is_held_along_a_free_swing() {
    let body = double_pendulum_plain::<f64>();
    let start = Vector::new([1.2, -0.5, 0.0, 0.0]);
    let reference = total_energy(&body, &start);

    let mut worst_drift = 0.0_f64;
    let mut step_count = 0_usize;
    let end = Rk4::integrate(
        &|_, state: &Vector<4, f64>| swing_rate(&body, state),
        0.0,
        &start,
        1e-4,
        10_000,
        |_, state| {
            step_count += 1;
            if step_count.is_multiple_of(1_000) {
                let drift = (total_energy(&body, state) - reference).abs() / reference.abs();
                worst_drift = worst_drift.max(drift);
            }
        },
    );

    let final_drift = (total_energy(&body, &end) - reference).abs() / reference.abs();
    assert!(worst_drift < 1e-6, "worst sampled drift {worst_drift}");
    assert!(final_drift < 1e-6, "drift at one second {final_drift}");
}

#[test]
fn energy_falls_when_friction_is_present() {
    let body = double_pendulum::<f64>();
    let start = Vector::new([1.2, -0.5, 0.0, 0.0]);
    let reference = total_energy(&body, &start);

    let end = Rk4::integrate(
        &|_, state: &Vector<4, f64>| swing_rate(&body, state),
        0.0,
        &start,
        1e-4,
        10_000,
        |_, _| {},
    );

    assert!(total_energy(&body, &end) < reference);
}

#[test]
fn potential_energy_matches_the_hand_computation() {
    let body = single_pendulum::<f64>();
    for angle in [0.0, FRAC_PI_2] {
        let state = body
            .tree()
            .forward_kinematics(&Vector::new([angle]))
            .unwrap();
        // A turn of q about +y carries the balance point to (L cos q, 0, −L sin q), so
        // `V = m·g·z` falls as the link swings down.
        let want = 2.0 * GRAVITY * 0.5 * angle.sin();
        assert!((body.potential_energy(&state).unwrap() - want).abs() < 1e-12);
    }
}

#[test]
fn kinetic_energy_matches_the_inertia_matrix_form() {
    let body = double_pendulum_plain::<f64>();
    let velocity = pair([1.1, -0.7]);
    for configuration in SAMPLE_CONFIGURATIONS {
        let position = pair(configuration);
        let state = body.tree().forward_kinematics(&position).unwrap();
        let got = body.kinetic_energy(&state, &velocity).unwrap();
        let inertia = body.joint_space_inertia_at(&position).unwrap();
        let want = 0.5 * velocity.dot(inertia * velocity);
        assert!(
            (got - want).abs() <= 1e-10 * want.abs().max(1.0),
            "{configuration:?}: {got} vs {want}"
        );
    }
}

#[test]
fn dual_derivative_of_acceleration_matches_the_inverse_inertia() {
    let body = double_pendulum::<Dual<f64>>();
    let position = Vector::new([Dual::constant(0.4), Dual::constant(-0.9)]);
    let velocity = Vector::new([Dual::constant(0.7), Dual::constant(-0.2)]);
    let acceleration = body
        .forward_dynamics_at(
            &position,
            &velocity,
            &Vector::new([Dual::variable(2.4), Dual::constant(-0.9)]),
        )
        .unwrap();

    let plain = double_pendulum::<f64>();
    let inverse_inertia = plain
        .joint_space_inertia_at(&pair([0.4, -0.9]))
        .unwrap()
        .cholesky()
        .unwrap()
        .solve(Vector::new([1.0, 0.0]));

    for index in 0..2 {
        let want = inverse_inertia[index];
        assert!(
            (acceleration[index].deriv - want).abs() <= 1e-10 * want.abs().max(1.0),
            "entry {index}: {} vs {want}",
            acceleration[index].deriv
        );
    }
}

#[test]
fn dual_derivative_of_acceleration_matches_finite_differences() {
    let dual_body = double_pendulum::<Dual<f64>>();
    let plain = double_pendulum::<f64>();
    let torque = pair([2.4, -0.9]);
    let base_position = [0.4_f64, -0.9];
    let base_velocity = [0.7_f64, -0.2];
    let step = 1e-6;

    let by_position = dual_body
        .forward_dynamics_at(
            &Vector::new([
                Dual::variable(base_position[0]),
                Dual::constant(base_position[1]),
            ]),
            &Vector::new([
                Dual::constant(base_velocity[0]),
                Dual::constant(base_velocity[1]),
            ]),
            &Vector::new([Dual::constant(torque[0]), Dual::constant(torque[1])]),
        )
        .unwrap()[0]
        .deriv;

    let acceleration_at = |position: [f64; 2], velocity: [f64; 2]| {
        plain
            .forward_dynamics_at(&pair(position), &pair(velocity), &torque)
            .unwrap()[0]
    };
    let position_difference =
        (acceleration_at([base_position[0] + step, base_position[1]], base_velocity)
            - acceleration_at([base_position[0] - step, base_position[1]], base_velocity))
            / (2.0 * step);
    assert!(
        (by_position - position_difference).abs() <= 1e-6 * position_difference.abs().max(1.0),
        "{by_position} vs {position_difference}"
    );

    let by_rate = dual_body
        .forward_dynamics_at(
            &Vector::new([
                Dual::constant(base_position[0]),
                Dual::constant(base_position[1]),
            ]),
            &Vector::new([
                Dual::variable(base_velocity[0]),
                Dual::constant(base_velocity[1]),
            ]),
            &Vector::new([Dual::constant(torque[0]), Dual::constant(torque[1])]),
        )
        .unwrap()[0]
        .deriv;

    let rate_difference =
        (acceleration_at(base_position, [base_velocity[0] + step, base_velocity[1]])
            - acceleration_at(base_position, [base_velocity[0] - step, base_velocity[1]]))
            / (2.0 * step);
    assert!(
        (by_rate - rate_difference).abs() <= 1e-6 * rate_difference.abs().max(1.0),
        "{by_rate} vs {rate_difference}"
    );
}

#[test]
fn rejects_a_degree_of_freedom_with_no_inertia_behind_it() {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let bare = KinematicTree::<1, 1, f64>::try_from_joints(
        &[Joint::revolute(axis, SE3::identity())],
        &[JointParent::World],
    )
    .unwrap();
    let massless = ArticulatedBody::new(bare, &[None], earth_gravity()).unwrap();
    assert_eq!(
        massless.forward_dynamics_at(&Vector::zeros(), &Vector::zeros(), &Vector::new([1.0])),
        Err(DynamicsError::NonPositiveArticulatedInertia)
    );

    let with_rotor = KinematicTree::<1, 1, f64>::try_from_joints(
        &[Joint::revolute(axis, SE3::identity()).with_armature(0.02)],
        &[JointParent::World],
    )
    .unwrap();
    let driven = ArticulatedBody::new(with_rotor, &[None], earth_gravity()).unwrap();
    let acceleration = driven
        .forward_dynamics_at(&Vector::zeros(), &Vector::zeros(), &Vector::new([1.0]))
        .unwrap();
    assert!((acceleration[0] - 1.0 / 0.02).abs() < 1e-12);
}

#[test]
fn workspace_is_reusable_across_calls() {
    let body = double_pendulum::<f64>();
    let torque = pair([2.4, -0.9]);
    let mut shared = DynamicsWorkspace::<2, 2, f64>::new();

    for configuration in SAMPLE_CONFIGURATIONS {
        let state = body
            .tree()
            .forward_kinematics(&pair(configuration))
            .unwrap();
        let velocity = pair([configuration[1], configuration[0]]);
        let mut fresh = DynamicsWorkspace::<2, 2, f64>::new();

        let reused = body
            .forward_dynamics_with_workspace(&state, &velocity, &torque, &mut shared)
            .unwrap();
        let clean = body
            .forward_dynamics_with_workspace(&state, &velocity, &torque, &mut fresh)
            .unwrap();
        assert_eq!(reused[0], clean[0], "{configuration:?}");
        assert_eq!(reused[1], clean[1], "{configuration:?}");
    }
}
