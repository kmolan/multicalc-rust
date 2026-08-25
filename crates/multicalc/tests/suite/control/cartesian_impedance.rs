//! Cartesian-impedance tests: what is left at zero error, that the wrench really goes through `Jᵀ`,
//! that the frame choice is not silently ignored, that the posture term leaves the tool alone, and
//! the refusals.

use multicalc::control::{CartesianImpedanceController, CartesianReference};
use multicalc::dynamics::ArticulatedBody;
use multicalc::error::{ControlError, KinematicsError};
use multicalc::kinematics::{JacobianFrame, Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::spatial::{SE3, SO3, SpatialInertia, Twist};

const GRAVITY: f64 = -9.81;
const TOOL: usize = 2;

/// Three revolute joints about `+y`, each 0.3 m past its parent, each link 1 kg balancing 0.15 m
/// out. The tool is the last joint's frame.
fn planar_arm() -> ArticulatedBody<3, 3, f64> {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let offset = SE3::from_parts(SO3::identity(), Vector::new([0.3, 0.0, 0.0]));
    let tree = KinematicTree::<3, 3, f64>::try_from_joints(
        &[
            Joint::revolute(axis, SE3::identity()),
            Joint::revolute(axis, offset),
            Joint::revolute(axis, offset),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
        ],
    )
    .unwrap();
    let inertia = SpatialInertia::new(
        1.0,
        Vector::new([0.15, 0.0, 0.0]),
        Matrix::from_diagonal([0.01, 0.01, 0.01]),
    )
    .unwrap();
    ArticulatedBody::new(
        tree,
        &[Some(inertia), Some(inertia), Some(inertia)],
        Vector::new([0.0, 0.0, GRAVITY]),
    )
    .unwrap()
}

/// Four revolute joints about `+y`, each 0.3 m past its parent — one more than a planar task needs,
/// so the Jacobian has a null space to project into.
fn redundant_planar_arm() -> ArticulatedBody<4, 4, f64> {
    let axis = Vector::new([0.0, 1.0, 0.0]);
    let offset = SE3::from_parts(SO3::identity(), Vector::new([0.3, 0.0, 0.0]));
    let tree = KinematicTree::<4, 4, f64>::try_from_joints(
        &[
            Joint::revolute(axis, SE3::identity()),
            Joint::revolute(axis, offset),
            Joint::revolute(axis, offset),
            Joint::revolute(axis, offset),
        ],
        &[
            JointParent::World,
            JointParent::Joint(0),
            JointParent::Joint(1),
            JointParent::Joint(2),
        ],
    )
    .unwrap();
    let inertia = SpatialInertia::new(
        1.0,
        Vector::new([0.15, 0.0, 0.0]),
        Matrix::from_diagonal([0.01, 0.01, 0.01]),
    )
    .unwrap();
    ArticulatedBody::new(
        tree,
        &[Some(inertia), Some(inertia), Some(inertia), Some(inertia)],
        Vector::new([0.0, 0.0, GRAVITY]),
    )
    .unwrap()
}

fn stiffness() -> Vector<6, f64> {
    Vector::new([800.0, 800.0, 800.0, 40.0, 40.0, 40.0])
}

/// Stiff laterally, soft along the tool's own `x`: an isotropic stiffness commutes with the tool
/// rotation, so only an anisotropic one can tell the two frames apart at all.
fn anisotropic_stiffness() -> Vector<6, f64> {
    Vector::new([200.0, 800.0, 1600.0, 10.0, 40.0, 80.0])
}

fn damping() -> Vector<6, f64> {
    stiffness().map(|entry: f64| 2.0 * entry.sqrt())
}

#[test]
fn zero_error_leaves_only_the_bias_term() {
    let body = planar_arm();
    let controller = CartesianImpedanceController::new(stiffness(), damping(), TOOL).unwrap();

    let position = Vector::new([0.2, -0.4, 0.3]);
    let velocity = Vector::zeros();
    let state = body.tree().forward_kinematics(&position).unwrap();
    let reference = CartesianReference::at_rest(state.pose(TOOL).unwrap());

    let torque = controller
        .torque(&body, &state, &position, &velocity, &reference)
        .unwrap();
    let bias = body.bias_torque(&state, &velocity).unwrap();
    for joint in 0..3 {
        assert!(
            (torque[joint] - bias[joint]).abs() < 1e-12,
            "joint {joint}: {} vs {}",
            torque[joint],
            bias[joint]
        );
    }
}

#[test]
fn wrench_maps_through_the_jacobian_transpose() {
    let body = planar_arm();
    let controller = CartesianImpedanceController::new(stiffness(), damping(), TOOL).unwrap();

    let position = Vector::new([0.2, -0.4, 0.3]);
    let velocity = Vector::new([0.5, -0.3, 0.9]);
    let state = body.tree().forward_kinematics(&position).unwrap();
    let here = state.pose(TOOL).unwrap();
    let target = SE3::from_parts(
        here.rotation()
            .compose(SO3::exp(Vector::new([0.0, 0.05, 0.0]))),
        here.translation() + Vector::new([0.01, -0.02, 0.015]),
    );
    let reference =
        CartesianReference::new(target, Twist::from_array([0.1, 0.0, -0.05, 0.0, 0.2, 0.0]));

    let torque = controller
        .torque(&body, &state, &position, &velocity, &reference)
        .unwrap();
    let bias = body.bias_torque(&state, &velocity).unwrap();

    // The wrench, rebuilt here from the pose error and the twist error rather than taken from the
    // controller.
    let jacobian = body
        .tree()
        .geometric_jacobian(&state, TOOL, JacobianFrame::Body)
        .unwrap();
    let error = here.inverse().compose(target).log();
    let twist_error = reference.twist.to_vector() - jacobian.tool_twist(&velocity).to_vector();
    let wrench = Vector::<6, f64>::from_fn(|axis| {
        stiffness()[axis] * error[axis] + damping()[axis] * twist_error[axis]
    });
    let want = jacobian.matrix().transpose() * wrench;

    for joint in 0..3 {
        assert!(
            (torque[joint] - bias[joint] - want[joint]).abs() < 1e-12,
            "joint {joint}: {} vs {}",
            torque[joint] - bias[joint],
            want[joint]
        );
    }
}

#[test]
fn body_and_world_frames_agree_on_an_identity_tool_rotation() {
    let body = planar_arm();
    let in_body =
        CartesianImpedanceController::new(anisotropic_stiffness(), damping(), TOOL).unwrap();
    let in_world = in_body.with_frame(JacobianFrame::World);

    let velocity = Vector::new([0.5, -0.3, 0.9]);
    let offset = Vector::new([0.01, -0.02, 0.015]);

    // The three parallel hinges sum to zero, so the tool frame is world-aligned.
    let aligned = Vector::new([0.4, -0.9, 0.5]);
    // And here they do not.
    let turned = Vector::new([0.2, -0.4, 0.3]);

    for (position, agree) in [(aligned, true), (turned, false)] {
        let state = body.tree().forward_kinematics(&position).unwrap();
        let here = state.pose(TOOL).unwrap();
        let target = SE3::from_parts(here.rotation(), here.translation() + offset);
        let reference = CartesianReference::at_rest(target);

        let from_body = in_body
            .torque(&body, &state, &position, &velocity, &reference)
            .unwrap();
        let from_world = in_world
            .torque(&body, &state, &position, &velocity, &reference)
            .unwrap();
        let gap = (from_body - from_world).norm();
        if agree {
            assert!(gap < 1e-12, "world-aligned tool: {gap}");
        } else {
            assert!(gap > 1e-6, "turned tool: {gap}");
        }
    }
}

#[test]
fn null_space_term_does_not_move_the_tool() {
    let body = redundant_planar_arm();
    let tool = 3;
    let plain = CartesianImpedanceController::new(stiffness(), damping(), tool).unwrap();
    let with_posture = plain
        .with_null_space_posture(
            Vector::new([0.0, 0.4, -0.4, 0.2]),
            25.0,
            5.0,
            Vector::new([1.0, 1.0, 1.0, 1.0]),
            0.0,
        )
        .unwrap();

    let position = Vector::new([0.2, -0.4, 0.3, 0.6]);
    let velocity = Vector::new([0.5, -0.3, 0.9, -0.1]);
    let state = body.tree().forward_kinematics(&position).unwrap();
    let here = state.pose(tool).unwrap();
    let reference = CartesianReference::at_rest(SE3::from_parts(
        here.rotation(),
        here.translation() + Vector::new([0.01, -0.02, 0.015]),
    ));

    let difference = with_posture
        .torque(&body, &state, &position, &velocity, &reference)
        .unwrap()
        - plain
            .torque(&body, &state, &position, &velocity, &reference)
            .unwrap();
    assert!(difference.norm() > 1e-6, "{}", difference.norm());

    let jacobian = body
        .tree()
        .geometric_jacobian(&state, tool, JacobianFrame::Body)
        .unwrap();
    let leaked = jacobian.tool_twist(&difference).to_vector().norm();
    assert!(leaked < 1e-9, "{leaked}");
}

#[test]
fn bad_tool_index_is_reported() {
    let body = planar_arm();
    let controller = CartesianImpedanceController::new(stiffness(), damping(), 7).unwrap();
    let position = Vector::new([0.2, -0.4, 0.3]);
    let reference = CartesianReference::at_rest(SE3::identity());

    assert_eq!(
        controller.torque_at(&body, &position, &Vector::zeros(), &reference),
        Err(ControlError::Kinematics(
            KinematicsError::ToolIndexOutOfRange
        ))
    );
}

#[test]
fn stiffness_is_validated() {
    let damped = damping();
    assert_eq!(
        CartesianImpedanceController::<3, f64>::new(
            Vector::new([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            damped,
            TOOL
        ),
        Err(ControlError::NegativeGain)
    );
    assert_eq!(
        CartesianImpedanceController::<3, f64>::new(
            Vector::new([f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0]),
            damped,
            TOOL
        ),
        Err(ControlError::NonFinite)
    );
    let controller =
        CartesianImpedanceController::<3, f64>::new(stiffness(), damped, TOOL).unwrap();
    assert_eq!(
        controller.with_null_space_posture(
            Vector::zeros(),
            -1.0,
            1.0,
            Vector::new([1.0, 1.0, 1.0]),
            0.0
        ),
        Err(ControlError::NegativeGain)
    );
}
