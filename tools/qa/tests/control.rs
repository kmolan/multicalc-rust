#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the two matrix equations behind optimal linear feedback against scipy, the geometric
//! attitude law against the same law evaluated in numpy on scipy-built rotations, and the
//! model-based arm laws against Pinocchio's model terms with the law re-evaluated in numpy.
//!
//! The arm fixtures carry the model — joint kinds, parents, origins, axes, anchors, masses, centres
//! of mass, rotational inertias, armature, damping and friction loss — so the model built here is
//! the model the oracle was given.

use multicalc::SO3;
use multicalc::control::{
    CartesianImpedanceController, CartesianReference, ComputedTorqueController,
    GeometricAttitudeController, JointImpedanceController, JointPdController, JointReference, Lqr,
};
use multicalc::kinematics::JacobianFrame;
use multicalc::linear_algebra::{
    Matrix4D, Vector, solve_discrete_lyapunov, solve_discrete_riccati,
};
use multicalc::spatial::{SE3, Twist};
use multicalc_qa::articulated::*;
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

#[test]
fn control_goldens() {
    let fixtures = load_dir("control");
    let mut checked = 0;
    for fixture in &fixtures {
        match fixture.case.as_str() {
            "riccati_double_integrator" => check_riccati::<2, 1>(fixture),
            "riccati_cart_pole" => check_riccati::<4, 1>(fixture),
            "riccati_quadrotor_hover" => check_riccati::<6, 3>(fixture),
            "riccati_mixed_5x2" => check_riccati::<5, 2>(fixture),
            "lyapunov_stable_3x3" => check_lyapunov::<3>(fixture),
            "lyapunov_stable_5x5" => check_lyapunov::<5>(fixture),
            "lyapunov_closed_loop_double_integrator" => check_lyapunov::<2>(fixture),
            "lyapunov_closed_loop_quadrotor_hover" => check_lyapunov::<6>(fixture),
            "geometric_attitude_general" | "geometric_attitude_near_target" => {
                check_geometric_attitude(fixture);
            }
            "computed_torque_double_pendulum" | "computed_torque_franka_panda" => {
                check_computed_torque(fixture);
            }
            "joint_impedance_franka_panda" => check_joint_impedance(fixture),
            "joint_pd_franka_panda" => check_joint_pd(fixture),
            "cartesian_impedance_franka_panda" => check_cartesian_impedance(fixture),
            other => panic!("no check registered for control fixture {other}"),
        }
        checked += 1;
    }
    assert_eq!(
        checked, 15,
        "expected fifteen control fixtures, found {checked}"
    );
}

fn check_riccati<const N: usize, const M: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let b = to_matrix::<N, M>(&fixture.inputs["B"]);
    let state_cost = to_matrix::<N, N>(&fixture.inputs["Q"]);
    let input_cost = to_matrix::<M, M>(&fixture.inputs["R"]);
    let tolerance = fixture.tolerances.f64;

    let cost_to_go = solve_discrete_riccati(a, b, state_cost, input_cost).unwrap();
    assert_matrix(
        &cost_to_go,
        &fixture.expected["P"],
        tolerance,
        &format!("{}: cost-to-go", fixture.case),
    );

    // The gain the crate's own controller produces, from the same inputs.
    let controller = Lqr::<N, M>::new(a, b, state_cost, input_cost).unwrap();
    assert_matrix(
        &controller.gain(),
        &fixture.expected["K"],
        tolerance,
        &format!("{}: gain", fixture.case),
    );
}

fn check_lyapunov<const N: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let state_cost = to_matrix::<N, N>(&fixture.inputs["Q"]);
    let certificate = solve_discrete_lyapunov(a, state_cost).unwrap();
    assert_matrix(
        &certificate,
        &fixture.expected["P"],
        fixture.tolerances.f64,
        &format!("{}: certificate", fixture.case),
    );
}

fn check_geometric_attitude(fixture: &Fixture) {
    let attitude = SO3::try_from_matrix(to_matrix::<3, 3>(&fixture.inputs["R"])).unwrap();
    let desired = SO3::try_from_matrix(to_matrix::<3, 3>(&fixture.inputs["R_desired"])).unwrap();
    let body_rate = to_vector::<3>(&fixture.inputs["omega"]);
    let desired_rate = to_vector::<3>(&fixture.inputs["omega_desired"]);
    let desired_rate_change = to_vector::<3>(&fixture.inputs["omega_desired_derivative"]);
    let inertia = to_matrix::<3, 3>(&fixture.inputs["inertia"]);
    let attitude_gain = fixture.inputs["attitude_gain"].as_scalar();
    let rate_gain = fixture.inputs["rate_gain"].as_scalar();
    let tolerance = fixture.tolerances.f64;

    let controller = GeometricAttitudeController::new(attitude_gain, rate_gain, inertia).unwrap();
    let error = GeometricAttitudeController::attitude_error(attitude, desired);
    assert_vector(
        &error,
        &fixture.expected["attitude_error"],
        tolerance,
        &format!("{}: attitude error", fixture.case),
    );

    let torque = controller.torque(
        attitude,
        body_rate,
        desired,
        desired_rate,
        desired_rate_change,
    );
    assert_vector(
        &torque,
        &fixture.expected["torque"],
        tolerance,
        &format!("{}: torque", fixture.case),
    );
}

/// The measured and desired readings a joint-space case carries, per state.
struct JointCase {
    joint_count: usize,
    state_count: usize,
    positions: Vec<f64>,
    velocities: Vec<f64>,
    desired_positions: Vec<f64>,
    desired_velocities: Vec<f64>,
    desired_accelerations: Vec<f64>,
    want_torques: Vec<f64>,
}

impl JointCase {
    fn read(fixture: &Fixture, joint_count: usize) -> Self {
        let (state_count, _, positions) = fixture.inputs["joint_positions"].as_matrix();
        let (_, _, velocities) = fixture.inputs["joint_velocities"].as_matrix();
        let (_, _, desired_positions) = fixture.inputs["desired_positions"].as_matrix();
        let (_, _, desired_velocities) = fixture.inputs["desired_velocities"].as_matrix();
        let (_, _, desired_accelerations) = fixture.inputs["desired_accelerations"].as_matrix();
        let (_, _, want_torques) = fixture.expected["torques"].as_matrix();
        JointCase {
            joint_count,
            state_count,
            positions,
            velocities,
            desired_positions,
            desired_velocities,
            desired_accelerations,
            want_torques,
        }
    }

    /// The measured configuration, measured rate and reference for state `state`.
    fn state(
        &self,
        state: usize,
    ) -> (
        Vector<MAX_JOINTS, f64>,
        Vector<MAX_JOINTS, f64>,
        JointReference<MAX_JOINTS, f64>,
    ) {
        (
            row_readings(&self.positions, self.joint_count, state),
            row_readings(&self.velocities, self.joint_count, state),
            JointReference::new(
                row_readings(&self.desired_positions, self.joint_count, state),
                row_readings(&self.desired_velocities, self.joint_count, state),
                row_readings(&self.desired_accelerations, self.joint_count, state),
            ),
        )
    }
}

/// Every torque entry of one state against its golden.
fn assert_torques(
    got: &Vector<MAX_JOINTS, f64>,
    want: &[f64],
    joint_count: usize,
    state: usize,
    tolerance: Tol,
    context: &str,
) {
    let start = state * joint_count;
    for index in 0..joint_count {
        assert!(
            close(got[index], want[start + index], tolerance),
            "{context}: torque[{index}] got {}, want {}",
            got[index],
            want[start + index]
        );
    }
}

/// A gain vector padded out to the model's width.
fn gains(fixture: &Fixture, name: &str) -> Vector<MAX_JOINTS, f64> {
    let values = fixture.inputs[name].as_vector();
    Vector::from_fn(|index| values.get(index).copied().unwrap_or(0.0))
}

fn check_computed_torque(fixture: &Fixture) {
    let case = fixture.case.as_str();
    let tolerance = fixture.tolerances.f64;
    let (body, joint_count) = body_from_fixture(fixture, true);
    let rows = JointCase::read(fixture, joint_count);

    let controller = ComputedTorqueController::new(
        gains(fixture, "position_gains"),
        gains(fixture, "velocity_gains"),
    )
    .unwrap();

    for state in 0..rows.state_count {
        let (position, velocity, reference) = rows.state(state);
        let torque = controller
            .torque_at(&body, &position, &velocity, &reference)
            .unwrap_or_else(|err| unreachable!("{case} state {state}: {err:?}"));
        assert_torques(
            &torque,
            &rows.want_torques,
            joint_count,
            state,
            tolerance,
            &format!("{case} state {state}"),
        );
    }
}

fn check_joint_impedance(fixture: &Fixture) {
    let case = fixture.case.as_str();
    let tolerance = fixture.tolerances.f64;
    let (body, joint_count) = body_from_fixture(fixture, true);
    let rows = JointCase::read(fixture, joint_count);

    let controller =
        JointImpedanceController::new(gains(fixture, "stiffness"), gains(fixture, "damping"))
            .unwrap();

    for state in 0..rows.state_count {
        let (position, velocity, reference) = rows.state(state);
        let torque = controller
            .torque_at(&body, &position, &velocity, &reference)
            .unwrap_or_else(|err| unreachable!("{case} state {state}: {err:?}"));
        assert_torques(
            &torque,
            &rows.want_torques,
            joint_count,
            state,
            tolerance,
            &format!("{case} state {state}"),
        );
    }
}

fn check_joint_pd(fixture: &Fixture) {
    let case = fixture.case.as_str();
    let tolerance = fixture.tolerances.f64;
    let (body, joint_count) = body_from_fixture(fixture, true);
    let rows = JointCase::read(fixture, joint_count);

    let controller = JointPdController::new(
        gains(fixture, "position_gains"),
        gains(fixture, "velocity_gains"),
    )
    .unwrap()
    .with_gravity_compensation(true);

    for state in 0..rows.state_count {
        let (position, velocity, reference) = rows.state(state);
        let torque = controller
            .torque_at(&body, &position, &velocity, &reference)
            .unwrap_or_else(|err| unreachable!("{case} state {state}: {err:?}"));
        assert_torques(
            &torque,
            &rows.want_torques,
            joint_count,
            state,
            tolerance,
            &format!("{case} state {state}"),
        );
    }
}

fn check_cartesian_impedance(fixture: &Fixture) {
    let case = fixture.case.as_str();
    let tolerance = fixture.tolerances.f64;
    let (body, joint_count) = body_from_fixture(fixture, true);
    let tool_index = fixture.inputs["tool_index"].as_int() as usize;
    assert_eq!(fixture.inputs["frame"].as_str(), "body", "{case}: frame");

    let (state_count, _, positions) = fixture.inputs["joint_positions"].as_matrix();
    let (_, _, velocities) = fixture.inputs["joint_velocities"].as_matrix();
    let (_, _, desired_poses) = fixture.inputs["desired_poses"].as_matrix();
    let (_, _, desired_twists) = fixture.inputs["desired_twists"].as_matrix();
    let (_, _, want_torques) = fixture.expected["torques"].as_matrix();
    let (_, _, want_pose_errors) = fixture.expected["pose_errors"].as_matrix();
    let (_, _, want_jacobians) = fixture.expected["jacobians"].as_matrix();

    let controller = CartesianImpedanceController::new(
        to_vector::<6>(&fixture.inputs["stiffness"]),
        to_vector::<6>(&fixture.inputs["damping"]),
        tool_index,
    )
    .unwrap();

    for state in 0..state_count {
        let context = format!("{case} state {state}");
        let position = row_readings(&positions, joint_count, state);
        let velocity = row_readings(&velocities, joint_count, state);

        let start = state * 16;
        let pose = SE3::try_from_matrix(Matrix4D::from_fn(|row, column| {
            desired_poses[start + row * 4 + column]
        }))
        .unwrap_or_else(|| unreachable!("{context}: desired pose is not a rigid transform"));
        let twist_start = state * 6;
        let twist = Twist::from_vector(Vector::from_fn(|axis| desired_twists[twist_start + axis]));
        let reference = CartesianReference::new(pose, twist);

        let solved = body
            .tree()
            .forward_kinematics(&position)
            .unwrap_or_else(|err| unreachable!("{context}: forward kinematics: {err:?}"));

        // The pose error and the Jacobian separately, so a frame-convention bug and a
        // wrench-composition bug cannot look alike.
        let error = controller
            .pose_error(&solved, &reference)
            .unwrap_or_else(|err| unreachable!("{context}: pose error: {err:?}"))
            .to_vector();
        for axis in 0..6 {
            assert!(
                close(error[axis], want_pose_errors[twist_start + axis], tolerance),
                "{context}: pose error[{axis}] got {}, want {}",
                error[axis],
                want_pose_errors[twist_start + axis]
            );
        }

        let jacobian = body
            .tree()
            .geometric_jacobian(&solved, tool_index, JacobianFrame::Body)
            .unwrap_or_else(|err| unreachable!("{context}: Jacobian: {err:?}"));
        let block = state * 6 * joint_count;
        for row in 0..6 {
            for column in 0..joint_count {
                let want = want_jacobians[block + row * joint_count + column];
                assert!(
                    close(jacobian.matrix()[(row, column)], want, tolerance),
                    "{context}: J[({row}, {column})] got {}, want {want}",
                    jacobian.matrix()[(row, column)]
                );
            }
        }

        let torque = controller
            .torque(&body, &solved, &position, &velocity, &reference)
            .unwrap_or_else(|err| unreachable!("{context}: torque: {err:?}"));
        assert_torques(
            &torque,
            &want_torques,
            joint_count,
            state,
            tolerance,
            &context,
        );
    }
}
