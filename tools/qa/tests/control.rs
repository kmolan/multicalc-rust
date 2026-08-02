#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the two matrix equations behind optimal linear feedback against scipy, and the geometric
//! attitude law against the same law evaluated in numpy on scipy-built rotations.

use multicalc::SO3;
use multicalc::control::{GeometricAttitudeController, Lqr};
use multicalc::linear_algebra::{solve_discrete_lyapunov, solve_discrete_riccati};
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
            other => panic!("no check registered for control fixture {other}"),
        }
        checked += 1;
    }
    assert_eq!(
        checked, 10,
        "expected ten control fixtures, found {checked}"
    );
}

fn check_riccati<const N: usize, const M: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let b = to_matrix::<N, M>(&fixture.inputs["B"]);
    let q = to_matrix::<N, N>(&fixture.inputs["Q"]);
    let r = to_matrix::<M, M>(&fixture.inputs["R"]);
    let tolerance = fixture.tolerances.f64;

    let cost_to_go = solve_discrete_riccati(a, b, q, r).unwrap();
    assert_matrix(
        &cost_to_go,
        &fixture.expected["P"],
        tolerance,
        &format!("{}: cost-to-go", fixture.case),
    );

    // The gain the crate's own controller produces, from the same inputs.
    let controller = Lqr::<N, M>::new(a, b, q, r).unwrap();
    assert_matrix(
        &controller.gain(),
        &fixture.expected["K"],
        tolerance,
        &format!("{}: gain", fixture.case),
    );
}

fn check_lyapunov<const N: usize>(fixture: &Fixture) {
    let a = to_matrix::<N, N>(&fixture.inputs["A"]);
    let q = to_matrix::<N, N>(&fixture.inputs["Q"]);
    let certificate = solve_discrete_lyapunov(a, q).unwrap();
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
