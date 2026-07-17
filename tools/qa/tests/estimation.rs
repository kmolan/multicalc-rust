#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the linear Kalman filter against filterpy goldens.

use multicalc::estimation::KalmanFilter;
use multicalc::linear_algebra::Vector;
use multicalc_qa::load::*;
use multicalc_qa::schema::*;

fn build_filter<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize>(
    fx: &Fixture,
) -> KalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION> {
    KalmanFilter::new(
        to_vector::<STATE_DIMENSION>(&fx.inputs["initial_state"]),
        to_matrix::<STATE_DIMENSION, STATE_DIMENSION>(&fx.inputs["initial_covariance"]),
        to_matrix::<STATE_DIMENSION, STATE_DIMENSION>(&fx.inputs["state_transition"]),
        to_matrix::<MEASUREMENT_DIMENSION, STATE_DIMENSION>(&fx.inputs["measurement_model"]),
        to_matrix::<STATE_DIMENSION, STATE_DIMENSION>(&fx.inputs["process_noise"]),
        to_matrix::<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION>(&fx.inputs["measurement_noise"]),
    )
}

fn assert_final_estimate<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize>(
    filter: &KalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION>,
    fx: &Fixture,
) {
    let t = fx.tolerances.get("f64", "host");
    assert_vector(&filter.state(), &fx.expected["state"], t, "state");
    assert_matrix(
        &filter.covariance(),
        &fx.expected["covariance"],
        t,
        "covariance",
    );
    assert_vector(
        &filter.innovation(),
        &fx.expected["innovation"],
        t,
        "innovation",
    );
    assert_matrix(
        &filter.innovation_covariance(),
        &fx.expected["innovation_covariance"],
        t,
        "innovation_covariance",
    );
}

fn run_kalman_filter<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize>(
    fx: &Fixture,
) {
    let mut filter = build_filter::<STATE_DIMENSION, MEASUREMENT_DIMENSION>(fx);
    let (steps, _, measurements) = fx.inputs["measurements"].as_matrix();
    for step in 0..steps {
        filter.predict();
        let measurement = Vector::from_fn(|i| measurements[step * MEASUREMENT_DIMENSION + i]);
        filter.update(measurement).unwrap();
    }
    assert_final_estimate(&filter, fx);
}

fn run_kalman_filter_with_control<
    const STATE_DIMENSION: usize,
    const MEASUREMENT_DIMENSION: usize,
    const CONTROL_DIMENSION: usize,
>(
    fx: &Fixture,
) {
    let mut filter = build_filter::<STATE_DIMENSION, MEASUREMENT_DIMENSION>(fx);
    let control_model =
        to_matrix::<STATE_DIMENSION, CONTROL_DIMENSION>(&fx.inputs["control_model"]);
    let (steps, _, measurements) = fx.inputs["measurements"].as_matrix();
    let (_, _, controls) = fx.inputs["control_inputs"].as_matrix();
    for step in 0..steps {
        let control_input = Vector::from_fn(|i| controls[step * CONTROL_DIMENSION + i]);
        filter.predict_with_control(control_model, control_input);
        let measurement = Vector::from_fn(|i| measurements[step * MEASUREMENT_DIMENSION + i]);
        filter.update(measurement).unwrap();
    }
    assert_final_estimate(&filter, fx);
}

#[test]
fn kalman_filter_cases() {
    for fx in load_dir("fixtures/v1/estimation") {
        if fx.inputs["kind"].as_str() != "kalman_filter" {
            continue;
        }
        let (state_dimension, ..) = fx.inputs["state_transition"].as_matrix();
        let (measurement_dimension, ..) = fx.inputs["measurement_model"].as_matrix();
        match (state_dimension, measurement_dimension) {
            (2, 1) => run_kalman_filter::<2, 1>(&fx),
            (4, 2) => run_kalman_filter::<4, 2>(&fx),
            shape => panic!("unregistered kalman filter shape {shape:?}"),
        }
    }
}

#[test]
fn kalman_filter_with_control_cases() {
    for fx in load_dir("fixtures/v1/estimation") {
        if fx.inputs["kind"].as_str() != "kalman_filter_with_control" {
            continue;
        }
        let (state_dimension, ..) = fx.inputs["state_transition"].as_matrix();
        let (measurement_dimension, ..) = fx.inputs["measurement_model"].as_matrix();
        let (_, control_dimension, _) = fx.inputs["control_model"].as_matrix();
        match (state_dimension, measurement_dimension, control_dimension) {
            (2, 1, 1) => run_kalman_filter_with_control::<2, 1, 1>(&fx),
            shape => panic!("unregistered kalman filter control shape {shape:?}"),
        }
    }
}
