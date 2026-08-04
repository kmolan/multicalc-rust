#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Checks the linear Kalman filter against filterpy goldens.

use multicalc::estimation::{
    ErrorStateKalmanFilter, ExtendedKalmanFilter, ImuNoise, KalmanFilter, KalmanModel,
    MadgwickFilter, MahonyFilter, NominalState, NominalStateFn, UnscentedKalmanFilter,
};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Numeric, VectorFn};
use multicalc::spatial::{Quaternion, SO3};
use multicalc_qa::load::*;
use multicalc_qa::problems::{ConstantTurnAndSpeed, GlobalPosition, StationaryPose};
use multicalc_qa::schema::*;

#[must_use]
fn build_filter<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize>(
    fx: &Fixture,
) -> KalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION> {
    KalmanFilter::new(
        to_vector::<STATE_DIMENSION>(&fx.inputs["initial_state"]),
        to_matrix::<STATE_DIMENSION, STATE_DIMENSION>(&fx.inputs["initial_covariance"]),
        KalmanModel {
            state_transition: to_matrix::<STATE_DIMENSION, STATE_DIMENSION>(
                &fx.inputs["state_transition"],
            ),
            measurement_model: to_matrix::<MEASUREMENT_DIMENSION, STATE_DIMENSION>(
                &fx.inputs["measurement_model"],
            ),
            process_noise: to_matrix::<STATE_DIMENSION, STATE_DIMENSION>(
                &fx.inputs["process_noise"],
            ),
            measurement_noise: to_matrix::<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION>(
                &fx.inputs["measurement_noise"],
            ),
        },
    )
}

/// Compares a filter's final estimate against the golden. Takes the four
/// quantities rather than the filter so the linear and extended filters, which
/// are separate types, can share it.
fn assert_final_estimate<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize>(
    state: &Vector<STATE_DIMENSION>,
    covariance: &Matrix<STATE_DIMENSION, STATE_DIMENSION>,
    innovation: &Vector<MEASUREMENT_DIMENSION>,
    innovation_covariance: &Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION>,
    fx: &Fixture,
) {
    let t = fx.tolerances.f64;
    assert_vector(state, &fx.expected["state"], t, "state");
    assert_matrix(covariance, &fx.expected["covariance"], t, "covariance");
    assert_vector(innovation, &fx.expected["innovation"], t, "innovation");
    assert_matrix(
        innovation_covariance,
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
    assert_final_estimate(
        &filter.state(),
        &filter.covariance(),
        &filter.innovation(),
        &filter.innovation_covariance(),
        fx,
    );
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
    assert_final_estimate(
        &filter.state(),
        &filter.covariance(),
        &filter.innovation(),
        &filter.innovation_covariance(),
        fx,
    );
}

#[test]
fn kalman_filter_cases() {
    for fx in load_dir("estimation") {
        if fx.inputs["kind"].as_str() != "kalman_filter" {
            continue;
        }
        let (state_dimension, _) = fx.inputs["state_transition"].shape();
        let (measurement_dimension, _) = fx.inputs["measurement_model"].shape();
        match (state_dimension, measurement_dimension) {
            (2, 1) => run_kalman_filter::<2, 1>(&fx),
            (4, 2) => run_kalman_filter::<4, 2>(&fx),
            shape => panic!("unregistered kalman filter shape {shape:?}"),
        }
    }
}

#[test]
fn kalman_filter_with_control_cases() {
    for fx in load_dir("estimation") {
        if fx.inputs["kind"].as_str() != "kalman_filter_with_control" {
            continue;
        }
        let (state_dimension, _) = fx.inputs["state_transition"].shape();
        let (measurement_dimension, _) = fx.inputs["measurement_model"].shape();
        let (_, control_dimension) = fx.inputs["control_model"].shape();
        match (state_dimension, measurement_dimension, control_dimension) {
            (2, 1, 1) => run_kalman_filter_with_control::<2, 1, 1>(&fx),
            shape => panic!("unregistered kalman filter control shape {shape:?}"),
        }
    }
}

/// Range and bearing to a known landmark. Mirrors the model in
/// `tools/qa/gen/generators/estimation.py`; the two must stay in step.
struct LandmarkRangeAndBearing {
    landmark_x: f64,
    landmark_y: f64,
}

impl VectorFn<3, 2> for LandmarkRangeAndBearing {
    fn eval<S: Numeric>(&self, state: &[S; 3]) -> [S; 2] {
        let to_landmark_x = S::from_f64(self.landmark_x) - state[0];
        let to_landmark_y = S::from_f64(self.landmark_y) - state[1];
        [
            (to_landmark_x * to_landmark_x + to_landmark_y * to_landmark_y).sqrt(),
            to_landmark_y.atan2(to_landmark_x) - state[2],
        ]
    }
}

fn run_landmark_range_and_bearing(fx: &Fixture) {
    let landmark = fx.inputs["landmark"].as_vector();
    let model = LandmarkRangeAndBearing {
        landmark_x: landmark[0],
        landmark_y: landmark[1],
    };
    let mut filter = ExtendedKalmanFilter::<3, 2>::new(
        to_vector::<3>(&fx.inputs["initial_state"]),
        to_matrix::<3, 3>(&fx.inputs["initial_covariance"]),
        to_matrix::<3, 3>(&fx.inputs["process_noise"]),
        to_matrix::<2, 2>(&fx.inputs["measurement_noise"]),
    );

    let (steps, _, measurements) = fx.inputs["measurements"].as_matrix();
    for step in 0..steps {
        filter.predict(&StationaryPose).unwrap();
        let measurement = Vector::from_fn(|i| measurements[step * 2 + i]);
        filter.update(&model, measurement).unwrap();
    }

    assert_final_estimate(
        &filter.state(),
        &filter.covariance(),
        &filter.innovation(),
        &filter.innovation_covariance(),
        fx,
    );
}

fn run_coordinated_turn_fusion(fx: &Fixture) {
    let motion = ConstantTurnAndSpeed {
        timestep: fx.inputs["timestep"].as_scalar(),
    };
    let mut filter = ExtendedKalmanFilter::<5, 2>::new(
        to_vector::<5>(&fx.inputs["initial_state"]),
        to_matrix::<5, 5>(&fx.inputs["initial_covariance"]),
        to_matrix::<5, 5>(&fx.inputs["process_noise"]),
        to_matrix::<2, 2>(&fx.inputs["measurement_noise"]),
    );

    let (steps, _, measurements) = fx.inputs["measurements"].as_matrix();
    for step in 0..steps {
        filter.predict(&motion).unwrap();
        let measurement = Vector::from_fn(|i| measurements[step * 2 + i]);
        filter.update(&GlobalPosition, measurement).unwrap();
    }

    assert_final_estimate(
        &filter.state(),
        &filter.covariance(),
        &filter.innovation(),
        &filter.innovation_covariance(),
        fx,
    );
}

#[test]
fn extended_kalman_filter_cases() {
    for fx in load_dir("estimation") {
        if fx.inputs["kind"].as_str() != "extended_kalman_filter" {
            continue;
        }
        match fx.inputs["case"].as_str() {
            "landmark_range_and_bearing" => run_landmark_range_and_bearing(&fx),
            "coordinated_turn_fusion" => run_coordinated_turn_fusion(&fx),
            case => panic!("unregistered extended kalman filter case {case:?}"),
        }
    }
}

#[must_use]
fn build_unscented<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize>(
    fx: &Fixture,
) -> UnscentedKalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION> {
    UnscentedKalmanFilter::new(
        to_vector::<STATE_DIMENSION>(&fx.inputs["initial_state"]),
        to_matrix::<STATE_DIMENSION, STATE_DIMENSION>(&fx.inputs["initial_covariance"]),
        to_matrix::<STATE_DIMENSION, STATE_DIMENSION>(&fx.inputs["process_noise"]),
        to_matrix::<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION>(&fx.inputs["measurement_noise"]),
    )
    .with_scaling(
        fx.inputs["alpha"].as_scalar(),
        fx.inputs["beta"].as_scalar(),
        fx.inputs["kappa"].as_scalar(),
    )
    .unwrap()
}

fn run_unscented_coordinated_turn_fusion(fx: &Fixture) {
    let motion = ConstantTurnAndSpeed {
        timestep: fx.inputs["timestep"].as_scalar(),
    };
    let mut filter = build_unscented::<5, 2>(fx);

    let (steps, _, measurements) = fx.inputs["measurements"].as_matrix();
    for step in 0..steps {
        filter.predict(&motion).unwrap();
        let measurement = Vector::from_fn(|i| measurements[step * 2 + i]);
        filter.update(&GlobalPosition, measurement).unwrap();
    }

    assert_final_estimate(
        &filter.state(),
        &filter.covariance(),
        &filter.innovation(),
        &filter.innovation_covariance(),
        fx,
    );
}

fn run_unscented_landmark_range_and_bearing(fx: &Fixture) {
    let landmark = fx.inputs["landmark"].as_vector();
    let model = LandmarkRangeAndBearing {
        landmark_x: landmark[0],
        landmark_y: landmark[1],
    };
    let mut filter = build_unscented::<3, 2>(fx);

    let (steps, _, measurements) = fx.inputs["measurements"].as_matrix();
    for step in 0..steps {
        filter.predict(&StationaryPose).unwrap();
        let measurement = Vector::from_fn(|i| measurements[step * 2 + i]);
        filter.update(&model, measurement).unwrap();
    }

    assert_final_estimate(
        &filter.state(),
        &filter.covariance(),
        &filter.innovation(),
        &filter.innovation_covariance(),
        fx,
    );
}

#[test]
fn unscented_kalman_filter_cases() {
    for fx in load_dir("estimation") {
        if fx.inputs["kind"].as_str() != "unscented_kalman_filter" {
            continue;
        }
        match fx.inputs["case"].as_str() {
            "coordinated_turn_fusion" => run_unscented_coordinated_turn_fusion(&fx),
            "landmark_range_and_bearing" => run_unscented_landmark_range_and_bearing(&fx),
            case => panic!("unregistered unscented kalman filter case {case:?}"),
        }
    }
}

// ----- Error-state filter and two-direction attitude -----

/// A tracker in the room that reports where the body is.
struct RoomTracker;

impl NominalStateFn<3> for RoomTracker {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 3] {
        *state.position().as_array()
    }
}

/// A heading aid that reports which way the body is pointing about the vertical.
struct HeadingAid;

impl NominalStateFn<1> for HeadingAid {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 1] {
        let (_, _, heading) = state.orientation().quaternion().to_euler_zyx();
        [heading]
    }
}

/// A quaternion as `[w, x, y, z]` turned so the scalar part is not negative, which is how the
/// fixture stores it — the two opposite quaternions are the same rotation.
fn canonical_quaternion(orientation: SO3<f64>) -> Vector<4> {
    let quaternion = Vector::new(orientation.quaternion().as_array());
    if quaternion[0] < 0.0 {
        -quaternion
    } else {
        quaternion
    }
}

fn run_error_state_kalman_filter_imu_trajectory(fx: &Fixture) {
    let timestep = fx.inputs["timestep"].as_scalar();
    let gravity = to_vector::<3>(&fx.inputs["gravity"]);
    let initial_state = NominalState::new(
        to_vector::<3>(&fx.inputs["initial_position"]),
        to_vector::<3>(&fx.inputs["initial_velocity"]),
        SO3::from_quaternion(Quaternion::from_array(
            *to_vector::<4>(&fx.inputs["initial_orientation"]).as_array(),
        )),
        to_vector::<3>(&fx.inputs["initial_gyroscope_bias"]),
        to_vector::<3>(&fx.inputs["initial_accelerometer_bias"]),
    );
    let imu_noise = ImuNoise {
        gyroscope_noise_density: fx.inputs["gyroscope_noise_density"].as_scalar(),
        accelerometer_noise_density: fx.inputs["accelerometer_noise_density"].as_scalar(),
        gyroscope_bias_random_walk: fx.inputs["gyroscope_bias_random_walk"].as_scalar(),
        accelerometer_bias_random_walk: fx.inputs["accelerometer_bias_random_walk"].as_scalar(),
    };
    let mut filter = ErrorStateKalmanFilter::<3>::new(
        initial_state,
        to_matrix::<15, 15>(&fx.inputs["initial_covariance"]),
        imu_noise,
        to_matrix::<3, 3>(&fx.inputs["position_fix_noise"]),
    )
    .with_gravity(gravity);

    let (steps, _, gyroscope_readings) = fx.inputs["gyroscope_readings"].as_matrix();
    let (_, _, accelerometer_readings) = fx.inputs["accelerometer_readings"].as_matrix();
    let (_, _, position_fixes) = fx.inputs["position_fixes"].as_matrix();
    let (_, _, heading_aids) = fx.inputs["heading_aids"].as_matrix();
    let position_fix_period = fx.inputs["position_fix_period"].as_int() as usize;
    let heading_aid_period = fx.inputs["heading_aid_period"].as_int() as usize;
    let heading_aid_noise = to_matrix::<1, 1>(&fx.inputs["heading_aid_noise"]);

    let mut position_fix_index = 0;
    let mut heading_aid_index = 0;
    for step in 0..steps {
        let gyroscope_reading = Vector::from_fn(|axis| gyroscope_readings[step * 3 + axis]);
        let accelerometer_reading = Vector::from_fn(|axis| accelerometer_readings[step * 3 + axis]);
        filter
            .predict(gyroscope_reading, accelerometer_reading, timestep)
            .unwrap();

        // The position fix is three numbers wide, matching the width the filter is declared with,
        // so it goes in whole.
        if (step + 1) % position_fix_period == 0 {
            let fix = Vector::from_fn(|axis| position_fixes[position_fix_index * 3 + axis]);
            position_fix_index += 1;
            filter.update(&RoomTracker, fix).unwrap();
        }

        // The heading aid is one number wide and it is an angle, so the residual is formed and
        // wrapped here rather than left to plain subtraction.
        if (step + 1) % heading_aid_period == 0 {
            let reading = heading_aids[heading_aid_index];
            heading_aid_index += 1;
            let predicted = HeadingAid.eval(&filter.nominal_state())[0];
            let residual = Vector::new([(reading - predicted).wrap_to_pi()]);
            filter
                .update_other(&HeadingAid, residual, heading_aid_noise)
                .unwrap();
        }
    }

    let t = fx.tolerances.f64;
    let state = filter.nominal_state();
    assert_vector(&state.position(), &fx.expected["position"], t, "position");
    assert_vector(&state.velocity(), &fx.expected["velocity"], t, "velocity");
    assert_vector(
        &canonical_quaternion(state.orientation()),
        &fx.expected["orientation"],
        t,
        "orientation",
    );
    assert_vector(
        &state.gyroscope_bias(),
        &fx.expected["gyroscope_bias"],
        t,
        "gyroscope_bias",
    );
    assert_vector(
        &state.accelerometer_bias(),
        &fx.expected["accelerometer_bias"],
        t,
        "accelerometer_bias",
    );
    assert_matrix(
        &filter.covariance(),
        &fx.expected["covariance"],
        t,
        "covariance",
    );
}

fn run_triad_attitude_from_two_directions(fx: &Fixture) {
    let orientation = SO3::from_two_direction_pairs(
        to_vector::<3>(&fx.inputs["primary_observed"]),
        to_vector::<3>(&fx.inputs["secondary_observed"]),
        to_vector::<3>(&fx.inputs["primary_reference"]),
        to_vector::<3>(&fx.inputs["secondary_reference"]),
    )
    .unwrap();

    let t = fx.tolerances.f64;
    assert_vector(
        &canonical_quaternion(orientation),
        &fx.expected["orientation"],
        t,
        "orientation",
    );
    assert_matrix(
        &orientation.to_matrix(),
        &fx.expected["rotation_matrix"],
        t,
        "rotation_matrix",
    );
}

fn run_mahony_attitude_filter(fx: &Fixture) {
    let timestep = fx.inputs["timestep"].as_scalar();
    let initial_orientation = SO3::from_quaternion(Quaternion::from_array(
        *to_vector::<4>(&fx.inputs["initial_orientation"]).as_array(),
    ));
    let mut filter = MahonyFilter::new(initial_orientation)
        .with_proportional_gain(fx.inputs["proportional_gain"].as_scalar())
        .with_integral_gain(fx.inputs["integral_gain"].as_scalar())
        .with_reference_directions(
            to_vector::<3>(&fx.inputs["upward_reference"]),
            to_vector::<3>(&fx.inputs["north_reference"]),
        );

    let (steps, _, gyroscope_readings) = fx.inputs["gyroscope_readings"].as_matrix();
    let (_, _, accelerometer_readings) = fx.inputs["accelerometer_readings"].as_matrix();
    let (_, _, magnetometer_readings) = fx.inputs["magnetometer_readings"].as_matrix();

    for step in 0..steps {
        let gyroscope_reading = Vector::from_fn(|axis| gyroscope_readings[step * 3 + axis]);
        let accelerometer_reading = Vector::from_fn(|axis| accelerometer_readings[step * 3 + axis]);
        let magnetometer_reading = Vector::from_fn(|axis| magnetometer_readings[step * 3 + axis]);
        filter
            .step(
                gyroscope_reading,
                accelerometer_reading,
                Some(magnetometer_reading),
                timestep,
            )
            .unwrap();
    }

    let t = fx.tolerances.f64;
    assert_vector(
        &canonical_quaternion(filter.orientation()),
        &fx.expected["orientation"],
        t,
        "orientation",
    );
    assert_vector(
        &filter.gyroscope_bias(),
        &fx.expected["gyroscope_bias"],
        t,
        "gyroscope_bias",
    );
}

fn run_madgwick_attitude_filter(fx: &Fixture) {
    let timestep = fx.inputs["timestep"].as_scalar();
    let initial_orientation = SO3::from_quaternion(Quaternion::from_array(
        *to_vector::<4>(&fx.inputs["initial_orientation"]).as_array(),
    ));
    let mut filter = MadgwickFilter::new(initial_orientation)
        .with_correction_gain(fx.inputs["correction_gain"].as_scalar())
        .with_bias_gain(fx.inputs["bias_gain"].as_scalar())
        .with_reference_directions(
            to_vector::<3>(&fx.inputs["upward_reference"]),
            to_vector::<3>(&fx.inputs["north_reference"]),
        );

    let (steps, _, gyroscope_readings) = fx.inputs["gyroscope_readings"].as_matrix();
    let (_, _, accelerometer_readings) = fx.inputs["accelerometer_readings"].as_matrix();
    let (_, _, magnetometer_readings) = fx.inputs["magnetometer_readings"].as_matrix();

    for step in 0..steps {
        let gyroscope_reading = Vector::from_fn(|axis| gyroscope_readings[step * 3 + axis]);
        let accelerometer_reading = Vector::from_fn(|axis| accelerometer_readings[step * 3 + axis]);
        let magnetometer_reading = Vector::from_fn(|axis| magnetometer_readings[step * 3 + axis]);
        filter
            .step(
                gyroscope_reading,
                accelerometer_reading,
                Some(magnetometer_reading),
                timestep,
            )
            .unwrap();
    }

    let t = fx.tolerances.f64;
    assert_vector(
        &canonical_quaternion(filter.orientation()),
        &fx.expected["orientation"],
        t,
        "orientation",
    );
    assert_vector(
        &filter.gyroscope_bias(),
        &fx.expected["gyroscope_bias"],
        t,
        "gyroscope_bias",
    );
}

#[test]
fn error_state_kalman_filter_cases() {
    for fx in load_dir("estimation") {
        if fx.inputs["kind"].as_str() != "error_state_kalman_filter" {
            continue;
        }
        match fx.inputs["case"].as_str() {
            "imu_trajectory" => run_error_state_kalman_filter_imu_trajectory(&fx),
            case => panic!("unregistered error-state kalman filter case {case:?}"),
        }
    }
}

#[test]
fn triad_cases() {
    for fx in load_dir("estimation") {
        if fx.inputs["kind"].as_str() != "triad" {
            continue;
        }
        match fx.inputs["case"].as_str() {
            "attitude_from_two_directions" => run_triad_attitude_from_two_directions(&fx),
            case => panic!("unregistered triad case {case:?}"),
        }
    }
}

#[test]
fn attitude_filter_cases() {
    for fx in load_dir("estimation") {
        if fx.inputs["kind"].as_str() != "attitude_filter" {
            continue;
        }
        match fx.inputs["case"].as_str() {
            "mahony_gyroscope_accelerometer_magnetometer" => run_mahony_attitude_filter(&fx),
            "madgwick_gyroscope_accelerometer_magnetometer" => run_madgwick_attitude_filter(&fx),
            case => panic!("unregistered attitude filter case {case:?}"),
        }
    }
}
