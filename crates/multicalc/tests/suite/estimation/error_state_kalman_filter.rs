//! Error-state filter tests: nominal propagation, the error-transition matrix against an
//! autodiff Jacobian, injection and reset round trips, the non-finite policy, degeneration to a
//! plain linear filter, statistical consistency, and bias convergence.

use multicalc::error::EstimationError;
use multicalc::estimation::{
    ErrorStateKalmanFilter, ImuNoise, KalmanFilter, KalmanModel, NominalState, NominalStateFn,
};
use multicalc::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use multicalc::numerical_derivative::Jacobian;
use multicalc::random::{Pcg32, RandomSource};
use multicalc::scalar::{Numeric, VectorFn};
use multicalc::spatial::SO3;

const GRAVITY_STRENGTH: f64 = 9.81;

// ---- helpers ----------------------------------------------------------------

/// A tracker in the room reports where the body is, and nothing else.
struct RoomTracker;

impl NominalStateFn<3> for RoomTracker {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 3] {
        *state.position().as_array()
    }
}

/// A compass reads which way the body faces about the vertical.
struct Compass;

impl NominalStateFn<1> for Compass {
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 1] {
        let forward = state
            .orientation()
            .act(Vector::new([S::ONE, S::ZERO, S::ZERO]));
        [forward[1].atan2(forward[0])]
    }
}

fn quiet_imu_noise() -> ImuNoise<f64> {
    ImuNoise {
        gyroscope_noise_density: 0.001,
        accelerometer_noise_density: 0.001,
        gyroscope_bias_random_walk: 1e-6,
        accelerometer_bias_random_walk: 1e-6,
    }
}

fn realistic_imu_noise() -> ImuNoise<f64> {
    ImuNoise {
        gyroscope_noise_density: 0.02,
        accelerometer_noise_density: 0.05,
        gyroscope_bias_random_walk: 1e-4,
        accelerometer_bias_random_walk: 1e-3,
    }
}

/// A filter that believes its starting guess and expects a tight position fix.
fn tracker_filter(initial_state: NominalState<f64>) -> ErrorStateKalmanFilter<3, f64> {
    let tracker_spread = 0.03;
    ErrorStateKalmanFilter::new(
        initial_state,
        Matrix::from_diagonal([0.01; 15]),
        quiet_imu_noise(),
        Matrix::from_diagonal([tracker_spread * tracker_spread; 3]),
    )
}

// ---- nominal propagation ----------------------------------------------------

#[test]
fn still_body_does_not_move() {
    let level = SO3::<f64>::identity();
    let mut filter = tracker_filter(NominalState::at_rest(level));

    // Held against gravity, the push sensor reads a full gravity upward.
    let gyroscope_reading = Vector::new([0.0, 0.0, 0.0]);
    let accelerometer_reading = Vector::new([0.0, 0.0, GRAVITY_STRENGTH]);
    let timestep = 0.001;
    let step_count = 1000;
    for _ in 0..step_count {
        filter
            .predict(gyroscope_reading, accelerometer_reading, timestep)
            .unwrap();
    }

    let state = filter.nominal_state();
    assert!(state.position().norm() < 1e-9);
    assert!(state.velocity().norm() < 1e-9);
    assert!(state.orientation().log().norm() < 1e-12);
}

#[test]
fn free_fall_accelerates_downward() {
    let level = SO3::<f64>::identity();
    let mut filter = tracker_filter(NominalState::at_rest(level));

    // In free fall the push sensor reads nothing at all.
    let gyroscope_reading = Vector::new([0.0, 0.0, 0.0]);
    let accelerometer_reading = Vector::new([0.0, 0.0, 0.0]);
    let timestep = 0.001;
    let fall_time = 1.0;
    let step_count = (fall_time / timestep) as usize;
    for _ in 0..step_count {
        filter
            .predict(gyroscope_reading, accelerometer_reading, timestep)
            .unwrap();
    }

    let state = filter.nominal_state();
    let expected_speed = -GRAVITY_STRENGTH * fall_time;
    let expected_drop = -0.5 * GRAVITY_STRENGTH * fall_time * fall_time;
    assert!((state.velocity()[2] - expected_speed).abs() < 1e-6);
    // Looser than the speed: each step's half-timestep-squared term is exact, but a thousand of
    // them summed in floating point is not.
    assert!((state.position()[2] - expected_drop).abs() < 1e-4);
}

#[test]
fn constant_turn_rate_integrates_to_the_right_angle() {
    let level = SO3::<f64>::identity();
    let mut filter = tracker_filter(NominalState::at_rest(level));

    let turn_rate = 1.5;
    let turn_time = 2.0;
    let timestep = 0.001;
    let gyroscope_reading = Vector::new([0.0, 0.0, turn_rate]);
    let accelerometer_reading = Vector::new([0.0, 0.0, GRAVITY_STRENGTH]);
    let step_count = (turn_time / timestep) as usize;
    for _ in 0..step_count {
        filter
            .predict(gyroscope_reading, accelerometer_reading, timestep)
            .unwrap();
    }

    let turned = filter.nominal_state().orientation().log()[2];
    assert!((turned - (turn_rate * turn_time).wrap_to_pi()).abs() < 1e-9);
}

#[test]
fn bias_is_subtracted_before_use() {
    let level = SO3::<f64>::identity();
    let gyroscope_bias = Vector::new([0.02, -0.015, 0.01]);
    let accelerometer_bias = Vector::new([0.0, 0.0, 0.0]);
    let start = NominalState::new(
        Vector::zeros(),
        Vector::zeros(),
        level,
        gyroscope_bias,
        accelerometer_bias,
    );
    let mut filter = tracker_filter(start);

    // The turn-rate sensor reads exactly its own offset, so the body is not really turning.
    let accelerometer_reading = Vector::new([0.0, 0.0, GRAVITY_STRENGTH]);
    let timestep = 0.001;
    let run_time = 1.0;
    let step_count = (run_time / timestep) as usize;
    for _ in 0..step_count {
        filter
            .predict(gyroscope_bias, accelerometer_reading, timestep)
            .unwrap();
    }

    assert!(filter.nominal_state().orientation().log().norm() < 1e-9);
}

#[test]
fn error_round_trips_through_plus_and_minus() {
    let mut generator = Pcg32::new(20260802);
    for _ in 0..200 {
        let state = random_state(&mut generator);
        let error = random_error(&mut generator);

        let round_tripped = state.plus_error(error).error_from(state);
        for index in 0..15 {
            assert!(
                (round_tripped[index] - error[index]).abs() < 1e-12,
                "[{index}]: {} vs {}",
                round_tripped[index],
                error[index]
            );
        }
    }
}

fn random_state(generator: &mut Pcg32) -> NominalState<f64> {
    let mut draw = || generator.standard_normal();
    let position = Vector::new([draw(), draw(), draw()]);
    let velocity = Vector::new([draw(), draw(), draw()]);
    let rotation_vector = Vector::new([draw(), draw(), draw()]);
    let gyroscope_bias = Vector::new([draw() * 0.01, draw() * 0.01, draw() * 0.01]);
    let accelerometer_bias = Vector::new([draw() * 0.1, draw() * 0.1, draw() * 0.1]);
    NominalState::new(
        position,
        velocity,
        SO3::exp(rotation_vector),
        gyroscope_bias,
        accelerometer_bias,
    )
}

/// A random error whose rotation part stays under a radian, so the round trip is inside the half
/// turn where the two directions agree.
fn random_error(generator: &mut Pcg32) -> Vector<15, f64> {
    let mut error = Vector::zeros();
    for index in 0..15 {
        error[index] = generator.standard_normal() * 0.3;
    }
    let rotation_error = Vector::new([error[6], error[7], error[8]]);
    let rotation_size = rotation_error.norm();
    if rotation_size > 1.0 {
        for index in 6..9 {
            error[index] /= rotation_size;
        }
    }
    error
}

// ---- the error transition matrix --------------------------------------------

/// The exact error transition, written as a function of the error so it can be differentiated.
///
/// It perturbs the state by the error, propagates, and reports the error of the result against the
/// unperturbed propagation. Its derivative at zero error is by definition the matrix the filter
/// builds in closed form.
struct PropagationError {
    position: [f64; 3],
    velocity: [f64; 3],
    orientation: [f64; 4],
    gyroscope_bias: [f64; 3],
    accelerometer_bias: [f64; 3],
    gyroscope_reading: [f64; 3],
    accelerometer_reading: [f64; 3],
    timestep: f64,
    gravity: [f64; 3],
}

impl PropagationError {
    fn nominal<S: Numeric>(&self) -> NominalState<S> {
        let [w, x, y, z] = self.orientation;
        NominalState::new(
            Vector::new(self.position.map(S::from_f64)),
            Vector::new(self.velocity.map(S::from_f64)),
            SO3::from_quaternion(multicalc::Quaternion::new(
                S::from_f64(w),
                S::from_f64(x),
                S::from_f64(y),
                S::from_f64(z),
            )),
            Vector::new(self.gyroscope_bias.map(S::from_f64)),
            Vector::new(self.accelerometer_bias.map(S::from_f64)),
        )
    }
}

impl VectorFn<15, 15> for PropagationError {
    fn eval<S: Numeric>(&self, error: &[S; 15]) -> [S; 15] {
        let nominal = self.nominal::<S>();
        let gyroscope_reading = Vector::new(self.gyroscope_reading.map(S::from_f64));
        let accelerometer_reading = Vector::new(self.accelerometer_reading.map(S::from_f64));
        let timestep = S::from_f64(self.timestep);
        let gravity = Vector::new(self.gravity.map(S::from_f64));

        let perturbed = nominal.plus_error(Vector::new(*error)).propagated(
            gyroscope_reading,
            accelerometer_reading,
            timestep,
            gravity,
        );
        let plain = nominal.propagated(gyroscope_reading, accelerometer_reading, timestep, gravity);
        *perturbed.error_from(plain).as_array()
    }
}

#[test]
fn transition_matrix_matches_an_autodiff_jacobian_to_second_order() {
    let orientation = SO3::exp(Vector::new([0.3, -0.2, 0.5]));
    let gyroscope_reading = [0.4, -0.3, 0.9];
    let accelerometer_reading = [0.5, -1.2, 9.4];
    let gyroscope_bias = [0.02, -0.015, 0.01];
    let accelerometer_bias = [0.15, -0.10, 0.05];
    let gravity = [0.0, 0.0, -GRAVITY_STRENGTH];

    let largest_difference = |timestep: f64| {
        let start = NominalState::new(
            Vector::new([1.0, -2.0, 0.5]),
            Vector::new([0.3, 0.1, -0.2]),
            orientation,
            Vector::new(gyroscope_bias),
            Vector::new(accelerometer_bias),
        );
        let filter = tracker_filter(start);
        let closed_form = filter.error_state_transition(
            Vector::new(gyroscope_reading),
            Vector::new(accelerometer_reading),
            timestep,
        );

        let model = PropagationError {
            position: *start.position().as_array(),
            velocity: *start.velocity().as_array(),
            orientation: start.orientation().quaternion().as_array(),
            gyroscope_bias,
            accelerometer_bias,
            gyroscope_reading,
            accelerometer_reading,
            timestep,
            gravity,
        };
        let jacobian: Jacobian = Jacobian::default();
        let exact = jacobian.evaluate(&model, &[0.0_f64; 15]).unwrap();

        let mut largest = 0.0_f64;
        for row in 0..15 {
            for column in 0..15 {
                largest = largest.max((closed_form[(row, column)] - exact[(row, column)]).abs());
            }
        }
        largest
    };

    // The largest term the closed form leaves out is the position row's half-timestep-squared
    // dependence on a tilt, which is the push itself times that half square. Allow twice it.
    let push_size = (Vector::new(accelerometer_reading) - Vector::new(accelerometer_bias)).norm();
    let coarse_timestep = 0.01;
    let coarse = largest_difference(coarse_timestep);
    assert!(
        coarse < push_size * coarse_timestep * coarse_timestep,
        "largest difference {coarse} at timestep {coarse_timestep}"
    );

    // The closed form drops terms that shrink with the timestep squared, so halving the step must
    // shrink the gap about fourfold. A missing or mistyped block would leave a gap that shrinks
    // only twofold, or not at all.
    let fine_timestep = 0.005;
    let fine = largest_difference(fine_timestep);
    let shrink = coarse / fine;
    assert!(
        (3.0..5.0).contains(&shrink),
        "gap shrank by {shrink} when the timestep halved: {coarse} then {fine}"
    );
}

// ---- injection and reset ----------------------------------------------------

#[test]
fn reset_round_trips_a_known_error() {
    let level = SO3::<f64>::identity();
    let rotation_spread = Matrix3D::from_diagonal([0.09, 0.04, 0.16]);
    let mut covariance = Matrix::<15, 15, f64>::from_diagonal([0.02; 15]);
    for row in 0..3 {
        for column in 0..3 {
            covariance[(6 + row, 6 + column)] = rotation_spread[(row, column)];
        }
    }
    let mut filter = ErrorStateKalmanFilter::<3, f64>::new(
        NominalState::at_rest(level),
        covariance,
        quiet_imu_noise(),
        Matrix::from_diagonal([0.001; 3]),
    );

    let turn = 0.2;
    let rotation_error = Vector::new([0.0, 0.0, turn]);
    let mut error = Vector::zeros();
    error[6] = rotation_error[0];
    error[7] = rotation_error[1];
    error[8] = rotation_error[2];
    filter.inject_error_and_reset(error);

    // The guess turned by exactly the error that was folded in.
    let turned = filter.nominal_state().orientation().log();
    assert!((turned - rotation_error).norm() < 1e-12);

    // The rotation block came across through `G = I − ½[δθ]×`, written out here by hand.
    let carry = Matrix3D::identity() - SO3::hat(rotation_error).scale(0.5);
    let expected = carry * rotation_spread * carry.transpose();
    for row in 0..3 {
        for column in 0..3 {
            let got = filter.covariance()[(6 + row, 6 + column)];
            assert!(
                (got - expected[(row, column)]).abs() < 1e-12,
                "({row},{column}): {got} vs {}",
                expected[(row, column)]
            );
        }
    }

    // Everywhere else `G` is the identity, so nothing moved at all.
    for row in 0..15 {
        for column in 0..15 {
            let inside_rotation = (6..9).contains(&row) && (6..9).contains(&column);
            if !inside_rotation {
                assert_eq!(
                    filter.covariance()[(row, column)],
                    covariance[(row, column)]
                );
            }
        }
    }
}

#[test]
fn reset_jacobian_is_not_the_identity() {
    let level = SO3::<f64>::identity();
    let rotation_spread = Matrix3D::from_diagonal([0.09, 0.04, 0.16]);
    let mut covariance = Matrix::<15, 15, f64>::zeros();
    for row in 0..3 {
        for column in 0..3 {
            covariance[(6 + row, 6 + column)] = rotation_spread[(row, column)];
        }
    }
    let mut filter = ErrorStateKalmanFilter::<3, f64>::new(
        NominalState::at_rest(level),
        covariance,
        quiet_imu_noise(),
        Matrix::from_diagonal([0.001; 3]),
    );

    let mut error = Vector::zeros();
    error[8] = 0.4;
    filter.inject_error_and_reset(error);

    // Dropping the carry-across would leave the rotation block untouched, which the round trip on
    // the guess alone cannot see.
    let mut moved = 0.0_f64;
    for row in 0..3 {
        for column in 0..3 {
            let difference =
                filter.covariance()[(6 + row, 6 + column)] - rotation_spread[(row, column)];
            moved += difference * difference;
        }
    }
    assert!(
        moved.sqrt() > 1e-3,
        "the rotation block barely moved: {moved}"
    );
}

// ---- the non-finite policy --------------------------------------------------

#[test]
fn non_finite_inputs_are_rejected() {
    let level = SO3::<f64>::identity();
    let good_gyroscope = Vector::new([0.0, 0.0, 0.0]);
    let good_accelerometer = Vector::new([0.0, 0.0, GRAVITY_STRENGTH]);
    let timestep = 0.001;

    let mut filter = tracker_filter(NominalState::at_rest(level));
    let nan_gyroscope = Vector::new([f64::NAN, 0.0, 0.0]);
    assert_eq!(
        filter.predict(nan_gyroscope, good_accelerometer, timestep),
        Err(EstimationError::NonFinite)
    );

    let infinite_accelerometer = Vector::new([0.0, 0.0, f64::INFINITY]);
    assert_eq!(
        filter.predict(good_gyroscope, infinite_accelerometer, timestep),
        Err(EstimationError::NonFinite)
    );

    assert_eq!(
        filter.predict(good_gyroscope, good_accelerometer, f64::NAN),
        Err(EstimationError::NonFinite)
    );

    let nan_measurement = Vector::new([f64::NAN, 0.0, 0.0]);
    assert_eq!(
        filter.update(&RoomTracker, nan_measurement),
        Err(EstimationError::NonFinite)
    );

    let infinite_residual = Vector::new([f64::INFINITY, 0.0, 0.0]);
    assert_eq!(
        filter.update_with_residual(&RoomTracker, infinite_residual),
        Err(EstimationError::NonFinite)
    );
}

#[test]
fn singular_innovation_covariance_is_rejected() {
    let level = SO3::<f64>::identity();
    let mut filter = ErrorStateKalmanFilter::<3, f64>::new(
        NominalState::at_rest(level),
        Matrix::zeros(),
        quiet_imu_noise(),
        Matrix::zeros(),
    );

    let fix = Vector::new([0.1, 0.0, 0.0]);
    assert_eq!(
        filter.update(&RoomTracker, fix),
        Err(EstimationError::NotPositiveDefinite)
    );
}

// ---- degeneration to a linear filter ----------------------------------------

#[test]
fn position_only_filter_matches_a_linear_filter() {
    // Held level, with the push sensor reading exactly a full gravity upward and no bias wander,
    // the position and velocity rows are a plain constant-acceleration model with no acceleration.
    let level = SO3::<f64>::identity();
    let timestep = 0.01;
    let accelerometer_noise_density = 0.05;
    let imu_noise = ImuNoise {
        gyroscope_noise_density: 0.0,
        accelerometer_noise_density,
        gyroscope_bias_random_walk: 0.0,
        accelerometer_bias_random_walk: 0.0,
    };
    let tracker_spread = 0.03;
    let measurement_noise = Matrix::from_diagonal([tracker_spread * tracker_spread; 3]);
    let initial_spread = 0.2;

    // The facing and the two offsets are held certain as well as still. Any spread left on them
    // would leak into the velocity through the tilt and offset couplings, which is exactly the part
    // a six-number linear filter has no room for.
    let mut error_state_covariance = Matrix::<15, 15, f64>::zeros();
    for index in 0..6 {
        error_state_covariance[(index, index)] = initial_spread;
    }
    let mut error_state = ErrorStateKalmanFilter::<3, f64>::new(
        NominalState::at_rest(level),
        error_state_covariance,
        imu_noise,
        measurement_noise,
    );

    // The same model as six numbers: position then velocity, each of three axes.
    let mut state_transition = Matrix::<6, 6, f64>::identity();
    for axis in 0..3 {
        state_transition[(axis, 3 + axis)] = timestep;
    }
    let mut linear_measurement = Matrix::<3, 6, f64>::zeros();
    for axis in 0..3 {
        linear_measurement[(axis, axis)] = 1.0;
    }
    let velocity_noise = accelerometer_noise_density * timestep;
    let mut process_noise = Matrix::<6, 6, f64>::zeros();
    for axis in 0..3 {
        process_noise[(3 + axis, 3 + axis)] = velocity_noise * velocity_noise;
    }
    let mut linear = KalmanFilter::<6, 3, f64>::new(
        Vector::zeros(),
        Matrix::from_diagonal([initial_spread; 6]),
        KalmanModel {
            state_transition,
            measurement_model: linear_measurement,
            process_noise,
            measurement_noise,
        },
    );

    let gyroscope_reading = Vector::new([0.0, 0.0, 0.0]);
    let accelerometer_reading = Vector::new([0.0, 0.0, GRAVITY_STRENGTH]);
    for step in 0..20 {
        error_state
            .predict(gyroscope_reading, accelerometer_reading, timestep)
            .unwrap();
        linear.predict();

        let drift = step as f64 * 0.01;
        let fix = Vector::new([drift, -drift, 0.5 * drift]);
        error_state.update(&RoomTracker, fix).unwrap();
        linear.update(fix).unwrap();
    }

    let position = error_state.nominal_state().position();
    let velocity = error_state.nominal_state().velocity();
    for axis in 0..3 {
        assert!((position[axis] - linear.state()[axis]).abs() < 1e-8);
        assert!((velocity[axis] - linear.state()[3 + axis]).abs() < 1e-8);
    }
}

// ---- covariance conditioning ------------------------------------------------

#[test]
fn conditioning_repairs_a_drifted_covariance() {
    let level = SO3::<f64>::identity();
    let mut generator = Pcg32::new(20260804);
    let directions = random_orthonormal_matrix(&mut generator);

    // One direction has drifted just below zero, which is what makes the next gain meaningless.
    let mut spread = [0.05_f64; 15];
    spread[7] = -1e-9;
    let drifted = directions * Matrix::from_diagonal(spread) * directions.transpose();

    let mut filter = ErrorStateKalmanFilter::<3, f64>::new(
        NominalState::at_rest(level),
        drifted,
        quiet_imu_noise(),
        Matrix::from_diagonal([0.001; 3]),
    );

    let minimum_eigenvalue = 1e-12;
    filter.condition_covariance(minimum_eigenvalue).unwrap();

    let repaired = filter.covariance().symmetric_eigendecomposition().unwrap();
    assert!(repaired.is_positive_definite());

    let before = drifted.symmetric_eigendecomposition().unwrap();
    for index in 0..15 {
        let moved = (repaired.eigenvalues()[index] - before.eigenvalues()[index]).abs();
        assert!(moved < 1e-8, "[{index}] moved by {moved}");
    }
}

/// Fifteen perpendicular unit directions, built by orthogonalizing random ones against each other.
fn random_orthonormal_matrix(generator: &mut Pcg32) -> Matrix<15, 15, f64> {
    let mut columns = [[0.0_f64; 15]; 15];
    for index in 0..15 {
        let mut candidate = Vector::<15, f64>::zeros();
        for entry in 0..15 {
            candidate[entry] = generator.standard_normal();
        }
        for earlier in columns.iter().take(index) {
            let settled = Vector::new(*earlier);
            candidate -= settled * candidate.dot(settled);
        }
        columns[index] = candidate.try_normalized().unwrap().into_array();
    }
    Matrix::from_fn(|row, column| columns[column][row])
}

// ---- statistical consistency ------------------------------------------------

/// One synthetic flight: a body turning at a fixed rate with a mild sway, carrying an IMU whose
/// readings are offset and noisy.
struct Flight {
    truth: NominalState<f64>,
    gyroscope_bias: Vector3D<f64>,
    accelerometer_bias: Vector3D<f64>,
    turn_rate: Vector3D<f64>,
    gravity: Vector3D<f64>,
    step: usize,
}

impl Flight {
    fn new(gyroscope_bias: Vector3D<f64>, accelerometer_bias: Vector3D<f64>) -> Self {
        Flight {
            // The truth carries the sensors' real offsets, since those are among the fifteen
            // numbers the filter is trying to get right.
            truth: NominalState::new(
                Vector::zeros(),
                Vector::zeros(),
                SO3::identity(),
                gyroscope_bias,
                accelerometer_bias,
            ),
            gyroscope_bias,
            accelerometer_bias,
            turn_rate: Vector::new([0.3, -0.2, 0.5]),
            gravity: Vector::new([0.0, 0.0, -GRAVITY_STRENGTH]),
            step: 0,
        }
    }

    /// The offsets the bias-convergence tests inject, big enough to be worth learning and small
    /// enough to be realistic for a hobby-grade IMU.
    fn standard_biases() -> (Vector3D<f64>, Vector3D<f64>) {
        (
            Vector::new([0.02, -0.015, 0.01]),
            Vector::new([0.15, -0.10, 0.05]),
        )
    }

    /// The proper push the body really feels: gravity resisted, plus a slow sway along x.
    fn proper_push(&self, timestep: f64) -> Vector3D<f64> {
        let time = self.step as f64 * timestep;
        let sway = 0.5 * (2.0 * core::f64::consts::PI * 0.5 * time).sin();
        let world_push = Vector::new([sway, 0.0, 0.0]);
        self.truth
            .orientation()
            .inverse()
            .act(world_push - self.gravity)
    }

    /// Rolls the truth forward and hands back the readings a real IMU would have produced: the
    /// truth, plus each sensor's steady offset, plus jitter.
    fn step(
        &mut self,
        timestep: f64,
        imu_noise: &ImuNoise<f64>,
        generator: &mut Pcg32,
    ) -> (Vector3D<f64>, Vector3D<f64>) {
        let proper_push = self.proper_push(timestep);
        // The filter carries a reading's jitter into the error as `(density · Δt)²` per step, so
        // the jitter put in here has to be the per-sample spread the filter is told, straight.
        let turn_jitter = imu_noise.gyroscope_noise_density;
        let push_jitter = imu_noise.accelerometer_noise_density;

        // What a perfect copy of these sensors would read: the real motion seen through their
        // steady offsets. The truth is rolled forward with these, so it takes its own offsets back
        // off and moves by the real rates.
        let clean_gyroscope = self.turn_rate + self.gyroscope_bias;
        let clean_accelerometer = proper_push + self.accelerometer_bias;
        self.truth =
            self.truth
                .propagated(clean_gyroscope, clean_accelerometer, timestep, self.gravity);
        self.step += 1;

        (
            clean_gyroscope + random_vector(generator, turn_jitter),
            clean_accelerometer + random_vector(generator, push_jitter),
        )
    }
}

fn random_vector(generator: &mut Pcg32, spread: f64) -> Vector3D<f64> {
    Vector::new([
        generator.standard_normal() * spread,
        generator.standard_normal() * spread,
        generator.standard_normal() * spread,
    ])
}

/// How unsure the filter starts, one entry per error number.
fn starting_spread() -> [f64; 15] {
    let position_spread = 0.1;
    let velocity_spread = 0.1;
    let tilt_spread = 0.05;
    let gyroscope_bias_spread = 0.01;
    let accelerometer_bias_spread = 0.1;
    let mut spread = [0.0; 15];
    for index in 0..3 {
        spread[index] = position_spread * position_spread;
        spread[3 + index] = velocity_spread * velocity_spread;
        spread[6 + index] = tilt_spread * tilt_spread;
        spread[9 + index] = gyroscope_bias_spread * gyroscope_bias_spread;
        spread[12 + index] = accelerometer_bias_spread * accelerometer_bias_spread;
    }
    spread
}

fn flight_filter(initial_state: NominalState<f64>) -> ErrorStateKalmanFilter<3, f64> {
    let tracker_spread = 0.03;
    ErrorStateKalmanFilter::new(
        initial_state,
        Matrix::from_diagonal(starting_spread()),
        realistic_imu_noise(),
        Matrix::from_diagonal([tracker_spread * tracker_spread; 3]),
    )
}

/// Runs one flight, correcting from a position fix and a heading aid, and returns the filter beside
/// the truth it was chasing.
fn run_flight(
    mut filter: ErrorStateKalmanFilter<3, f64>,
    mut flight: Flight,
    step_count: usize,
    timestep: f64,
    generator: &mut Pcg32,
) -> (ErrorStateKalmanFilter<3, f64>, NominalState<f64>) {
    let imu_noise = realistic_imu_noise();
    let position_fix_period = 20;
    let position_fix_spread = 0.03;
    let heading_aid_period = 40;
    let heading_aid_spread = 2.0_f64.to_radians();
    let heading_aid_noise = Matrix::from_diagonal([heading_aid_spread * heading_aid_spread; 1]);

    for step in 0..step_count {
        let (gyroscope_reading, accelerometer_reading) =
            flight.step(timestep, &imu_noise, generator);
        filter
            .predict(gyroscope_reading, accelerometer_reading, timestep)
            .unwrap();

        if step % position_fix_period == position_fix_period - 1 {
            let fix = flight.truth.position() + random_vector(generator, position_fix_spread);
            filter.update(&RoomTracker, fix).unwrap();
        }

        if step % heading_aid_period == heading_aid_period - 1 {
            let reading =
                Compass.eval(&flight.truth)[0] + generator.standard_normal() * heading_aid_spread;
            let predicted = Compass.eval(&filter.nominal_state())[0];
            let residual = Vector::new([(reading - predicted).wrap_to_pi()]);
            filter
                .update_other(&Compass, residual, heading_aid_noise)
                .unwrap();
        }
    }

    (filter, flight.truth)
}

#[test]
fn post_reset_consistency_stays_in_bounds() {
    let timestep = 0.005;
    let step_count = 500;
    let run_count = 100;

    let mut total = 0.0_f64;
    for run in 0..run_count {
        let mut generator = Pcg32::new(20260803 + run as u64);

        // The filter has to start as wrong as it claims to be. Each run draws the offsets it is
        // facing, and its own starting guess, from the very spread it is seeded with — otherwise
        // the same fixed mistake in every run would show up as a filter that is worse than it
        // admits, when really it is the experiment that was rigged.
        let spread = starting_spread();
        let mut starting_error = Vector::<15, f64>::zeros();
        for index in 0..15 {
            starting_error[index] = generator.standard_normal() * spread[index].sqrt();
        }
        let gyroscope_bias = random_vector(&mut generator, spread[9].sqrt());
        let accelerometer_bias = random_vector(&mut generator, spread[12].sqrt());

        let flight = Flight::new(gyroscope_bias, accelerometer_bias);
        let start = NominalState::new(
            Vector::zeros(),
            Vector::zeros(),
            SO3::identity(),
            gyroscope_bias,
            accelerometer_bias,
        )
        .plus_error(-starting_error);
        let filter = flight_filter(start);

        let (filter, truth) = run_flight(filter, flight, step_count, timestep, &mut generator);
        total += filter.normalized_estimation_error_squared(truth).unwrap();
    }
    let mean = total / run_count as f64;

    // The two-sided 99 % band for the mean of 100 draws from a chi-squared with fifteen degrees of
    // freedom, rounded outward. It is wide on purpose: this guards against a structurally wrong
    // filter — a missing reset, a mis-scaled noise block — not against imperfect tuning.
    assert!(
        (11.6..18.8).contains(&mean),
        "average consistency score {mean} outside [11.6, 18.8]"
    );
}

/// A filter that has to learn both offsets from scratch, starting at zero and unsure of them.
fn learning_start() -> NominalState<f64> {
    NominalState::at_rest(SO3::identity())
}

#[test]
fn gyroscope_bias_converges_toward_truth() {
    // Thirty seconds, and a loose tolerance: the turn-rate offset about the vertical shows up only
    // through the heading aid, so it is the last of the three to settle.
    let timestep = 0.005;
    let run_time = 30.0;
    let step_count = (run_time / timestep) as usize;
    let mut generator = Pcg32::new(20260805);

    let (gyroscope_bias, accelerometer_bias) = Flight::standard_biases();
    let filter = flight_filter(learning_start());
    let mut starting_bias_spread = 0.0;
    for index in 9..12 {
        starting_bias_spread += filter.covariance()[(index, index)];
    }

    let flight = Flight::new(gyroscope_bias, accelerometer_bias);
    let (filter, _) = run_flight(filter, flight, step_count, timestep, &mut generator);

    let learned = filter.nominal_state().gyroscope_bias();
    for axis in 0..3 {
        let error = (learned[axis] - gyroscope_bias[axis]).abs();
        assert!(
            error < 0.2 * gyroscope_bias[axis].abs(),
            "[{axis}]: learned {} against truth {}",
            learned[axis],
            gyroscope_bias[axis]
        );
    }

    let mut ending_bias_spread = 0.0;
    for index in 9..12 {
        ending_bias_spread += filter.covariance()[(index, index)];
    }
    assert!(
        ending_bias_spread < 0.1 * starting_bias_spread,
        "bias spread only fell from {starting_bias_spread} to {ending_bias_spread}"
    );
}

#[test]
fn accelerometer_bias_converges_toward_truth() {
    let timestep = 0.005;
    let run_time = 30.0;
    let step_count = (run_time / timestep) as usize;
    let mut generator = Pcg32::new(20260805);

    let (gyroscope_bias, accelerometer_bias) = Flight::standard_biases();
    let flight = Flight::new(gyroscope_bias, accelerometer_bias);
    let filter = flight_filter(learning_start());
    let (filter, _) = run_flight(filter, flight, step_count, timestep, &mut generator);

    let learned = filter.nominal_state().accelerometer_bias();

    // Only the two horizontal components are checked. With no change in thrust, a push offset along
    // the vertical looks exactly like a small tilt, so nothing in this run can tell them apart.
    for axis in 0..2 {
        let error = (learned[axis] - accelerometer_bias[axis]).abs();
        assert!(
            error < 0.25 * accelerometer_bias[axis].abs(),
            "[{axis}]: learned {} against truth {}",
            learned[axis],
            accelerometer_bias[axis]
        );
    }
}

// ---- single precision -------------------------------------------------------

#[test]
fn single_precision_filter_runs_and_stays_finite() {
    let level = SO3::<f32>::identity();
    let imu_noise = ImuNoise {
        gyroscope_noise_density: 0.001_f32,
        accelerometer_noise_density: 0.001,
        gyroscope_bias_random_walk: 1e-6,
        accelerometer_bias_random_walk: 1e-6,
    };
    let tracker_spread = 0.03_f32;
    let mut filter = ErrorStateKalmanFilter::<3, f32>::new(
        NominalState::at_rest(level),
        Matrix::from_diagonal([0.01_f32; 15]),
        imu_noise,
        Matrix::from_diagonal([tracker_spread * tracker_spread; 3]),
    );

    let gravity_strength = 9.81_f32;
    let gyroscope_reading = Vector::new([0.0_f32, 0.0, 0.0]);
    let accelerometer_reading = Vector::new([0.0_f32, 0.0, gravity_strength]);
    let timestep = 0.001_f32;
    let step_count = 1000;
    for step in 0..step_count {
        filter
            .predict(gyroscope_reading, accelerometer_reading, timestep)
            .unwrap();
        if step % 100 == 0 {
            assert!(filter.covariance().is_finite());
            assert!(filter.covariance().is_symmetric());
        }
    }

    let state = filter.nominal_state();
    assert!(state.position().norm() < 1e-3);
    assert!(state.velocity().norm() < 1e-3);
}

#[test]
fn single_precision_conditioning_repairs_the_covariance() {
    let level = SO3::<f32>::identity();
    let mut generator = Pcg32::new(20260806);
    let directions = random_orthonormal_matrix(&mut generator);
    let mut spread = [0.05_f64; 15];
    spread[7] = -1e-6;
    let drifted = directions * Matrix::from_diagonal(spread) * directions.transpose();
    let drifted = Matrix::<15, 15, f32>::from_fn(|row, column| drifted[(row, column)] as f32);

    let imu_noise = ImuNoise {
        gyroscope_noise_density: 0.001_f32,
        accelerometer_noise_density: 0.001,
        gyroscope_bias_random_walk: 1e-6,
        accelerometer_bias_random_walk: 1e-6,
    };
    let mut filter = ErrorStateKalmanFilter::<3, f32>::new(
        NominalState::at_rest(level),
        drifted,
        imu_noise,
        Matrix::from_diagonal([0.001_f32; 3]),
    );

    let minimum_eigenvalue = 1e-7_f32;
    filter.condition_covariance(minimum_eigenvalue).unwrap();

    assert!(
        filter
            .covariance()
            .symmetric_eigendecomposition()
            .unwrap()
            .is_positive_definite()
    );
}
