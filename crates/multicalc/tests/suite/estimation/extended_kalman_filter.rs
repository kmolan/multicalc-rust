//! Extended Kalman filter goldens, invariants, and error paths.

use multicalc::error::{DiffError, EstimationError};
use multicalc::estimation::{CovarianceUpdate, ExtendedKalmanFilter, KalmanFilter};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::numerical_derivative::finite_difference::FiniteDifferenceMulti;
use multicalc::numerical_derivative::mode::FiniteDifferenceMode;
use multicalc::scalar::{Dual, Numeric, VectorFn};
use multicalc_testkit::tol::{Tol, assert_matrix_close, assert_vector_close};
use proptest::prelude::*;

/// Unicycle motion: [x, y, heading] driven by a forward and an angular velocity over one step.
struct UnicycleMotion {
    timestep: f64,
    forward_velocity: f64,
    angular_velocity: f64,
}

impl VectorFn<3, 3> for UnicycleMotion {
    fn eval<S: Numeric>(&self, state: &[S; 3]) -> [S; 3] {
        let timestep = S::from_f64(self.timestep);
        let forward_velocity = S::from_f64(self.forward_velocity);
        let angular_velocity = S::from_f64(self.angular_velocity);
        let heading = state[2];
        [
            state[0] + forward_velocity * heading.cos() * timestep,
            state[1] + forward_velocity * heading.sin() * timestep,
            heading + angular_velocity * timestep,
        ]
    }
}

/// Range and bearing to a known landmark, from a [x, y, heading] pose.
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

/// The constant-velocity transition `[[1, 1], [0, 1]]` written as a function, so the extended filter
/// can be given the same model the linear filter gets as a matrix.
struct ConstantVelocityMotion;
impl VectorFn<2, 2> for ConstantVelocityMotion {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [state[0] + state[1], state[1]]
    }
}

/// The measurement model `[[1, 0]]`: position is measured, velocity is not.
struct PositionMeasurement;
impl VectorFn<2, 1> for PositionMeasurement {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
        [state[0]]
    }
}

/// A process model that injects a non-finite component, to exercise the predict guard.
struct NonFiniteMotion;
impl VectorFn<3, 3> for NonFiniteMotion {
    fn eval<S: Numeric>(&self, state: &[S; 3]) -> [S; 3] {
        [state[0], state[1], S::from_f64(f64::NAN)]
    }
}

/// A symmetric positive-definite matrix from arbitrary entries as `M·Mᵀ`, ridged so the
/// factorization is well conditioned rather than merely non-singular.
fn symmetric_positive_definite<const N: usize>(entries: &[f64]) -> Matrix<N, N> {
    let m = Matrix::<N, N>::from_fn(|row, column| entries[row * N + column]);
    m * m.transpose() + Matrix::<N, N>::identity().scale(0.25)
}

fn trace<const N: usize>(m: Matrix<N, N>) -> f64 {
    (0..N).map(|i| m.get(i, i).copied().unwrap()).sum()
}

// ----- Agreement with the linear filter -----

/// Given linear models, the extended filter must reproduce the linear filter exactly: its Jacobians
/// are then the constant matrices the linear filter is handed.
#[test]
fn extended_filter_with_linear_models_matches_linear_filter() {
    let initial_covariance = Matrix::<2, 2>::identity();
    let process_noise = Matrix::<2, 2>::identity().scale(0.01);
    let measurement_noise = Matrix::new([[0.5]]);

    let mut extended = ExtendedKalmanFilter::<2, 1>::new(
        Vector::new([0.0, 0.0]),
        initial_covariance,
        process_noise,
        measurement_noise,
    );
    let mut linear = KalmanFilter::<2, 1>::new(
        Vector::new([0.0, 0.0]),
        initial_covariance,
        Matrix::new([[1.0, 1.0], [0.0, 1.0]]),
        Matrix::new([[1.0, 0.0]]),
        process_noise,
        measurement_noise,
    );

    for step in 0..8 {
        let measurement = Vector::new([step as f64 * 0.5]);
        extended.predict(&ConstantVelocityMotion).unwrap();
        extended.update(&PositionMeasurement, measurement).unwrap();
        linear.predict();
        linear.update(measurement).unwrap();
    }

    let t = Tol {
        abs: 1e-12,
        rel: 0.0,
    };
    assert_vector_close(&extended.state(), &linear.state(), t);
    assert_matrix_close(extended.covariance(), linear.covariance(), 1e-12);
    assert_vector_close(&extended.innovation(), &linear.innovation(), t);
    assert_matrix_close(
        extended.innovation_covariance(),
        linear.innovation_covariance(),
        1e-12,
    );
}

// ----- Goldens -----

/// One update of the landmark filter from the pose [0, 0, 0] with the landmark at (3, 4) — a 3-4-5
/// triangle, so `H` is exact in decimals and `S` is diagonal. Worked in exact rational arithmetic.
#[test]
fn landmark_range_and_bearing_update_matches_hand_computation() {
    let mut filter = ExtendedKalmanFilter::<3, 2>::new(
        Vector::new([0.0, 0.0, 0.0]),
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::identity(),
    );
    let model = LandmarkRangeAndBearing {
        landmark_x: 3.0,
        landmark_y: 4.0,
    };

    // h(x) = [5, atan2(4, 3)], so this measurement makes the residual exactly [0.5, 0.1].
    let measurement = Vector::new([5.5, 4.0_f64.atan2(3.0) + 0.1]);
    filter.update(&model, measurement).unwrap();

    assert_vector_close(
        &filter.state(),
        &Vector::new([-29.0 / 204.0, -7.0 / 34.0, -5.0 / 102.0]),
        Tol {
            abs: 1e-12,
            rel: 0.0,
        },
    );
    assert_matrix_close(
        filter.covariance(),
        Matrix::new([
            [2059.0 / 2550.0, -98.0 / 425.0, 4.0 / 51.0],
            [-98.0 / 425.0, 286.0 / 425.0, -1.0 / 17.0],
            [4.0 / 51.0, -1.0 / 17.0, 26.0 / 51.0],
        ]),
        1e-12,
    );
    assert_vector_close(
        &filter.innovation(),
        &Vector::new([0.5, 0.1]),
        Tol {
            abs: 1e-12,
            rel: 0.0,
        },
    );
    assert_matrix_close(
        filter.innovation_covariance(),
        Matrix::new([[2.0, 0.0], [0.0, 2.04]]),
        1e-12,
    );
}

/// One predict of the unicycle from [0, 0, 0] at heading 0 with `v·dt = 1`: the state moves one metre
/// along x, and `F = [[1, 0, 0], [0, 1, 1], [0, 0, 1]]`, so `P = F·I·Fᵀ`.
#[test]
fn unicycle_predict_matches_hand_computation() {
    let mut filter = ExtendedKalmanFilter::<3, 2>::new(
        Vector::new([0.0, 0.0, 0.0]),
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::identity(),
    );
    let motion = UnicycleMotion {
        timestep: 0.5,
        forward_velocity: 2.0,
        angular_velocity: 0.0,
    };
    filter.predict(&motion).unwrap();

    assert_vector_close(
        &filter.state(),
        &Vector::new([1.0, 0.0, 0.0]),
        Tol {
            abs: 1e-12,
            rel: 0.0,
        },
    );
    assert_matrix_close(
        filter.covariance(),
        Matrix::new([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 1.0]]),
        1e-12,
    );
}

// ----- Properties -----

proptest! {
    #[test]
    fn covariance_stays_symmetric_and_positive_definite_after_joseph_update(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 9),
        measurement_noise in 0.05f64..5.0,
        range_error in -1.0f64..1.0,
        bearing_error in -0.5f64..0.5,
    ) {
        let mut filter = ExtendedKalmanFilter::<3, 2>::new(
            Vector::new([0.0, 0.0, 0.0]),
            symmetric_positive_definite::<3>(&covariance_entries),
            Matrix::<3, 3>::identity().scale(0.01),
            Matrix::<2, 2>::identity().scale(measurement_noise),
        );
        let model = LandmarkRangeAndBearing { landmark_x: 3.0, landmark_y: 4.0 };
        let measurement = Vector::new([
            5.0 + range_error,
            4.0_f64.atan2(3.0) + bearing_error,
        ]);
        filter.update(&model, measurement).unwrap();

        let covariance = filter.covariance();
        prop_assert!(
            (covariance.get(0, 1).copied().unwrap() - covariance.get(1, 0).copied().unwrap()).abs()
            < 1e-12
        );
        prop_assert!(
            (covariance.get(0, 2).copied().unwrap() - covariance.get(2, 0).copied().unwrap()).abs()
            < 1e-12
        );
        prop_assert!(
            (covariance.get(1, 2).copied().unwrap() - covariance.get(2, 1).copied().unwrap()).abs()
            < 1e-12
        );
        prop_assert!(covariance.cholesky().is_ok());
    }

    #[test]
    fn update_never_increases_covariance_trace(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 9),
        measurement_noise in 0.05f64..5.0,
        range_error in -1.0f64..1.0,
        bearing_error in -0.5f64..0.5,
    ) {
        let mut filter = ExtendedKalmanFilter::<3, 2>::new(
            Vector::new([0.0, 0.0, 0.0]),
            symmetric_positive_definite::<3>(&covariance_entries),
            Matrix::zeros(),
            Matrix::<2, 2>::identity().scale(measurement_noise),
        );
        let model = LandmarkRangeAndBearing { landmark_x: 3.0, landmark_y: 4.0 };
        let measurement = Vector::new([
            5.0 + range_error,
            4.0_f64.atan2(3.0) + bearing_error,
        ]);
        let before = trace(filter.covariance());
        filter.update(&model, measurement).unwrap();

        prop_assert!(trace(filter.covariance()) <= before + 1e-9);
    }

    #[test]
    fn infinite_measurement_noise_leaves_state_unchanged(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 9),
        range_error in -1.0f64..1.0,
        bearing_error in -0.5f64..0.5,
    ) {
        let mut filter = ExtendedKalmanFilter::<3, 2>::new(
            Vector::new([0.0, 0.0, 0.0]),
            symmetric_positive_definite::<3>(&covariance_entries),
            Matrix::zeros(),
            Matrix::<2, 2>::identity().scale(1e14),
        );
        let model = LandmarkRangeAndBearing { landmark_x: 3.0, landmark_y: 4.0 };
        let measurement = Vector::new([
            5.0 + range_error,
            4.0_f64.atan2(3.0) + bearing_error,
        ]);
        let before = filter.state();
        filter.update(&model, measurement).unwrap();

        assert_vector_close(&filter.state(), &before, Tol { abs: 1e-6, rel: 1e-6 });
    }

    #[test]
    fn update_with_residual_agrees_with_update_on_a_plain_residual(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 9),
        measurement_noise in 0.05f64..5.0,
        range_error in -1.0f64..1.0,
        bearing_error in -0.5f64..0.5,
    ) {
        let initial_covariance = symmetric_positive_definite::<3>(&covariance_entries);
        let process_noise = Matrix::<3, 3>::identity().scale(0.01);
        let noise = Matrix::<2, 2>::identity().scale(measurement_noise);
        let model = LandmarkRangeAndBearing { landmark_x: 3.0, landmark_y: 4.0 };
        let measurement = Vector::new([
            5.0 + range_error,
            4.0_f64.atan2(3.0) + bearing_error,
        ]);

        let mut direct = ExtendedKalmanFilter::<3, 2>::new(
            Vector::new([0.0, 0.0, 0.0]), initial_covariance, process_noise, noise,
        );
        let mut seamed = ExtendedKalmanFilter::<3, 2>::new(
            Vector::new([0.0, 0.0, 0.0]), initial_covariance, process_noise, noise,
        );

        direct.update(&model, measurement).unwrap();
        let predicted = Vector::new(model.eval(seamed.state().as_array()));
        seamed.update_with_residual(&model, measurement - predicted).unwrap();

        let t = Tol { abs: 1e-12, rel: 0.0 };
        assert_vector_close(&direct.state(), &seamed.state(), t);
        assert_matrix_close(direct.covariance(), seamed.covariance(), 1e-12);
    }

    #[test]
    fn zero_velocity_motion_leaves_the_pose_unchanged(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 9),
    ) {
        let initial_state = Vector::new([1.0, -2.0, 0.7]);
        let initial_covariance = symmetric_positive_definite::<3>(&covariance_entries);
        let mut filter = ExtendedKalmanFilter::<3, 2>::new(
            initial_state,
            initial_covariance,
            Matrix::zeros(),
            Matrix::<2, 2>::identity(),
        );
        let motion = UnicycleMotion { timestep: 0.1, forward_velocity: 0.0, angular_velocity: 0.0 };
        filter.predict(&motion).unwrap();

        assert_vector_close(&filter.state(), &initial_state, Tol { abs: 1e-12, rel: 0.0 });
        assert_matrix_close(filter.covariance(), initial_covariance, 1e-12);
    }

    #[test]
    fn joseph_and_naive_agree_on_well_conditioned_problems(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 9),
        measurement_noise in 0.05f64..5.0,
        range_error in -1.0f64..1.0,
        bearing_error in -0.5f64..0.5,
    ) {
        let initial_covariance = symmetric_positive_definite::<3>(&covariance_entries);
        let process_noise = Matrix::<3, 3>::identity().scale(0.01);
        let noise = Matrix::<2, 2>::identity().scale(measurement_noise);
        let model = LandmarkRangeAndBearing { landmark_x: 3.0, landmark_y: 4.0 };
        let measurement = Vector::new([
            5.0 + range_error,
            4.0_f64.atan2(3.0) + bearing_error,
        ]);

        let mut joseph = ExtendedKalmanFilter::<3, 2>::new(
            Vector::new([0.0, 0.0, 0.0]), initial_covariance, process_noise, noise,
        );
        let mut naive = ExtendedKalmanFilter::<3, 2>::new(
            Vector::new([0.0, 0.0, 0.0]), initial_covariance, process_noise, noise,
        )
        .with_covariance_update(CovarianceUpdate::Naive);

        joseph.update(&model, measurement).unwrap();
        naive.update(&model, measurement).unwrap();

        let t = Tol { abs: 1e-9, rel: 1e-9 };
        assert_vector_close(&joseph.state(), &naive.state(), t);
        assert_matrix_close(joseph.covariance(), naive.covariance(), 1e-9);
    }
}

// ----- f32 identities -----

/// f32 correctness is asserted via identities, never against an f64 golden: the covariance stays
/// symmetric and positive definite across a run.
#[test]
fn covariance_stays_symmetric_and_positive_definite_in_single_precision() {
    let mut filter = ExtendedKalmanFilter::<3, 2, f32>::new(
        Vector::new([0.0, 0.0, 0.0]),
        Matrix::identity(),
        Matrix::<3, 3, f32>::identity().scale(0.01),
        Matrix::<2, 2, f32>::identity().scale(0.5),
    );
    let motion = UnicycleMotion {
        timestep: 0.05,
        forward_velocity: 0.4,
        angular_velocity: 0.1,
    };
    let model = LandmarkRangeAndBearing {
        landmark_x: 3.0,
        landmark_y: 4.0,
    };

    for step in 0..64 {
        filter.predict(&motion).unwrap();
        let bearing = 4.0_f32.atan2(3.0) + (step as f32) * 0.001;
        filter.update(&model, Vector::new([5.0, bearing])).unwrap();

        let covariance = filter.covariance();
        let scale = covariance.get(0, 0).copied().unwrap().abs().max(1.0);
        for (row, column) in [(0, 1), (0, 2), (1, 2)] {
            assert!(
                (covariance.get(row, column).copied().unwrap()
                    - covariance.get(column, row).copied().unwrap())
                .abs()
                    < 512.0 * f32::EPSILON * scale
            );
        }
        assert!(covariance.cholesky().is_ok());
    }
}

// ----- Automatic differentiation -----

/// The posterior x of the landmark filter after one update, as a function of a
/// measurement-noise scaling.
fn posterior_x<T: Numeric>(measurement_noise_scale: T) -> T {
    let mut filter = ExtendedKalmanFilter::<3, 2, T>::new(
        Vector::new([T::ZERO, T::ZERO, T::ZERO]),
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::<2, 2, T>::identity().scale(measurement_noise_scale),
    );
    let model = LandmarkRangeAndBearing {
        landmark_x: 3.0,
        landmark_y: 4.0,
    };
    let measurement = Vector::new([T::from_f64(5.5), T::from_f64(4.0_f64.atan2(3.0) + 0.1)]);
    filter.update(&model, measurement).unwrap();
    filter.state().as_array()[0]
}

#[test]
fn posterior_derivative_in_measurement_noise_matches_finite_difference() {
    let scale = 1.0;
    let automatic = posterior_x(Dual::new(scale, 1.0)).deriv;

    let step = 1e-6;
    let finite_difference = (posterior_x(scale + step) - posterior_x(scale - step)) / (2.0 * step);

    assert!(automatic.is_finite());
    assert!((automatic - finite_difference).abs() < 1e-7);
}

// ----- Error paths -----

#[test]
fn non_finite_measurement_is_rejected() {
    let mut filter = ExtendedKalmanFilter::<3, 2>::new(
        Vector::new([0.0, 0.0, 0.0]),
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::identity(),
    );
    let model = LandmarkRangeAndBearing {
        landmark_x: 3.0,
        landmark_y: 4.0,
    };

    assert_eq!(
        filter.update(&model, Vector::new([f64::NAN, 0.0])),
        Err(EstimationError::NonFinite)
    );
}

#[test]
fn non_finite_process_model_is_rejected() {
    let mut filter = ExtendedKalmanFilter::<3, 2>::new(
        Vector::new([0.0, 0.0, 0.0]),
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::identity(),
    );

    assert_eq!(
        filter.predict(&NonFiniteMotion),
        Err(EstimationError::NonFinite)
    );
}

#[test]
fn singular_innovation_covariance_is_rejected() {
    let mut filter = ExtendedKalmanFilter::<3, 2>::new(
        Vector::new([0.0, 0.0, 0.0]),
        Matrix::zeros(),
        Matrix::zeros(),
        Matrix::zeros(),
    );
    let model = LandmarkRangeAndBearing {
        landmark_x: 3.0,
        landmark_y: 4.0,
    };

    assert_eq!(
        filter.update(&model, Vector::new([5.0, 4.0_f64.atan2(3.0)])),
        Err(EstimationError::NotPositiveDefinite)
    );
}

#[test]
fn normalized_innovation_squared_before_first_update_is_rejected() {
    let filter = ExtendedKalmanFilter::<3, 2>::new(
        Vector::new([0.0, 0.0, 0.0]),
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::identity(),
    );

    assert_eq!(
        filter.normalized_innovation_squared(),
        Err(EstimationError::NotPositiveDefinite)
    );
}

/// The autodiff path cannot fail, so `Diff` is reachable only through a finite-difference derivator
/// with an invalid step.
#[test]
fn zero_step_size_derivator_is_rejected() {
    let derivator = FiniteDifferenceMulti::from_parameters(0.0, FiniteDifferenceMode::Central, 1.0);
    let mut filter = ExtendedKalmanFilter::<3, 2, f64, FiniteDifferenceMulti<f64>>::from_derivator(
        Vector::new([0.0, 0.0, 0.0]),
        Matrix::identity(),
        Matrix::zeros(),
        Matrix::identity(),
        derivator,
    );
    let motion = UnicycleMotion {
        timestep: 0.5,
        forward_velocity: 2.0,
        angular_velocity: 0.0,
    };

    assert_eq!(
        filter.predict(&motion),
        Err(EstimationError::Diff(DiffError::StepSizeZero))
    );
}
