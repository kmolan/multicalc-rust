//! Linear Kalman filter goldens, invariants, and error paths.

use multicalc::error::EstimationError;
use multicalc::estimation::{CovarianceUpdate, KalmanFilter, KalmanModel};
use multicalc::linear_algebra::{Matrix, Matrix2D, Vector};
use multicalc::scalar::{Dual, Numeric};
use multicalc_testkit::tol::{Tol, assert_matrix_close, assert_vector_close};
use proptest::prelude::*;

/// A constant-velocity tracker over a 1 s step: position integrates velocity, position is measured.
fn constant_velocity_filter<T: Numeric>(
    initial_covariance: Matrix2D<T>,
    process_noise: Matrix2D<T>,
    measurement_noise: Matrix<1, 1, T>,
) -> KalmanFilter<2, 1, T> {
    KalmanFilter::new(
        Vector::new([T::ZERO, T::ZERO]),
        initial_covariance,
        KalmanModel {
            state_transition: Matrix::new([[T::ONE, T::ONE], [T::ZERO, T::ONE]]),
            measurement_model: Matrix::new([[T::ONE, T::ZERO]]),
            process_noise,
            measurement_noise,
        },
    )
}

// ----- Goldens -----

/// Two steps of a noiseless-process constant-velocity tracker, worked by hand:
/// after `z = [1, 2]` the posterior is `x = [5/3, 2/3]`, `P = [[2/3, 1/3], [1/3, 1/3]]`.
#[test]
fn constant_velocity_two_steps_matches_hand_computation() {
    let mut filter =
        constant_velocity_filter::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[1.0]]));

    for measurement in [1.0, 2.0] {
        filter.predict();
        filter.update(Vector::new([measurement])).unwrap();
    }

    let tolerance = Tol {
        abs: 1e-12,
        rel: 0.0,
    };
    assert_vector_close(
        &filter.state(),
        &Vector::new([5.0 / 3.0, 2.0 / 3.0]),
        tolerance,
    );
    assert_matrix_close(
        filter.covariance(),
        Matrix::new([[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0]]),
        1e-12,
    );
    assert_vector_close(&filter.innovation(), &Vector::new([1.0]), tolerance);
    assert_matrix_close(filter.innovation_covariance(), Matrix::new([[3.0]]), 1e-12);
}

/// `yᵀ·S⁻¹·y` for the final step above: `y = 1`, `S = 3`.
#[test]
fn normalized_innovation_squared_matches_hand_computation() {
    let mut filter =
        constant_velocity_filter::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[1.0]]));

    for measurement in [1.0, 2.0] {
        filter.predict();
        filter.update(Vector::new([measurement])).unwrap();
    }

    let normalized = filter.normalized_innovation_squared().unwrap();
    assert!((normalized - 1.0 / 3.0).abs() < 1e-12);
}

// ----- Properties -----

proptest! {
    #[test]
    fn covariance_stays_symmetric_after_joseph_update(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
        measurement_noise in 0.05f64..5.0,
        measurement in -10.0f64..10.0,
    ) {
        let mut filter = constant_velocity_filter::<f64>(
            Matrix::<2, 2>::symmetric_positive_definite(&covariance_entries),
            Matrix::identity().scale(0.01),
            Matrix::new([[measurement_noise]]),
        );
        filter.predict();
        filter.update(Vector::new([measurement])).unwrap();

        let covariance = filter.covariance();
        prop_assert!(
            (covariance[(0, 1)] - covariance[(1, 0)]).abs()
            < 1e-12
        );
    }

    #[test]
    fn covariance_stays_positive_definite_after_update(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
        measurement_noise in 0.05f64..5.0,
        measurement in -10.0f64..10.0,
    ) {
        let mut filter = constant_velocity_filter::<f64>(
            Matrix::<2, 2>::symmetric_positive_definite(&covariance_entries),
            Matrix::identity().scale(0.01),
            Matrix::new([[measurement_noise]]),
        );
        filter.predict();
        filter.update(Vector::new([measurement])).unwrap();

        prop_assert!(filter.covariance().cholesky().is_ok());
    }

    #[test]
    fn joseph_and_naive_agree_on_well_conditioned_problems(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
        measurement_noise in 0.05f64..5.0,
        measurement in -10.0f64..10.0,
    ) {
        let initial_covariance = Matrix::<2, 2>::symmetric_positive_definite(&covariance_entries);
        let process_noise = Matrix2D::identity().scale(0.01);
        let noise = Matrix::new([[measurement_noise]]);

        let mut joseph = constant_velocity_filter::<f64>(initial_covariance, process_noise, noise);
        let mut naive = constant_velocity_filter::<f64>(initial_covariance, process_noise, noise)
            .with_covariance_update(CovarianceUpdate::Naive);

        for filter in [&mut joseph, &mut naive] {
            filter.predict();
            filter.update(Vector::new([measurement])).unwrap();
        }

        let tolerance = Tol { abs: 1e-9, rel: 1e-9 };
        assert_vector_close(&joseph.state(), &naive.state(), tolerance);
        assert_matrix_close(joseph.covariance(), naive.covariance(), 1e-9);
    }

    #[test]
    fn update_never_increases_covariance_trace(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
        measurement_noise in 0.05f64..5.0,
        measurement in -10.0f64..10.0,
    ) {
        let mut filter = constant_velocity_filter::<f64>(
            Matrix::<2, 2>::symmetric_positive_definite(&covariance_entries),
            Matrix::zeros(),
            Matrix::new([[measurement_noise]]),
        );
        filter.predict();
        let before = filter.covariance().trace();
        filter.update(Vector::new([measurement])).unwrap();

        prop_assert!(filter.covariance().trace() <= before + 1e-9);
    }

    #[test]
    fn infinite_measurement_noise_leaves_state_unchanged(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
        measurement in -10.0f64..10.0,
    ) {
        let mut filter = constant_velocity_filter::<f64>(
            Matrix::<2, 2>::symmetric_positive_definite(&covariance_entries),
            Matrix::zeros(),
            Matrix::new([[1e14]]),
        );
        filter.predict();
        let before = filter.state();
        filter.update(Vector::new([measurement])).unwrap();

        assert_vector_close(&filter.state(), &before, Tol { abs: 1e-6, rel: 1e-6 });
    }

    #[test]
    fn identity_transition_with_zero_process_noise_is_identity(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
        state in prop::collection::vec(-10.0f64..10.0, 2),
    ) {
        let initial_covariance = Matrix::<2, 2>::symmetric_positive_definite(&covariance_entries);
        let mut filter = KalmanFilter::<2, 1>::new(
            Vector::new([state[0], state[1]]),
            initial_covariance,
            KalmanModel {
                state_transition: Matrix::identity(),
                measurement_model: Matrix::new([[1.0, 0.0]]),
                process_noise: Matrix::zeros(),
                measurement_noise: Matrix::new([[1.0]]),
            },
        );
        filter.predict();

        assert_vector_close(&filter.state(), &Vector::new([state[0], state[1]]), Tol { abs: 1e-12, rel: 0.0 });
        assert_matrix_close(filter.covariance(), initial_covariance, 1e-12);
    }

    #[test]
    fn predict_with_zero_control_matches_plain_predict(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
    ) {
        let initial_covariance = Matrix::<2, 2>::symmetric_positive_definite(&covariance_entries);
        let process_noise = Matrix2D::identity().scale(0.01);
        let noise = Matrix::new([[1.0]]);

        let mut plain = constant_velocity_filter::<f64>(initial_covariance, process_noise, noise);
        let mut driven = constant_velocity_filter::<f64>(initial_covariance, process_noise, noise);
        plain.set_state(Vector::new([1.0, 2.0]));
        driven.set_state(Vector::new([1.0, 2.0]));

        plain.predict();
        driven.predict_with_control(Matrix::<2, 1>::new([[0.5], [1.0]]), Vector::new([0.0]));

        assert_vector_close(&plain.state(), &driven.state(), Tol { abs: 1e-12, rel: 0.0 });
        assert_matrix_close(plain.covariance(), driven.covariance(), 1e-12);
    }
}

// ----- f32 identities -----

/// f32 correctness is asserted via identities, never against an f64 golden: the covariance stays
/// symmetric and positive definite across a run.
#[test]
fn covariance_stays_symmetric_and_positive_definite_in_single_precision() {
    let mut filter = constant_velocity_filter::<f32>(
        Matrix::identity(),
        Matrix2D::<f32>::identity().scale(0.01),
        Matrix::new([[0.5]]),
    );

    for step in 0..64 {
        filter.predict();
        filter.update(Vector::new([step as f32 * 0.5])).unwrap();

        let covariance = filter.covariance();
        let scale = covariance[(0, 0)].abs().max(1.0);
        assert!((covariance[(0, 1)] - covariance[(1, 0)]).abs() < 512.0 * f32::EPSILON * scale);
        assert!(covariance.cholesky().is_ok());
    }
}

// ----- Automatic differentiation -----

/// The posterior position after one predict/update, as a function of a process-noise scaling.
fn posterior_position<T: Numeric>(process_noise_scale: T) -> T {
    let mut filter = constant_velocity_filter::<T>(
        Matrix::identity(),
        Matrix2D::<T>::identity().scale(process_noise_scale),
        Matrix::new([[T::from_f64(0.5)]]),
    );
    filter.predict();
    filter.update(Vector::new([T::from_f64(1.0)])).unwrap();
    filter.state()[0]
}

#[test]
fn posterior_derivative_in_process_noise_matches_finite_difference() {
    let scale = 0.05;
    let automatic = posterior_position(Dual::new(scale, 1.0)).deriv;

    let step = 1e-6;
    let finite_difference =
        (posterior_position(scale + step) - posterior_position(scale - step)) / (2.0 * step);

    assert!(automatic.is_finite());
    assert!((automatic - finite_difference).abs() < 1e-7);
}

// ----- Error paths -----

#[test]
fn non_finite_measurement_is_rejected() {
    let mut filter =
        constant_velocity_filter::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[1.0]]));
    filter.predict();

    assert_eq!(
        filter.update(Vector::new([f64::NAN])),
        Err(EstimationError::NonFinite)
    );
}

#[test]
fn singular_innovation_covariance_is_rejected() {
    let mut filter =
        constant_velocity_filter::<f64>(Matrix::zeros(), Matrix::zeros(), Matrix::zeros());

    assert_eq!(
        filter.update(Vector::new([1.0])),
        Err(EstimationError::NotPositiveDefinite)
    );
}

#[test]
fn normalized_innovation_squared_before_first_update_is_rejected() {
    let filter =
        constant_velocity_filter::<f64>(Matrix::identity(), Matrix::zeros(), Matrix::new([[1.0]]));

    assert_eq!(
        filter.normalized_innovation_squared(),
        Err(EstimationError::NotPositiveDefinite)
    );
}

// ----- Innovation gate -----

const TRACK_STEP_COUNT: usize = 20;
const OUTLIER_STEP: usize = 10;
/// χ²₁ at 99.73 %.
const ONE_DEGREE_GATE: f64 = 9.0;

/// The position reading at `step` on a unit-speed track, with a 100 m outlier at `OUTLIER_STEP`.
fn track_reading(step: usize) -> Vector<1, f64> {
    let outlier = if step == OUTLIER_STEP { 100.0 } else { 0.0 };
    Vector::new([step as f64 + outlier])
}

fn tracking_filter() -> KalmanFilter<2, 1> {
    constant_velocity_filter(
        Matrix2D::identity(),
        Matrix::from_diagonal([0.01, 0.01]),
        Matrix::new([[0.1]]),
    )
}

/// Runs the track. `skip_outlier` drops the update at `OUTLIER_STEP`.
fn run_track(mut filter: KalmanFilter<2, 1>, skip_outlier: bool) -> KalmanFilter<2, 1> {
    for step in 1..=TRACK_STEP_COUNT {
        filter.predict();
        if skip_outlier && step == OUTLIER_STEP {
            continue;
        }
        match filter.update(track_reading(step)) {
            Ok(()) => {}
            Err(EstimationError::MeasurementRejected) => assert_eq!(step, OUTLIER_STEP),
            Err(error) => panic!("update failed at step {step}: {error:?}"),
        }
    }
    filter
}

#[test]
fn gate_rejects_an_outlier_and_matches_the_run_that_skips_it() {
    let gated = run_track(
        tracking_filter()
            .with_innovation_gate(ONE_DEGREE_GATE)
            .unwrap(),
        false,
    );
    let skipping = run_track(tracking_filter(), true);
    let ungated = run_track(tracking_filter(), false);

    assert_eq!(gated.state(), skipping.state());
    assert_eq!(gated.covariance(), skipping.covariance());
    assert!((gated.state()[0] - TRACK_STEP_COUNT as f64).abs() < 1e-4);
    assert!((ungated.state()[0] - TRACK_STEP_COUNT as f64).abs() > 0.1);
}

#[test]
fn rejected_update_keeps_the_state_and_records_the_innovation() {
    let mut filter = tracking_filter()
        .with_innovation_gate(ONE_DEGREE_GATE)
        .unwrap();
    for step in 1..OUTLIER_STEP {
        filter.predict();
        filter.update(track_reading(step)).unwrap();
    }
    filter.predict();
    let state = filter.state();
    let covariance = filter.covariance();

    assert_eq!(
        filter.update(track_reading(OUTLIER_STEP)),
        Err(EstimationError::MeasurementRejected)
    );
    assert_eq!(filter.state(), state);
    assert_eq!(filter.covariance(), covariance);
    assert!(filter.normalized_innovation_squared().unwrap() > 1e4);
    assert!(filter.innovation()[0] > 99.0);
}

#[test]
fn wide_gate_matches_no_gate() {
    let wide = run_track(tracking_filter().with_innovation_gate(1e12).unwrap(), false);
    let ungated = run_track(tracking_filter(), false);

    assert_eq!(wide.state(), ungated.state());
    assert_eq!(wide.covariance(), ungated.covariance());
}

#[test]
fn gate_rejects_unusable_thresholds() {
    for threshold in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            tracking_filter()
                .with_innovation_gate(threshold)
                .unwrap_err(),
            EstimationError::InvalidTuning
        );
    }
    assert_eq!(tracking_filter().innovation_gate(), None);
    assert_eq!(
        tracking_filter()
            .with_innovation_gate(ONE_DEGREE_GATE)
            .unwrap()
            .innovation_gate(),
        Some(ONE_DEGREE_GATE)
    );
}
