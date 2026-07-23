//! Linear Kalman filter goldens, invariants, and error paths.

use multicalc::error::EstimationError;
use multicalc::estimation::{CovarianceUpdate, KalmanFilter};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Dual, Numeric};
use multicalc_testkit::tol::{Tol, assert_matrix_close, assert_vector_close};
use proptest::prelude::*;

/// A constant-velocity tracker over a 1 s step: position integrates velocity, position is measured.
fn constant_velocity_filter<T: Numeric>(
    initial_covariance: Matrix<2, 2, T>,
    process_noise: Matrix<2, 2, T>,
    measurement_noise: Matrix<1, 1, T>,
) -> KalmanFilter<2, 1, T> {
    KalmanFilter::new(
        Vector::new([T::ZERO, T::ZERO]),
        initial_covariance,
        Matrix::new([[T::ONE, T::ONE], [T::ZERO, T::ONE]]),
        Matrix::new([[T::ONE, T::ZERO]]),
        process_noise,
        measurement_noise,
    )
}

/// Builds a symmetric positive-definite matrix from arbitrary entries as `M·Mᵀ`, ridged so the
/// factorization is well conditioned rather than merely non-singular.
fn symmetric_positive_definite<const N: usize>(entries: &[f64]) -> Matrix<N, N> {
    let m = Matrix::<N, N>::from_fn(|row, column| entries[row * N + column]);
    m * m.transpose() + Matrix::<N, N>::identity().scale(0.25)
}

fn trace<const N: usize>(m: Matrix<N, N>) -> f64 {
    (0..N).map(|i| m.get(i, i).copied().unwrap()).sum()
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

    let t = Tol {
        abs: 1e-12,
        rel: 0.0,
    };
    assert_vector_close(&filter.state(), &Vector::new([5.0 / 3.0, 2.0 / 3.0]), t);
    assert_matrix_close(
        filter.covariance(),
        Matrix::new([[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0]]),
        1e-12,
    );
    assert_vector_close(&filter.innovation(), &Vector::new([1.0]), t);
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
            symmetric_positive_definite::<2>(&covariance_entries),
            Matrix::identity().scale(0.01),
            Matrix::new([[measurement_noise]]),
        );
        filter.predict();
        filter.update(Vector::new([measurement])).unwrap();

        let covariance = filter.covariance();
        prop_assert!(
            (covariance.get(0, 1).copied().unwrap() - covariance.get(1, 0).copied().unwrap()).abs()
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
            symmetric_positive_definite::<2>(&covariance_entries),
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
        let initial_covariance = symmetric_positive_definite::<2>(&covariance_entries);
        let process_noise = Matrix::<2, 2>::identity().scale(0.01);
        let noise = Matrix::new([[measurement_noise]]);

        let mut joseph = constant_velocity_filter::<f64>(initial_covariance, process_noise, noise);
        let mut naive = constant_velocity_filter::<f64>(initial_covariance, process_noise, noise)
            .with_covariance_update(CovarianceUpdate::Naive);

        for filter in [&mut joseph, &mut naive] {
            filter.predict();
            filter.update(Vector::new([measurement])).unwrap();
        }

        let t = Tol { abs: 1e-9, rel: 1e-9 };
        assert_vector_close(&joseph.state(), &naive.state(), t);
        assert_matrix_close(joseph.covariance(), naive.covariance(), 1e-9);
    }

    #[test]
    fn update_never_increases_covariance_trace(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
        measurement_noise in 0.05f64..5.0,
        measurement in -10.0f64..10.0,
    ) {
        let mut filter = constant_velocity_filter::<f64>(
            symmetric_positive_definite::<2>(&covariance_entries),
            Matrix::zeros(),
            Matrix::new([[measurement_noise]]),
        );
        filter.predict();
        let before = trace(filter.covariance());
        filter.update(Vector::new([measurement])).unwrap();

        prop_assert!(trace(filter.covariance()) <= before + 1e-9);
    }

    #[test]
    fn infinite_measurement_noise_leaves_state_unchanged(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
        measurement in -10.0f64..10.0,
    ) {
        let mut filter = constant_velocity_filter::<f64>(
            symmetric_positive_definite::<2>(&covariance_entries),
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
        let initial_covariance = symmetric_positive_definite::<2>(&covariance_entries);
        let mut filter = KalmanFilter::<2, 1>::new(
            Vector::new([state[0], state[1]]),
            initial_covariance,
            Matrix::identity(),
            Matrix::new([[1.0, 0.0]]),
            Matrix::zeros(),
            Matrix::new([[1.0]]),
        );
        filter.predict();

        assert_vector_close(&filter.state(), &Vector::new([state[0], state[1]]), Tol { abs: 1e-12, rel: 0.0 });
        assert_matrix_close(filter.covariance(), initial_covariance, 1e-12);
    }

    #[test]
    fn predict_with_zero_control_matches_plain_predict(
        covariance_entries in prop::collection::vec(-2.0f64..2.0, 4),
    ) {
        let initial_covariance = symmetric_positive_definite::<2>(&covariance_entries);
        let process_noise = Matrix::<2, 2>::identity().scale(0.01);
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
        Matrix::<2, 2, f32>::identity().scale(0.01),
        Matrix::new([[0.5]]),
    );

    for step in 0..64 {
        filter.predict();
        filter.update(Vector::new([step as f32 * 0.5])).unwrap();

        let covariance = filter.covariance();
        let scale = covariance.get(0, 0).copied().unwrap().abs().max(1.0);
        assert!(
            (covariance.get(0, 1).copied().unwrap() - covariance.get(1, 0).copied().unwrap()).abs()
                < 512.0 * f32::EPSILON * scale
        );
        assert!(covariance.cholesky().is_ok());
    }
}

// ----- Automatic differentiation -----

/// The posterior position after one predict/update, as a function of a process-noise scaling.
fn posterior_position<T: Numeric>(process_noise_scale: T) -> T {
    let mut filter = constant_velocity_filter::<T>(
        Matrix::identity(),
        Matrix::<2, 2, T>::identity().scale(process_noise_scale),
        Matrix::new([[T::from_f64(0.5)]]),
    );
    filter.predict();
    filter.update(Vector::new([T::from_f64(1.0)])).unwrap();
    filter.state().as_array()[0]
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
