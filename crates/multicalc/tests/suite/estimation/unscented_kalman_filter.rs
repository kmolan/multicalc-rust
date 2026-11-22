//! Unscented Kalman filter identities, tuning rejection, and error paths.

use multicalc::error::EstimationError;
use multicalc::estimation::{KalmanFilter, KalmanModel, UnscentedKalmanFilter};
use multicalc::linear_algebra::{Matrix, Matrix2D, Vector};
use multicalc::scalar::{Dual, Numeric, VectorFn};
use multicalc_testkit::tol::{Tol, assert_vector_close};
use proptest::prelude::*;

/// The constant-velocity transition `[[1, 1], [0, 1]]` written as a function, so the unscented
/// filter can be given the same model the linear filter gets as a matrix.
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

/// Range to a known landmark at (3, 4) from an [x, y] position.
struct RangeToLandmark;
impl VectorFn<2, 1> for RangeToLandmark {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
        let to_landmark_x = S::from_f64(3.0) - state[0];
        let to_landmark_y = S::from_f64(4.0) - state[1];
        [(to_landmark_x * to_landmark_x + to_landmark_y * to_landmark_y).sqrt()]
    }
}

/// A target that does not move: the position carries over unchanged.
struct Stationary2D;
impl VectorFn<2, 2> for Stationary2D {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [state[0], state[1]]
    }
}

/// Heading, measured by a compass, and the same function as the process model that holds it.
struct Compass;
impl VectorFn<1, 1> for Compass {
    fn eval<S: Numeric>(&self, state: &[S; 1]) -> [S; 1] {
        [state[0]]
    }
}

/// A process model that injects a non-finite component, to exercise the predict guard.
struct NotFinite;
impl VectorFn<2, 2> for NotFinite {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [S::from_f64(f64::NAN), state[1]]
    }
}

/// Subtracts whole turns to fold an angle into a ±π band.
fn wrap_to_pi(angle: f64) -> f64 {
    angle - core::f64::consts::TAU * (angle / core::f64::consts::TAU).round()
}

/// Builds the [x, y] landmark filter every nonlinear case below starts from.
fn landmark_filter() -> UnscentedKalmanFilter<2, 1> {
    UnscentedKalmanFilter::<2, 1>::new(
        Vector::new([0.0, 0.0]),
        Matrix2D::identity(),
        Matrix2D::identity().scale(0.01),
        Matrix::new([[0.1]]),
    )
}

// ----- Agreement with the linear filter -----

/// Steps a linear and an unscented filter through the same four measurements on the same models,
/// and returns the two final beliefs so the caller can compare them.
fn run_both_filters(
    unscented: UnscentedKalmanFilter<2, 1>,
    process_noise: Matrix2D,
) -> (UnscentedKalmanFilter<2, 1>, KalmanFilter<2, 1>) {
    let measurement_noise = Matrix::new([[0.5]]);

    let mut unscented = unscented;
    let mut linear = KalmanFilter::<2, 1>::new(
        Vector::new([0.0, 0.0]),
        Matrix2D::identity(),
        KalmanModel {
            state_transition: Matrix::new([[1.0, 1.0], [0.0, 1.0]]),
            measurement_model: Matrix::new([[1.0, 0.0]]),
            process_noise,
            measurement_noise,
        },
    );

    for measurement in [0.5, 1.0, 1.5, 2.0] {
        let measurement = Vector::new([measurement]);
        unscented.predict(&ConstantVelocityMotion).unwrap();
        unscented.update(&PositionMeasurement, measurement).unwrap();
        linear.predict();
        linear.update(measurement).unwrap();
    }
    (unscented, linear)
}

fn linear_filter_starting_point(process_noise: Matrix2D) -> UnscentedKalmanFilter<2, 1> {
    UnscentedKalmanFilter::<2, 1>::new(
        Vector::new([0.0, 0.0]),
        Matrix2D::identity(),
        process_noise,
        Matrix::new([[0.5]]),
    )
}

/// The identity is asserted with no process noise, which is where it is exact — see
/// `process_noise_is_the_only_gap_against_the_linear_filter` for why.
///
/// On a linear model the middle point lands exactly on the new mean, so its deviation is zero and
/// the weight on it — about −10⁶ at the default spread — multiplies nothing. What is left is an
/// exact identity in real arithmetic, and `1e-8` is what floating point leaves of it once that
/// near-cancelling term is rounded.
#[test]
fn reduces_to_the_linear_filter_at_default_scaling() {
    let (unscented, linear) = run_both_filters(
        linear_filter_starting_point(Matrix2D::zeros()),
        Matrix2D::zeros(),
    );

    assert_vector_close(
        &unscented.state(),
        &linear.state(),
        Tol {
            abs: 1e-8,
            rel: 1e-8,
        },
    );
    for row in 0..2 {
        for column in 0..2 {
            assert!(
                (unscented.covariance()[(row, column)] - linear.covariance()[(row, column)]).abs()
                    < 1e-8
            );
        }
    }
}

/// With the points spread wide the weights are all of ordinary size, and the same identity holds
/// to nearly the last bit — the tolerance gap between this test and the last one is the price of
/// the tight default spread.
#[test]
fn reduces_to_the_linear_filter_at_a_wide_spread() {
    let unscented = linear_filter_starting_point(Matrix2D::zeros())
        .with_scaling(1.0, 2.0, 0.0)
        .unwrap();
    let (unscented, linear) = run_both_filters(unscented, Matrix2D::zeros());

    assert_vector_close(
        &unscented.state(),
        &linear.state(),
        Tol {
            abs: 1e-12,
            rel: 1e-12,
        },
    );
    for row in 0..2 {
        for column in 0..2 {
            assert!(
                (unscented.covariance()[(row, column)] - linear.covariance()[(row, column)]).abs()
                    < 1e-12
            );
        }
    }
}

/// Add process noise and the two filters no longer agree exactly, by design: `update` works from
/// the points `predict` left behind, and those were placed before the process noise was folded
/// into the covariance, so the spread it sees is short by that much. The gap is the size of the
/// process noise, not of the rounding — it does not shrink when the points are spread wide, which
/// is what tells the two apart. Close is what is wanted here; exact is what the other two filters
/// are for.
#[test]
fn process_noise_is_the_only_gap_against_the_linear_filter() {
    let process_noise = Matrix2D::identity().scale(0.01);
    let (unscented, linear) =
        run_both_filters(linear_filter_starting_point(process_noise), process_noise);

    assert_vector_close(
        &unscented.state(),
        &linear.state(),
        Tol {
            abs: 1e-3,
            rel: 1e-3,
        },
    );
    for row in 0..2 {
        for column in 0..2 {
            let gap =
                (unscented.covariance()[(row, column)] - linear.covariance()[(row, column)]).abs();
            assert!(gap < 0.011);
        }
    }
}

// ----- Nonlinear behaviour -----

#[test]
fn tracks_a_nonlinear_measurement() {
    let mut filter = landmark_filter();
    filter.predict(&Stationary2D).unwrap();
    filter.update(&RangeToLandmark, Vector::new([5.5])).unwrap();

    // A longer range than predicted pushes the estimate away from the landmark at (3, 4).
    assert!(filter.state()[0] < 0.0);
    assert!(filter.state()[1] < 0.0);
    assert!(filter.covariance().trace() < 2.0);
}

/// Bitwise, not approximate: that is what the symmetrizing step guarantees, and an approximate
/// check would not notice if it were dropped.
#[test]
fn covariance_stays_exactly_symmetric() {
    let mut filter = landmark_filter();
    for _ in 0..20 {
        filter.predict(&Stationary2D).unwrap();
        filter.update(&RangeToLandmark, Vector::new([5.5])).unwrap();

        let covariance = filter.covariance();
        assert_eq!(covariance[(0, 1)], covariance[(1, 0)]);
    }
}

#[test]
fn predict_grows_uncertainty_and_update_shrinks_it() {
    let mut filter = landmark_filter();
    let before = filter.covariance().trace();

    filter.predict(&Stationary2D).unwrap();
    let after_predict = filter.covariance().trace();
    assert!(after_predict > before);

    filter.update(&RangeToLandmark, Vector::new([5.5])).unwrap();
    assert!(filter.covariance().trace() < before);
}

/// The reading is reported against the initial estimate, but nothing moves.
#[test]
fn update_before_predict_leaves_the_estimate_alone() {
    let mut filter = landmark_filter();
    filter.update(&RangeToLandmark, Vector::new([5.5])).unwrap();

    assert_vector_close(
        &filter.state(),
        &Vector::new([0.0, 0.0]),
        Tol {
            abs: 1e-12,
            rel: 0.0,
        },
    );
    for row in 0..2 {
        for column in 0..2 {
            let expected = if row == column { 1.0 } else { 0.0 };
            assert!((filter.covariance()[(row, column)] - expected).abs() < 1e-12);
        }
    }
    assert!((filter.innovation()[0] - 0.5).abs() < 1e-12);
}

// ----- Tuning rejection -----

#[test]
fn rejects_unusable_scaling() {
    // No spread at all.
    assert_eq!(
        landmark_filter().with_scaling(0.0, 2.0, 0.0).err(),
        Some(EstimationError::InvalidTuning)
    );
    // STATE_DIMENSION + kappa is exactly zero on a two-state filter.
    assert_eq!(
        landmark_filter().with_scaling(1e-3, 2.0, -2.0).err(),
        Some(EstimationError::InvalidTuning)
    );
    assert_eq!(
        landmark_filter().with_scaling(f64::NAN, 2.0, 0.0).err(),
        Some(EstimationError::InvalidTuning)
    );
    assert_eq!(
        landmark_filter()
            .with_scaling(1e-3, f64::INFINITY, 0.0)
            .err(),
        Some(EstimationError::InvalidTuning)
    );
}

#[test]
fn rejects_unusable_regularization() {
    assert_eq!(
        landmark_filter().with_regularization(-1e-9).err(),
        Some(EstimationError::InvalidTuning)
    );
    assert_eq!(
        landmark_filter().with_regularization(f64::NAN).err(),
        Some(EstimationError::InvalidTuning)
    );
    assert!(landmark_filter().with_regularization(0.0).is_ok());
}

/// The default path refuses a covariance it cannot factorize, and the fix has to be asked for.
#[test]
fn regularization_rescues_a_flat_covariance() {
    let flat = Matrix::new([[1.0, 0.0], [0.0, 0.0]]);
    let mut refused = UnscentedKalmanFilter::<2, 1>::new(
        Vector::new([0.0, 0.0]),
        flat,
        Matrix2D::identity().scale(0.01),
        Matrix::new([[0.1]]),
    );
    assert_eq!(
        refused.predict(&Stationary2D),
        Err(EstimationError::NotPositiveDefinite)
    );

    let mut rescued = UnscentedKalmanFilter::<2, 1>::new(
        Vector::new([0.0, 0.0]),
        flat,
        Matrix2D::identity().scale(0.01),
        Matrix::new([[0.1]]),
    )
    .with_regularization(1e-9)
    .unwrap();
    assert!(rescued.predict(&Stationary2D).is_ok());
}

// ----- Error paths -----

#[test]
fn rejects_non_finite_values() {
    let mut filter = landmark_filter();
    assert_eq!(filter.predict(&NotFinite), Err(EstimationError::NonFinite));

    let mut filter = landmark_filter();
    assert_eq!(
        filter.update(&RangeToLandmark, Vector::new([f64::NAN])),
        Err(EstimationError::NonFinite)
    );

    let mut filter = landmark_filter();
    assert_eq!(
        filter.update_with_residual(&RangeToLandmark, Vector::new([f64::INFINITY])),
        Err(EstimationError::NonFinite)
    );
}

#[test]
fn normalized_innovation_squared_before_any_update() {
    assert_eq!(
        landmark_filter().normalized_innovation_squared(),
        Err(EstimationError::NotPositiveDefinite)
    );
}

// ----- Angular measurements -----

#[test]
fn wrapped_residual_crosses_the_angle_boundary() {
    let heading_filter = || {
        UnscentedKalmanFilter::<1, 1>::new(
            Vector::new([3.1]),
            Matrix::new([[0.1]]),
            Matrix::new([[0.0]]),
            Matrix::new([[0.05]]),
        )
    };

    // Plain subtraction reads the error as most of a full turn the wrong way.
    let mut unwrapped = heading_filter();
    unwrapped.predict(&Compass).unwrap();
    unwrapped.update(&Compass, Vector::new([-3.1])).unwrap();
    assert!(unwrapped.state()[0] < 0.0);

    // Wrapped, the same reading is a small step past +π.
    let mut wrapped = heading_filter();
    wrapped.predict(&Compass).unwrap();
    let residual = Vector::new([wrap_to_pi(-3.1 - 3.1)]);
    wrapped.update_with_residual(&Compass, residual).unwrap();
    assert!(wrapped.state()[0] > 3.1);
}

// ----- A differentiating scalar -----

/// The filter is generic over its scalar the same way the other two are; the exact derivative is
/// not the point, only that it flows through and lands somewhere sensible.
#[test]
fn runs_under_a_differentiating_scalar() {
    let mut filter = UnscentedKalmanFilter::<2, 1, Dual<f64>>::new(
        Vector::new([Dual::constant(0.0), Dual::constant(0.0)]),
        Matrix2D::<Dual<f64>>::identity(),
        Matrix2D::<Dual<f64>>::identity().scale(Dual::constant(0.01)),
        Matrix::new([[Dual::constant(0.5)]]),
    );
    let measurement = Vector::new([Dual::new(1.0, 1.0)]);

    filter.predict(&ConstantVelocityMotion).unwrap();
    filter.update(&PositionMeasurement, measurement).unwrap();

    let sensitivity = filter.state()[0].deriv;
    assert!(sensitivity.is_finite());
    assert!(sensitivity > 0.0);
    assert!(sensitivity < 1.0);
}

// ----- Properties -----

proptest! {
    /// Whatever covariance it starts from, one cycle leaves a symmetric covariance with a
    /// positive trace.
    #[test]
    fn stays_symmetric_from_any_starting_covariance(
        entries in prop::array::uniform4(-2.0f64..2.0),
        measurement in -5.0f64..5.0,
    ) {
        let mut filter = UnscentedKalmanFilter::<2, 1>::new(
            Vector::new([0.0, 0.0]),
            Matrix::<2, 2>::symmetric_positive_definite(&entries),
            Matrix2D::identity().scale(0.01),
            Matrix::new([[0.5]]),
        );
        filter.predict(&ConstantVelocityMotion).unwrap();
        filter.update(&PositionMeasurement, Vector::new([measurement])).unwrap();
        let covariance = filter.covariance();
        prop_assert_eq!(covariance[(0, 1)], covariance[(1, 0)]);
        prop_assert!(covariance.trace() > 0.0);
    }
}
