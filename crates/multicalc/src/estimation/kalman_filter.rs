//! Linear Kalman filtering.
//!
//! Non-finite policy: [`KalmanFilter::update`] returns
//! [`NonFinite`](EstimationError::NonFinite) when the measurement or the formed innovation
//! covariance holds an infinity or NaN. [`KalmanFilter::predict`] is a cheap element-wise path and
//! propagates non-finite values silently.
//!
//! Stack budget: the filter holds about `3·STATE_DIMENSION² + MEASUREMENT_DIMENSION² +
//! STATE_DIMENSION·MEASUREMENT_DIMENSION` scalars, and `update` forms several
//! `STATE_DIMENSION`-square temporaries. Scalar width dominates: `f64` 8 bytes, `Dual` 16,
//! `HyperDual` 32, `Jet<7>` 56. A 9-state `f64` filter is comfortable on a small microcontroller
//! stack; the same filter under `Jet<7>`, or any state beyond roughly 12 under `HyperDual`, is
//! host-only.

use crate::error::EstimationError;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::Numeric;

/// How [`KalmanFilter::update`] recomputes the covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum CovarianceUpdate {
    /// `(I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ`. Symmetric and positive definite by construction, at the
    /// cost of two extra products. The default.
    #[default]
    Joseph,
    /// `(I − K·H)·P`. Cheaper, and loses symmetry as rounding accumulates.
    Naive,
}

/// A linear Kalman filter over a `STATE_DIMENSION`-state model with `MEASUREMENT_DIMENSION`
/// measurements.
///
/// Each step does two things. [`predict`](Self::predict) rolls the state forward through the
/// transition model and grows the covariance by the process noise.
/// [`update`](Self::update) folds in a measurement, shrinking the covariance by what the
/// measurement resolved.
///
/// The covariance update is [`Joseph`](CovarianceUpdate::Joseph) by default: it stays symmetric and
/// positive definite by construction, where the naive form loses symmetry as rounding accumulates.
/// Joseph alone is not a guarantee at every scale — across roughly 10⁷ single-precision updates
/// (1 kHz for hours) it too drifts out of positive semi-definiteness, and symmetrize-and-clamp
/// conditioning is the answer there.
///
/// Cost: `predict` is two `STATE_DIMENSION`-cubed matrix products. `update` is one
/// `MEASUREMENT_DIMENSION`-square Cholesky factorization plus
/// O(`STATE_DIMENSION`²·`MEASUREMENT_DIMENSION`), with Joseph adding two `STATE_DIMENSION`-cubed
/// products over Naive.
///
/// # Examples
/// ```
/// use multicalc::estimation::KalmanFilter;
/// use multicalc::linear_algebra::{Matrix, Vector};
/// # fn main() -> Result<(), multicalc::error::EstimationError> {
/// // Constant velocity: position integrates velocity over a 1 s step; position is measured.
/// let mut filter = KalmanFilter::new(
///     Vector::new([0.0, 0.0]),                        // initial state
///     Matrix::new([[1.0, 0.0], [0.0, 1.0]]),          // initial covariance
///     Matrix::new([[1.0, 1.0], [0.0, 1.0]]),          // state transition
///     Matrix::new([[1.0, 0.0]]),                      // measurement model
///     Matrix::new([[0.01, 0.0], [0.0, 0.01]]),        // process noise
///     Matrix::new([[0.1]]),                           // measurement noise
/// );
/// filter.predict();
/// filter.update(Vector::new([1.0]))?;
/// assert!(filter.state()[0] > 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanFilter<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize, T = f64> {
    state: Vector<STATE_DIMENSION, T>,
    covariance: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    state_transition: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    measurement_model: Matrix<MEASUREMENT_DIMENSION, STATE_DIMENSION, T>,
    process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    innovation: Vector<MEASUREMENT_DIMENSION, T>,
    innovation_covariance: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    covariance_update: CovarianceUpdate,
}

impl<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize, T: Numeric>
    KalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>
{
    /// Builds a filter from an initial estimate and the four model matrices.
    ///
    /// The covariance update starts at [`Joseph`](CovarianceUpdate::Joseph); change it with
    /// [`with_covariance_update`](Self::with_covariance_update).
    pub fn new(
        initial_state: Vector<STATE_DIMENSION, T>,
        initial_covariance: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        state_transition: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        measurement_model: Matrix<MEASUREMENT_DIMENSION, STATE_DIMENSION, T>,
        process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    ) -> Self {
        const {
            assert!(
                STATE_DIMENSION > 0,
                "KalmanFilter: STATE_DIMENSION must be non-zero"
            )
        };
        const {
            assert!(
                MEASUREMENT_DIMENSION > 0,
                "KalmanFilter: MEASUREMENT_DIMENSION must be non-zero"
            )
        };
        KalmanFilter {
            state: initial_state,
            covariance: initial_covariance,
            state_transition,
            measurement_model,
            process_noise,
            measurement_noise,
            innovation: Vector::zeros(),
            innovation_covariance: Matrix::zeros(),
            covariance_update: CovarianceUpdate::Joseph,
        }
    }

    /// Selects how [`update`](Self::update) recomputes the covariance.
    ///
    /// ```
    /// use multicalc::estimation::{CovarianceUpdate, KalmanFilter};
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// # fn main() -> Result<(), multicalc::error::EstimationError> {
    /// let mut filter = KalmanFilter::new(
    ///     Vector::new([0.0, 0.0]),
    ///     Matrix::new([[1.0, 0.0], [0.0, 1.0]]),
    ///     Matrix::new([[1.0, 1.0], [0.0, 1.0]]),
    ///     Matrix::new([[1.0, 0.0]]),
    ///     Matrix::new([[0.01, 0.0], [0.0, 0.01]]),
    ///     Matrix::new([[0.1]]),
    /// )
    /// .with_covariance_update(CovarianceUpdate::Naive);
    /// filter.predict();
    /// filter.update(Vector::new([1.0]))?;
    /// assert!(filter.covariance()[(0, 0)] > 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn with_covariance_update(mut self, covariance_update: CovarianceUpdate) -> Self {
        self.covariance_update = covariance_update;
        self
    }

    /// Replaces the state estimate.
    pub fn set_state(&mut self, state: Vector<STATE_DIMENSION, T>) {
        self.state = state;
    }

    /// Replaces the transition model, for a system whose dynamics or timestep change between steps.
    pub fn set_state_transition(
        &mut self,
        state_transition: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    ) {
        self.state_transition = state_transition;
    }

    /// Replaces the process noise, which a changing timestep also changes.
    pub fn set_process_noise(
        &mut self,
        process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    ) {
        self.process_noise = process_noise;
    }

    /// Replaces the measurement model.
    pub fn set_measurement_model(
        &mut self,
        measurement_model: Matrix<MEASUREMENT_DIMENSION, STATE_DIMENSION, T>,
    ) {
        self.measurement_model = measurement_model;
    }

    /// Replaces the measurement noise.
    pub fn set_measurement_noise(
        &mut self,
        measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    ) {
        self.measurement_noise = measurement_noise;
    }

    // ----- Predict -----

    /// Rolls the state and covariance forward one step through the transition model.
    pub fn predict(&mut self) {
        self.state = self.state_transition * self.state;
        self.predict_covariance();
    }

    /// [`predict`](Self::predict) for a driven system, adding `control_model · control_input` to the
    /// state. The covariance is unaffected by the control term.
    pub fn predict_with_control<const CONTROL_DIMENSION: usize>(
        &mut self,
        control_model: Matrix<STATE_DIMENSION, CONTROL_DIMENSION, T>,
        control_input: Vector<CONTROL_DIMENSION, T>,
    ) {
        self.state = self.state_transition * self.state + control_model * control_input;
        self.predict_covariance();
    }

    fn predict_covariance(&mut self) {
        self.covariance =
            self.state_transition * self.covariance * self.state_transition.transpose()
                + self.process_noise;
    }

    // ----- Update -----

    /// Folds `measurement` into the estimate.
    ///
    /// Returns [`NonFinite`](EstimationError::NonFinite) when the measurement or the formed
    /// innovation covariance holds an infinity or NaN, and
    /// [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) when the innovation covariance
    /// cannot be factorized — the gain is undefined.
    pub fn update(
        &mut self,
        measurement: Vector<MEASUREMENT_DIMENSION, T>,
    ) -> Result<(), EstimationError> {
        if !measurement.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        self.innovation = measurement - self.measurement_model * self.state;
        self.innovation_covariance =
            self.measurement_model * self.covariance * self.measurement_model.transpose()
                + self.measurement_noise;

        if !self.innovation_covariance.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        // Kᵀ = S⁻¹·H·Pᵀ, solved rather than inverted. Pᵀ is written out so an asymmetric
        // caller-seeded covariance still gives the exact gain.
        let projected = self.measurement_model * self.covariance.transpose();
        let kalman_gain = self
            .innovation_covariance
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?
            .solve_matrix::<STATE_DIMENSION>(projected)
            .transpose();

        self.state += kalman_gain * self.innovation;

        let residual = Matrix::<STATE_DIMENSION, STATE_DIMENSION, T>::identity()
            - kalman_gain * self.measurement_model;
        self.covariance = match self.covariance_update {
            CovarianceUpdate::Joseph => {
                residual * self.covariance * residual.transpose()
                    + kalman_gain * self.measurement_noise * kalman_gain.transpose()
            }
            CovarianceUpdate::Naive => residual * self.covariance,
        };

        Ok(())
    }

    // ----- Accessors -----

    /// The current state estimate.
    pub fn state(&self) -> Vector<STATE_DIMENSION, T> {
        self.state
    }

    /// The current state covariance.
    pub fn covariance(&self) -> Matrix<STATE_DIMENSION, STATE_DIMENSION, T> {
        self.covariance
    }

    /// The innovation `z − H·x` from the last [`update`](Self::update). Zero before the first one.
    pub fn innovation(&self) -> Vector<MEASUREMENT_DIMENSION, T> {
        self.innovation
    }

    /// The innovation covariance `S` from the last [`update`](Self::update). Zero before the first.
    pub fn innovation_covariance(&self) -> Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T> {
        self.innovation_covariance
    }

    /// `yᵀ·S⁻¹·y` for the last update — the innovation weighted by its own covariance.
    ///
    /// Returns [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) if the innovation
    /// covariance cannot be factorized, including before the first update, when it is zero.
    pub fn normalized_innovation_squared(&self) -> Result<T, EstimationError> {
        let weighted = self
            .innovation_covariance
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?
            .solve(self.innovation);
        Ok(self.innovation.dot(weighted))
    }
}
