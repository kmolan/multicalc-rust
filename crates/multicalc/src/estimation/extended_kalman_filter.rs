//! Extended Kalman filtering.
//!
//! Non-finite policy: every operation is checked. [`ExtendedKalmanFilter::predict`] returns
//! [`NonFinite`](EstimationError::NonFinite) when the propagated state or the transition Jacobian
//! holds an infinity or NaN, and [`update`](ExtendedKalmanFilter::update) does the same for the
//! measurement, the residual, and the innovation covariance. This is stricter than
//! [`KalmanFilter`](crate::estimation::KalmanFilter), whose `predict` is a matrix product that
//! propagates silently: here `predict` evaluates a caller-supplied model and differentiates it, so a
//! non-finite result is a likely outcome rather than an unlikely one, and the method returns a
//! `Result` regardless.
//!
//! Stack budget: the filter holds about `2·STATE_DIMENSION² + MEASUREMENT_DIMENSION² +
//! STATE_DIMENSION·MEASUREMENT_DIMENSION` scalars — one square fewer than the linear filter, which
//! stores a transition and a measurement matrix this one recomputes. Against that, each Jacobian
//! evaluates the model at `Dual<T>`, twice the width of `T`, and `update` forms several
//! `STATE_DIMENSION`-square temporaries. Scalar width dominates: `f64` 8 bytes, `Dual` 16,
//! `HyperDual` 32, `Jet<7>` 56. A 9-state `f64` filter is comfortable on a small microcontroller
//! stack; the same filter under `Jet<7>`, or any state beyond roughly 12 under `HyperDual`, is
//! host-only.

use crate::error::EstimationError;
use crate::estimation::CovarianceUpdate;
use crate::linear_algebra::{Matrix, Vector};
use crate::numerical_derivative::autodiff::AutoDiffMulti;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::numerical_derivative::jacobian::Jacobian;
use crate::scalar::{Numeric, VectorFn};

/// An extended Kalman filter over a `STATE_DIMENSION`-state model with `MEASUREMENT_DIMENSION`
/// measurements.
///
/// Where [`KalmanFilter`](crate::estimation::KalmanFilter) takes the dynamics and the sensor model as
/// matrices, this filter takes them as functions — any [`VectorFn`] — and re-linearizes them at the
/// current estimate on every step. **The Jacobians are taken by automatic differentiation: write the
/// model once, and its partial derivatives are exact. No Jacobian is ever derived or coded by hand.**
///
/// The models are passed to [`predict`](Self::predict) and [`update`](Self::update) rather than
/// stored, so the filter's type never names them, and anything that varies per step — the timestep, a
/// control input — lives in the model as a plain field the caller changes between calls. There is no
/// `predict_with_control`; a control input is part of the process model, which is more general than a
/// separate `B·u` term.
///
/// The covariance update is [`Joseph`](CovarianceUpdate::Joseph) by default: it stays symmetric and
/// positive definite by construction, where the naive form loses symmetry as rounding accumulates.
/// Joseph alone is not a guarantee at every scale — across roughly 10⁷ single-precision updates
/// (1 kHz for hours) it too drifts out of positive semi-definiteness, and symmetrize-and-clamp
/// conditioning is the answer there.
///
/// Cost: `predict` is `STATE_DIMENSION` model evaluations (one seeded [`Dual`](crate::scalar::Dual)
/// pass per Jacobian column, each reading every output) plus two `STATE_DIMENSION`-cubed matrix
/// products. `update` adds one more model evaluation for the prediction, `STATE_DIMENSION` for its
/// Jacobian, one `MEASUREMENT_DIMENSION`-square Cholesky factorization, and
/// O(`STATE_DIMENSION`²·`MEASUREMENT_DIMENSION`), with Joseph adding two `STATE_DIMENSION`-cubed
/// products over Naive. The Jacobian passes run at `Dual<T>`, so they cost twice the scalar width.
///
/// # Examples
/// ```
/// use multicalc::estimation::ExtendedKalmanFilter;
/// use multicalc::linear_algebra::{Matrix, Vector};
/// use multicalc::scalar::{Numeric, VectorFn};
/// # fn main() -> Result<(), multicalc::error::EstimationError> {
/// // Range to a landmark at (3, 4): nonlinear in the state, so the linear filter cannot take it.
/// struct RangeToLandmark;
/// impl VectorFn<2, 1> for RangeToLandmark {
///     fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
///         let to_landmark_x = S::from_f64(3.0) - state[0];
///         let to_landmark_y = S::from_f64(4.0) - state[1];
///         [(to_landmark_x * to_landmark_x + to_landmark_y * to_landmark_y).sqrt()]
///     }
/// }
///
/// // A stationary target: the state carries over unchanged.
/// struct Stationary;
/// impl VectorFn<2, 2> for Stationary {
///     fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
///         [state[0], state[1]]
///     }
/// }
///
/// let mut filter = ExtendedKalmanFilter::<2, 1>::new(
///     Vector::new([0.0, 0.0]),                  // initial state, 5.0 from the landmark
///     Matrix::new([[1.0, 0.0], [0.0, 1.0]]),    // initial covariance
///     Matrix::new([[0.01, 0.0], [0.0, 0.01]]),  // process noise
///     Matrix::new([[0.1]]),                     // measurement noise
/// );
/// filter.predict(&Stationary)?;
/// filter.update(&RangeToLandmark, Vector::new([5.5]))?;
/// // A longer range than predicted moves the estimate away from the landmark.
/// assert!(filter.state()[0] < 0.0);
/// assert!(filter.state()[1] < 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtendedKalmanFilter<
    const STATE_DIMENSION: usize,
    const MEASUREMENT_DIMENSION: usize,
    T = f64,
    D = AutoDiffMulti<T>,
> {
    state: Vector<STATE_DIMENSION, T>,
    covariance: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    innovation: Vector<MEASUREMENT_DIMENSION, T>,
    innovation_covariance: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    covariance_update: CovarianceUpdate,
    derivator: D,
}

impl<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize, T: Numeric>
    ExtendedKalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T, AutoDiffMulti<T>>
{
    /// Builds a filter that takes its model Jacobians by automatic differentiation.
    ///
    /// The covariance update starts at [`Joseph`](CovarianceUpdate::Joseph); change it with
    /// [`with_covariance_update`](Self::with_covariance_update).
    pub fn new(
        initial_state: Vector<STATE_DIMENSION, T>,
        initial_covariance: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    ) -> Self {
        Self::from_derivator(
            initial_state,
            initial_covariance,
            process_noise,
            measurement_noise,
            AutoDiffMulti::new(),
        )
    }
}

impl<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize, T: Numeric, D>
    ExtendedKalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T, D>
where
    D: DerivatorMultiVariable<Scalar = T> + Clone,
{
    /// Builds a filter with an explicit differentiation backend — a
    /// [`FiniteDifferenceMulti`](crate::numerical_derivative::finite_difference::FiniteDifferenceMulti)
    /// or your own [`DerivatorMultiVariable`]. [`new`](Self::new) is the autodiff default.
    pub fn from_derivator(
        initial_state: Vector<STATE_DIMENSION, T>,
        initial_covariance: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
        derivator: D,
    ) -> Self {
        const {
            assert!(
                STATE_DIMENSION > 0,
                "ExtendedKalmanFilter: STATE_DIMENSION must be non-zero"
            )
        };
        const {
            assert!(
                MEASUREMENT_DIMENSION > 0,
                "ExtendedKalmanFilter: MEASUREMENT_DIMENSION must be non-zero"
            )
        };
        ExtendedKalmanFilter {
            state: initial_state,
            covariance: initial_covariance,
            process_noise,
            measurement_noise,
            innovation: Vector::zeros(),
            innovation_covariance: Matrix::zeros(),
            covariance_update: CovarianceUpdate::Joseph,
            derivator,
        }
    }

    /// Selects how [`update`](Self::update) recomputes the covariance.
    ///
    /// ```
    /// use multicalc::estimation::{CovarianceUpdate, ExtendedKalmanFilter};
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::scalar::{Numeric, VectorFn};
    /// # fn main() -> Result<(), multicalc::error::EstimationError> {
    /// struct Stationary;
    /// impl VectorFn<2, 2> for Stationary {
    ///     fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
    ///         [state[0], state[1]]
    ///     }
    /// }
    /// struct MeasurePosition;
    /// impl VectorFn<2, 1> for MeasurePosition {
    ///     fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
    ///         [state[0]]
    ///     }
    /// }
    /// let mut filter = ExtendedKalmanFilter::<2, 1>::new(
    ///     Vector::new([0.0, 0.0]),
    ///     Matrix::new([[1.0, 0.0], [0.0, 1.0]]),
    ///     Matrix::new([[0.01, 0.0], [0.0, 0.01]]),
    ///     Matrix::new([[0.1]]),
    /// )
    /// .with_covariance_update(CovarianceUpdate::Naive);
    /// filter.predict(&Stationary)?;
    /// filter.update(&MeasurePosition, Vector::new([1.0]))?;
    /// assert!(filter.covariance()[(0, 0)] > 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub const fn with_covariance_update(mut self, covariance_update: CovarianceUpdate) -> Self {
        self.covariance_update = covariance_update;
        self
    }

    /// Replaces the state estimate. Also the hook for re-wrapping an angular state component after
    /// an update — see [`update_with_residual`](Self::update_with_residual).
    pub fn set_state(&mut self, state: Vector<STATE_DIMENSION, T>) {
        self.state = state;
    }

    /// Replaces the process noise, which a changing timestep also changes.
    pub fn set_process_noise(
        &mut self,
        process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    ) {
        self.process_noise = process_noise;
    }

    /// Replaces the measurement noise.
    pub fn set_measurement_noise(
        &mut self,
        measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    ) {
        self.measurement_noise = measurement_noise;
    }

    /// Rolls the state and covariance forward one step through `process_model`.
    ///
    /// The model maps the current state to the next; its Jacobian, taken at the current estimate,
    /// propagates the covariance. The timestep and any control input belong to the model — carry them
    /// as fields and change them between steps.
    ///
    /// Returns [`Diff`](EstimationError::Diff) if the Jacobian cannot be taken, and
    /// [`NonFinite`](EstimationError::NonFinite) if the propagated state or the Jacobian holds an
    /// infinity or NaN.
    pub fn predict<ProcessModel>(
        &mut self,
        process_model: &ProcessModel,
    ) -> Result<(), EstimationError>
    where
        ProcessModel: VectorFn<STATE_DIMENSION, STATE_DIMENSION>,
    {
        let state_transition = Jacobian::from_derivator(self.derivator.clone())
            .get(process_model, self.state.as_array())?;

        let propagated = Vector::new(process_model.eval(self.state.as_array()));
        if !propagated.is_finite() || !state_transition.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        self.state = propagated;
        self.covariance =
            state_transition * self.covariance * state_transition.transpose() + self.process_noise;
        Ok(())
    }

    /// Folds `measurement` into the estimate, forming the residual as `measurement − h(state)`.
    ///
    /// Use [`update_with_residual`](Self::update_with_residual) when any measurement component is an
    /// angle: plain subtraction is wrong across the ±π wrap.
    ///
    /// Returns [`NonFinite`](EstimationError::NonFinite) when the measurement, the residual, or the
    /// formed innovation covariance holds an infinity or NaN, [`Diff`](EstimationError::Diff) if the
    /// Jacobian cannot be taken, and
    /// [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) when the innovation covariance
    /// cannot be factorized — the gain is undefined.
    pub fn update<MeasurementModel>(
        &mut self,
        measurement_model: &MeasurementModel,
        measurement: Vector<MEASUREMENT_DIMENSION, T>,
    ) -> Result<(), EstimationError>
    where
        MeasurementModel: VectorFn<STATE_DIMENSION, MEASUREMENT_DIMENSION>,
    {
        if !measurement.is_finite() {
            return Err(EstimationError::NonFinite);
        }
        let predicted = Vector::new(measurement_model.eval(self.state.as_array()));
        self.update_with_residual(measurement_model, measurement - predicted)
    }

    /// [`update`](Self::update) with a caller-formed residual, for measurements that plain
    /// subtraction cannot difference correctly.
    ///
    /// A bearing residual must be wrapped to (−π, π] before it reaches the filter: unwrapped, an
    /// error near ±π reads as most of a full turn, and the gain drives the estimate hard the wrong
    /// way — silently, since nothing about the arithmetic is invalid. The filter cannot do this
    /// itself; which components of a `MEASUREMENT_DIMENSION`-vector are angular is not something the
    /// type records. Re-wrapping an angular *state* component after the update is likewise the
    /// caller's, through [`set_state`](Self::set_state).
    ///
    /// ```
    /// use multicalc::estimation::ExtendedKalmanFilter;
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::scalar::{Numeric, VectorFn};
    /// # fn main() -> Result<(), multicalc::error::EstimationError> {
    /// // Heading, measured by a compass: the state is an angle, so the residual is too.
    /// struct Compass;
    /// impl VectorFn<1, 1> for Compass {
    ///     fn eval<S: Numeric>(&self, state: &[S; 1]) -> [S; 1] {
    ///         [state[0]]
    ///     }
    /// }
    ///
    /// // atan2 returns (−π, π], so this is the wrap.
    /// fn wrap_to_pi<T: Numeric>(angle: T) -> T {
    ///     angle.sin().atan2(angle.cos())
    /// }
    ///
    /// let mut filter = ExtendedKalmanFilter::<1, 1>::new(
    ///     Vector::new([3.1]),               // heading just under +π
    ///     Matrix::new([[0.1]]),
    ///     Matrix::new([[0.001]]),
    ///     Matrix::new([[0.05]]),
    /// );
    ///
    /// // The compass reads just over −π: a true error of about 0.08 rad, not −6.2.
    /// let measurement = Vector::new([-3.1]);
    /// let predicted = Vector::new(Compass.eval(filter.state().as_array()));
    /// let residual = Vector::new([wrap_to_pi(measurement[0] - predicted[0])]);
    /// filter.update_with_residual(&Compass, residual)?;
    ///
    /// // The estimate steps a little past +π, rather than most of the way around the circle.
    /// assert!(filter.state()[0] > 3.1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_with_residual<MeasurementModel>(
        &mut self,
        measurement_model: &MeasurementModel,
        residual: Vector<MEASUREMENT_DIMENSION, T>,
    ) -> Result<(), EstimationError>
    where
        MeasurementModel: VectorFn<STATE_DIMENSION, MEASUREMENT_DIMENSION>,
    {
        if !residual.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        let measurement_model_jacobian = Jacobian::from_derivator(self.derivator.clone())
            .get(measurement_model, self.state.as_array())?;

        self.innovation = residual;
        self.innovation_covariance =
            measurement_model_jacobian * self.covariance * measurement_model_jacobian.transpose()
                + self.measurement_noise;

        if !self.innovation_covariance.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        // Kᵀ = S⁻¹·H·Pᵀ, solved rather than inverted. Pᵀ is written out so an asymmetric
        // caller-seeded covariance still gives the exact gain.
        let projected = measurement_model_jacobian * self.covariance.transpose();
        let kalman_gain = self
            .innovation_covariance
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?
            .solve_matrix::<STATE_DIMENSION>(projected)
            .transpose();

        self.state += kalman_gain * self.innovation;

        let residual_transfer = Matrix::<STATE_DIMENSION, STATE_DIMENSION, T>::identity()
            - kalman_gain * measurement_model_jacobian;
        self.covariance = match self.covariance_update {
            CovarianceUpdate::Joseph => {
                residual_transfer * self.covariance * residual_transfer.transpose()
                    + kalman_gain * self.measurement_noise * kalman_gain.transpose()
            }
            CovarianceUpdate::Naive => residual_transfer * self.covariance,
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

    /// The innovation from the last [`update`](Self::update). Zero before the first one.
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
