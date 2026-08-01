//! Unscented Kalman filtering.

use crate::error::EstimationError;
use crate::linear_algebra::{Matrix, Vector};
use crate::scalar::{Numeric, VectorFn};

/// An unscented Kalman filter over a `STATE_DIMENSION`-state model with `MEASUREMENT_DIMENSION`
/// measurements.
///
/// Where [`ExtendedKalmanFilter`](crate::estimation::ExtendedKalmanFilter) flattens the model to a
/// straight line at the current estimate, this filter picks `2·STATE_DIMENSION + 1` points spread
/// around it, pushes each one through the model untouched, and rebuilds the estimate from where
/// they land. **The model is never differentiated, so it does not have to be smooth — a lookup
/// table, a saturating actuator, or a branch on a threshold is as acceptable as a formula.** On a
/// strongly curved model the answer is usually closer than a single straight-line fit gets.
///
/// The models are passed to [`predict`](Self::predict) and [`update`](Self::update) rather than
/// stored, so the filter's type never names them, and anything that varies per step — the timestep,
/// a control input — lives in the model as a plain field the caller changes between calls.
///
/// Three numbers set how far the points spread and how the middle one is weighted:
/// `alpha` = 1e-3, `beta` = 2, `kappa` = 0 by default, changed together through
/// [`with_scaling`](Self::with_scaling). The covariance is made exactly symmetric every time it is
/// formed, which is what keeps a long run from drifting the way an unsymmetrized sum does.
///
/// Cost: `predict` is `2·STATE_DIMENSION + 1` model evaluations, one `STATE_DIMENSION`-square
/// factorization, and `2·STATE_DIMENSION + 1` rank-one accumulations into a square matrix.
/// `update` is another `2·STATE_DIMENSION + 1` model evaluations, one
/// `MEASUREMENT_DIMENSION`-square factorization, and O(`STATE_DIMENSION`²·`MEASUREMENT_DIMENSION`).
/// That is more model evaluations than the extended filter needs but no derivatives, so which is
/// cheaper depends entirely on how expensive the model is to evaluate.
///
/// # Examples
/// ```
/// use multicalc::estimation::UnscentedKalmanFilter;
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
/// let mut filter = UnscentedKalmanFilter::<2, 1>::new(
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
pub struct UnscentedKalmanFilter<
    const STATE_DIMENSION: usize,
    const MEASUREMENT_DIMENSION: usize,
    T = f64,
> {
    state: Vector<STATE_DIMENSION, T>,
    covariance: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    innovation: Vector<MEASUREMENT_DIMENSION, T>,
    innovation_covariance: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    propagated_centre: Vector<STATE_DIMENSION, T>,
    propagated_plus: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    propagated_minus: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
    alpha: T,
    beta: T,
    kappa: T,
    regularization: T,
}

impl<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize, T: Numeric>
    UnscentedKalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>
{
    /// Builds a filter from an initial estimate and the noise it is working against.
    ///
    /// The spread starts at `alpha` = 1e-3, `beta` = 2, `kappa` = 0 and no regularization; change
    /// them with [`with_scaling`](Self::with_scaling) and
    /// [`with_regularization`](Self::with_regularization).
    #[must_use]
    pub fn new(
        initial_state: Vector<STATE_DIMENSION, T>,
        initial_covariance: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        process_noise: Matrix<STATE_DIMENSION, STATE_DIMENSION, T>,
        measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    ) -> Self {
        const {
            assert!(
                STATE_DIMENSION > 0,
                "UnscentedKalmanFilter: STATE_DIMENSION must be non-zero"
            )
        };
        const {
            assert!(
                MEASUREMENT_DIMENSION > 0,
                "UnscentedKalmanFilter: MEASUREMENT_DIMENSION must be non-zero"
            )
        };
        let seeded = Matrix::from_fn(|row, _| initial_state[row]);
        UnscentedKalmanFilter {
            state: initial_state,
            covariance: initial_covariance,
            process_noise,
            measurement_noise,
            innovation: Vector::zeros(),
            innovation_covariance: Matrix::zeros(),
            propagated_centre: initial_state,
            propagated_plus: seeded,
            propagated_minus: seeded,
            alpha: T::from_f64(1e-3),
            beta: T::TWO,
            kappa: T::ZERO,
            regularization: T::ZERO,
        }
    }

    /// Sets how far the points spread around the estimate and how the middle one is weighted.
    ///
    /// `alpha` sets the spread: small keeps the points tight around the estimate, large pushes them
    /// out to sample more of the model's curve. `beta` folds in what is known about the shape of
    /// the uncertainty — 2 is right when it is a bell curve. `kappa` is a second, rarely-changed
    /// spread term, normally left at zero.
    ///
    /// Returns [`InvalidTuning`](EstimationError::InvalidTuning) if any of the three is infinite or
    /// NaN, or if `alpha² · (STATE_DIMENSION + kappa)` is not above zero — below that the points
    /// have no real spread to sit at.
    ///
    /// ```
    /// use multicalc::error::EstimationError;
    /// use multicalc::estimation::UnscentedKalmanFilter;
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// # fn main() -> Result<(), EstimationError> {
    /// let filter = UnscentedKalmanFilter::<2, 1>::new(
    ///     Vector::new([0.0, 0.0]),
    ///     Matrix::new([[1.0, 0.0], [0.0, 1.0]]),
    ///     Matrix::new([[0.01, 0.0], [0.0, 0.01]]),
    ///     Matrix::new([[0.1]]),
    /// )
    /// .with_scaling(0.3, 2.0, 0.0)?;
    ///
    /// // A spread of zero leaves the points nowhere to sit.
    /// let rejected = UnscentedKalmanFilter::<2, 1>::new(
    ///     Vector::new([0.0, 0.0]),
    ///     Matrix::new([[1.0, 0.0], [0.0, 1.0]]),
    ///     Matrix::new([[0.01, 0.0], [0.0, 0.01]]),
    ///     Matrix::new([[0.1]]),
    /// )
    /// .with_scaling(0.0, 2.0, 0.0);
    /// assert_eq!(rejected.unwrap_err(), EstimationError::InvalidTuning);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_scaling(mut self, alpha: T, beta: T, kappa: T) -> Result<Self, EstimationError> {
        if !alpha.is_finite() || !beta.is_finite() || !kappa.is_finite() {
            return Err(EstimationError::InvalidTuning);
        }
        let spread = alpha * alpha * (T::from_f64(STATE_DIMENSION as f64) + kappa);
        if !spread.is_finite() || spread <= T::ZERO {
            return Err(EstimationError::InvalidTuning);
        }
        self.alpha = alpha;
        self.beta = beta;
        self.kappa = kappa;
        Ok(self)
    }

    /// Adds a small amount to the diagonal of the covariance before it is factorized, so a
    /// borderline one still factorizes.
    ///
    /// Off by default, and never applied on its own: a covariance that cannot be factorized returns
    /// [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) from
    /// [`predict`](Self::predict) unless this was set deliberately.
    ///
    /// Returns [`InvalidTuning`](EstimationError::InvalidTuning) if the value is negative,
    /// infinite, or NaN.
    pub fn with_regularization(mut self, regularization: T) -> Result<Self, EstimationError> {
        if !regularization.is_finite() || regularization < T::ZERO {
            return Err(EstimationError::InvalidTuning);
        }
        self.regularization = regularization;
        Ok(self)
    }

    /// Replaces the state estimate. Also the hook for re-wrapping an angular state component after
    /// an update — see [`update_with_residual`](Self::update_with_residual). The points
    /// [`predict`](Self::predict) left behind are not touched, so re-wrapping between an update and
    /// the next predict is safe.
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

    /// How far the points sit from the estimate, and the factor the covariance is scaled by before
    /// it is factorized.
    #[must_use]
    fn spread_factor(&self) -> T {
        self.alpha * self.alpha * (T::from_f64(STATE_DIMENSION as f64) + self.kappa)
    }

    /// The weight on the middle point when averaging, the weight on it when rebuilding the spread,
    /// and the weight shared by all the outer points.
    #[must_use]
    fn weights(&self) -> (T, T, T) {
        let spread = self.spread_factor();
        let centre_mean = (spread - T::from_f64(STATE_DIMENSION as f64)) / spread;
        let centre_covariance = centre_mean + (T::ONE - self.alpha * self.alpha + self.beta);
        let side = T::ONE / (T::TWO * spread);
        (centre_mean, centre_covariance, side)
    }

    /// The directions the points are offset along: the columns of the factor of the scaled
    /// covariance. Point `j` on the plus side is the estimate plus column `j`, and on the minus
    /// side the estimate minus it.
    fn offset_directions(
        &self,
    ) -> Result<Matrix<STATE_DIMENSION, STATE_DIMENSION, T>, EstimationError> {
        let scaled = self.covariance.scale(self.spread_factor())
            + Matrix::<STATE_DIMENSION, STATE_DIMENSION, T>::identity().scale(self.regularization);
        if !scaled.is_finite() {
            return Err(EstimationError::NonFinite);
        }
        Ok(scaled
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?
            .l())
    }

    /// Rolls the state and covariance forward one step through `process_model`.
    ///
    /// Picks `2·STATE_DIMENSION + 1` points around the current estimate, pushes each through the
    /// model, and rebuilds the estimate and its spread from where they land. The points are kept
    /// for the next [`update`](Self::update). The timestep and any control input belong to the
    /// model — carry them as fields and change them between steps.
    ///
    /// Returns [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) when the covariance
    /// cannot be factorized to place the points — see
    /// [`with_regularization`](Self::with_regularization) — and
    /// [`NonFinite`](EstimationError::NonFinite) when a propagated point or the formed covariance
    /// holds an infinity or NaN.
    pub fn predict<ProcessModel>(
        &mut self,
        process_model: &ProcessModel,
    ) -> Result<(), EstimationError>
    where
        ProcessModel: VectorFn<STATE_DIMENSION, STATE_DIMENSION>,
    {
        // One direction per state component, and how much each point counts for.
        let directions = self.offset_directions()?;
        let (centre_mean, centre_covariance, side) = self.weights();

        // Push every point through the model: the estimate itself, then a pair either side of it
        // along each direction. Where they land is kept for the next update.
        self.propagated_centre = Vector::new(process_model.eval(self.state.as_array()));
        let mut propagated_plus = Matrix::<STATE_DIMENSION, STATE_DIMENSION, T>::zeros();
        let mut propagated_minus = Matrix::<STATE_DIMENSION, STATE_DIMENSION, T>::zeros();
        for column in 0..STATE_DIMENSION {
            let direction = Vector::from_fn(|row| directions[(row, column)]);
            let plus = self.state + direction;
            let minus = self.state - direction;
            let plus = Vector::new(process_model.eval(plus.as_array()));
            let minus = Vector::new(process_model.eval(minus.as_array()));
            for row in 0..STATE_DIMENSION {
                propagated_plus[(row, column)] = plus[row];
                propagated_minus[(row, column)] = minus[row];
            }
        }
        self.propagated_plus = propagated_plus;
        self.propagated_minus = propagated_minus;

        // The model is the caller's, so it may have handed back an infinity or a NaN.
        if !self.propagated_centre.is_finite()
            || !self.propagated_plus.is_finite()
            || !self.propagated_minus.is_finite()
        {
            return Err(EstimationError::NonFinite);
        }

        // The new estimate is the weighted average of where the points landed.
        let mut mean = self.propagated_centre.scale(centre_mean);
        for column in 0..STATE_DIMENSION {
            let plus = Vector::from_fn(|row| self.propagated_plus[(row, column)]);
            let minus = Vector::from_fn(|row| self.propagated_minus[(row, column)]);
            mean += (plus + minus).scale(side);
        }

        // The new spread is how far the points sit from that average, weighted the same way. The
        // process noise starts the sum off rather than being added at the end.
        let mut covariance = add_weighted_outer_product(
            self.process_noise,
            centre_covariance,
            self.propagated_centre - mean,
            self.propagated_centre - mean,
        );
        for column in 0..STATE_DIMENSION {
            let plus = Vector::from_fn(|row| self.propagated_plus[(row, column)]) - mean;
            let minus = Vector::from_fn(|row| self.propagated_minus[(row, column)]) - mean;
            covariance = add_weighted_outer_product(covariance, side, plus, plus);
            covariance = add_weighted_outer_product(covariance, side, minus, minus);
        }

        // Adding many terms can overflow even when every point was fine on its own.
        if !mean.is_finite() || !covariance.is_finite() {
            return Err(EstimationError::NonFinite);
        }
        self.state = mean;

        // Rounding leaves the two halves of the sum slightly apart; average them back together.
        self.covariance = symmetrized(covariance);
        Ok(())
    }

    /// Pushes the points [`predict`](Self::predict) left behind through the measurement model, and
    /// returns where they landed together with their average.
    fn measurement_sigma_points<MeasurementModel>(
        &self,
        measurement_model: &MeasurementModel,
    ) -> Result<
        (
            Vector<MEASUREMENT_DIMENSION, T>,
            Matrix<MEASUREMENT_DIMENSION, STATE_DIMENSION, T>,
            Matrix<MEASUREMENT_DIMENSION, STATE_DIMENSION, T>,
            Vector<MEASUREMENT_DIMENSION, T>,
        ),
        EstimationError,
    >
    where
        MeasurementModel: VectorFn<STATE_DIMENSION, MEASUREMENT_DIMENSION>,
    {
        let (centre_mean, _, side) = self.weights();

        // What each point would read on the sensor. No new points are drawn; these are the ones
        // predict left behind.
        let centre = Vector::new(measurement_model.eval(self.propagated_centre.as_array()));
        let mut plus_side = Matrix::<MEASUREMENT_DIMENSION, STATE_DIMENSION, T>::zeros();
        let mut minus_side = Matrix::<MEASUREMENT_DIMENSION, STATE_DIMENSION, T>::zeros();
        for column in 0..STATE_DIMENSION {
            let plus_state = Vector::from_fn(|row| self.propagated_plus[(row, column)]);
            let minus_state = Vector::from_fn(|row| self.propagated_minus[(row, column)]);
            let plus = measurement_model.eval(plus_state.as_array());
            let minus = measurement_model.eval(minus_state.as_array());
            for row in 0..MEASUREMENT_DIMENSION {
                plus_side[(row, column)] = plus[row];
                minus_side[(row, column)] = minus[row];
            }
        }

        // The reading the filter expects is the weighted average of those.
        let mut mean = centre.scale(centre_mean);
        for column in 0..STATE_DIMENSION {
            let plus = Vector::from_fn(|row| plus_side[(row, column)]);
            let minus = Vector::from_fn(|row| minus_side[(row, column)]);
            mean += (plus + minus).scale(side);
        }

        // The model is the caller's, so it may have handed back an infinity or a NaN.
        if !centre.is_finite()
            || !plus_side.is_finite()
            || !minus_side.is_finite()
            || !mean.is_finite()
        {
            return Err(EstimationError::NonFinite);
        }
        Ok((centre, plus_side, minus_side, mean))
    }

    /// Forms the innovation covariance and the cross-covariance from the predicted measurements,
    /// solves for the gain, and folds `residual` into the estimate.
    fn fold_in(
        &mut self,
        centre: Vector<MEASUREMENT_DIMENSION, T>,
        plus_side: Matrix<MEASUREMENT_DIMENSION, STATE_DIMENSION, T>,
        minus_side: Matrix<MEASUREMENT_DIMENSION, STATE_DIMENSION, T>,
        predicted: Vector<MEASUREMENT_DIMENSION, T>,
        residual: Vector<MEASUREMENT_DIMENSION, T>,
    ) -> Result<(), EstimationError> {
        let (_, centre_covariance, side) = self.weights();

        // How far the middle point sits from the average, on the sensor and in the state.
        let centre_measurement_deviation = centre - predicted;
        let centre_state_deviation = self.propagated_centre - self.state;

        // Two sums built side by side from the same deviations: how much the readings disagree
        // among themselves, and how closely a reading moving tracks the state moving. The
        // measurement noise starts the first sum off rather than being added at the end.
        let mut innovation_covariance = add_weighted_outer_product(
            self.measurement_noise,
            centre_covariance,
            centre_measurement_deviation,
            centre_measurement_deviation,
        );
        let mut cross_covariance = add_weighted_outer_product(
            Matrix::<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>::zeros(),
            centre_covariance,
            centre_state_deviation,
            centre_measurement_deviation,
        );
        // Every outer point adds its own pair of deviations to both sums, all at the same weight.
        for column in 0..STATE_DIMENSION {
            let plus_measurement = Vector::from_fn(|row| plus_side[(row, column)]) - predicted;
            let minus_measurement = Vector::from_fn(|row| minus_side[(row, column)]) - predicted;
            let plus_state =
                Vector::from_fn(|row| self.propagated_plus[(row, column)]) - self.state;
            let minus_state =
                Vector::from_fn(|row| self.propagated_minus[(row, column)]) - self.state;
            innovation_covariance = add_weighted_outer_product(
                innovation_covariance,
                side,
                plus_measurement,
                plus_measurement,
            );
            innovation_covariance = add_weighted_outer_product(
                innovation_covariance,
                side,
                minus_measurement,
                minus_measurement,
            );
            cross_covariance =
                add_weighted_outer_product(cross_covariance, side, plus_state, plus_measurement);
            cross_covariance =
                add_weighted_outer_product(cross_covariance, side, minus_state, minus_measurement);
        }

        // Rounding leaves the two halves of the sum slightly apart; average them back together.
        let innovation_covariance = symmetrized(innovation_covariance);

        // Adding many terms can overflow even when every point was fine on its own.
        if !innovation_covariance.is_finite() || !cross_covariance.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        // The gain weighs the two sums against each other: how much of the reading to believe.
        // Dividing one matrix by another is a solve, never an inverse.
        // Kᵀ = S⁻¹·Pxzᵀ.
        let factorization = innovation_covariance
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?;
        let kalman_gain = factorization
            .solve_matrix::<STATE_DIMENSION>(cross_covariance.transpose())
            .transpose();

        // The gain sets both how far the estimate moves and how much uncertainty the reading
        // takes away. The residual and its spread are kept so the caller can judge the fit.
        self.innovation = residual;
        self.innovation_covariance = innovation_covariance;
        self.state += kalman_gain * residual;
        self.covariance = symmetrized(
            self.covariance - kalman_gain * innovation_covariance * kalman_gain.transpose(),
        );
        Ok(())
    }

    /// Folds `measurement` into the estimate, forming the residual as `measurement − h(state)`.
    ///
    /// Works from the points [`predict`](Self::predict) left behind, so call it after a predict: on
    /// a filter that has never predicted, the points all sit on the initial estimate, the gain
    /// works out to zero, and the estimate is left where it was.
    ///
    /// Use [`update_with_residual`](Self::update_with_residual) when any measurement component is
    /// an angle: plain subtraction is wrong across the ±π wrap.
    ///
    /// Returns [`NonFinite`](EstimationError::NonFinite) when the measurement, a predicted
    /// measurement, or the formed innovation covariance holds an infinity or NaN, and
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
        let (centre, plus_side, minus_side, predicted) =
            self.measurement_sigma_points(measurement_model)?;
        self.fold_in(
            centre,
            plus_side,
            minus_side,
            predicted,
            measurement - predicted,
        )
    }

    /// [`update`](Self::update) with a caller-formed residual, for measurements that plain
    /// subtraction cannot difference correctly.
    ///
    /// A bearing residual must be wrapped to (−π, π] before it reaches the filter: unwrapped, an
    /// error near ±π reads as most of a full turn, and the gain drives the estimate hard the wrong
    /// way — silently, since nothing about the arithmetic is invalid. The filter cannot do this
    /// itself; which components of a `MEASUREMENT_DIMENSION`-vector are angular is not something
    /// the type records. Re-wrapping an angular *state* component after the update is likewise the
    /// caller's, through [`set_state`](Self::set_state).
    ///
    /// One thing this filter asks that the other two do not: **an angular component must be left to
    /// run past ±π inside the process model, not wrapped there.** Everything else this filter
    /// averages is a set of points sitting a fraction of a standard deviation apart, which no wrap
    /// can come between — but a model that wraps its own output puts points at +π and −π at once,
    /// and their average is meaningless. Wrap after the update through
    /// [`set_state`](Self::set_state) instead.
    ///
    /// ```
    /// use multicalc::estimation::UnscentedKalmanFilter;
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
    /// // Nothing is turning the vehicle, so the heading carries over unchanged.
    /// struct HoldsHeading;
    /// impl VectorFn<1, 1> for HoldsHeading {
    ///     fn eval<S: Numeric>(&self, state: &[S; 1]) -> [S; 1] {
    ///         [state[0]]
    ///     }
    /// }
    ///
    /// // Subtract whole turns to fold the angle into a ±π band.
    /// fn wrap_to_pi<T: Numeric>(angle: T) -> T {
    ///     angle - T::TWO_PI * (angle / T::TWO_PI).round()
    /// }
    ///
    /// let mut filter = UnscentedKalmanFilter::<1, 1>::new(
    ///     Vector::new([3.1]),               // heading just under +π
    ///     Matrix::new([[0.1]]),
    ///     Matrix::new([[0.001]]),
    ///     Matrix::new([[0.05]]),
    /// );
    /// filter.predict(&HoldsHeading)?;
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
        let (centre, plus_side, minus_side, predicted) =
            self.measurement_sigma_points(measurement_model)?;
        self.fold_in(centre, plus_side, minus_side, predicted, residual)
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

/// `matrix + weight · left · rightᵀ`, the one accumulation both the covariance and the
/// cross-covariance are built from.
fn add_weighted_outer_product<const ROWS: usize, const COLUMNS: usize, T: Numeric>(
    matrix: Matrix<ROWS, COLUMNS, T>,
    weight: T,
    left: Vector<ROWS, T>,
    right: Vector<COLUMNS, T>,
) -> Matrix<ROWS, COLUMNS, T> {
    matrix + Matrix::from_fn(|row, column| weight * left[row] * right[column])
}

/// Averages a matrix with its own transpose, so rounding cannot leave the two halves disagreeing.
fn symmetrized<const N: usize, T: Numeric>(matrix: Matrix<N, N, T>) -> Matrix<N, N, T> {
    Matrix::from_fn(|row, column| (matrix[(row, column)] + matrix[(column, row)]) / T::TWO)
}
