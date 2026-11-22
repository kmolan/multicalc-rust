//! Error-state Kalman filtering: an IMU-driven filter that tracks where a body is, how fast it is
//! going, which way it faces, and what its own sensors are getting steadily wrong.
//!
//! Non-finite policy: every operation is checked. [`ErrorStateKalmanFilter::predict`] returns
//! [`NonFinite`](EstimationError::NonFinite) when either reading, the timestep, the propagated
//! state, or the resulting covariance holds an infinity or NaN, and
//! [`update`](ErrorStateKalmanFilter::update) does the same for the measurement, the residual, the
//! measurement Jacobian, and the innovation covariance.

use crate::error::EstimationError;
use crate::estimation::CovarianceUpdate;
use crate::linear_algebra::{Matrix, Matrix3D, Vector, Vector3D};
use crate::scalar::{Dual, Numeric};
use crate::spatial::{Quaternion, SO3};

/// How many numbers it takes to say how wrong the running guess is.
/// index  0  1  2    3   4   5    6   7   8    9    10   11    12   13   14
///        δpx δpy δpz δvx δvy δvz  δθx δθy δθz  δbgx δbgy δbgz  δbax δbay δbaz
/// δp — how far off the position is, in world axes, metres
/// δv — how far off the speed is, in world axes, m/s
/// δθ — a small turn taking the estimated facing to the true one, radians
/// δbg — the turn-rate sensor's steady offset, rad/s
/// δba — the push sensor's steady offset, m/s²
const ERROR_DIMENSION: usize = 15;

/// Where the body is, how fast it is going, which way it faces, and what its own sensors read when
/// they should read nothing.
///
/// This is the running guess the filter corrects. The orientation is kept on the rotation group
/// rather than as three angles, so it never has to be renormalized or unwrapped, while the
/// uncertainty about it lives in a flat 15-number error alongside.
///
/// The error's fifteen numbers run in this order, three each:
///
/// | Index range | Meaning |
/// | --- | --- |
/// | 0..3 | where the estimate has the body, in world axes, metres |
/// | 3..6 | how wrong the estimated speed is, world axes, m/s |
/// | 6..9 | a small turn taking the estimated facing to the true one, radians |
/// | 9..12 | the turn-rate sensor's steady offset, rad/s |
/// | 12..15 | the push sensor's steady offset, m/s² |
///
/// Index them by the associated constants rather than by number.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NominalState<T: Numeric = f64> {
    /// Where the body is, in world axes.
    position: Vector3D<T>,
    /// How fast it is going, in world axes.
    velocity: Vector3D<T>,
    /// Which way it faces.
    orientation: SO3<T>,
    /// What the turn-rate sensor reads when the body is still.
    gyroscope_bias: Vector3D<T>,
    /// What the push sensor reads beyond the real push.
    accelerometer_bias: Vector3D<T>,
}

impl<T: Numeric> NominalState<T> {
    /// How many numbers the error carries.
    pub const ERROR_DIMENSION: usize = ERROR_DIMENSION;
    /// Where the position error starts in the error vector.
    pub const POSITION_ERROR_INDEX: usize = 0;
    /// Where the velocity error starts.
    pub const VELOCITY_ERROR_INDEX: usize = 3;
    /// Where the rotation error starts.
    pub const ROTATION_ERROR_INDEX: usize = 6;
    /// Where the turn-rate sensor's offset error starts.
    pub const GYROSCOPE_BIAS_ERROR_INDEX: usize = 9;
    /// Where the push sensor's offset error starts.
    pub const ACCELEROMETER_BIAS_ERROR_INDEX: usize = 12;

    /// Builds a state from all five parts.
    pub fn new(
        position: Vector3D<T>,
        velocity: Vector3D<T>,
        orientation: SO3<T>,
        gyroscope_bias: Vector3D<T>,
        accelerometer_bias: Vector3D<T>,
    ) -> Self {
        NominalState {
            position,
            velocity,
            orientation,
            gyroscope_bias,
            accelerometer_bias,
        }
    }

    /// A body sitting at the origin, not moving, with sensors assumed honest.
    ///
    /// This is the usual starting point: the facing comes from something like
    /// [`SO3::from_two_direction_pairs`], and the biases are learned from there.
    #[must_use]
    pub fn at_rest(orientation: SO3<T>) -> Self {
        NominalState {
            position: Vector::zeros(),
            velocity: Vector::zeros(),
            orientation,
            gyroscope_bias: Vector::zeros(),
            accelerometer_bias: Vector::zeros(),
        }
    }

    /// Where the body is, in world axes.
    #[inline]
    pub fn position(self) -> Vector3D<T> {
        self.position
    }

    /// How fast it is going, in world axes.
    #[inline]
    pub fn velocity(self) -> Vector3D<T> {
        self.velocity
    }

    /// Which way it faces.
    #[inline]
    #[must_use]
    pub fn orientation(self) -> SO3<T> {
        self.orientation
    }

    /// What the turn-rate sensor reads when the body is still.
    #[inline]
    pub fn gyroscope_bias(self) -> Vector3D<T> {
        self.gyroscope_bias
    }

    /// What the push sensor reads beyond the real push.
    #[inline]
    pub fn accelerometer_bias(self) -> Vector3D<T> {
        self.accelerometer_bias
    }

    /// The state with an error folded in.
    ///
    /// The first two and last two groups add straight on. The rotation group is a small turn
    /// applied on the body's own side, so the facing stays a proper rotation instead of drifting
    /// off the group the way three added angles would.
    ///
    /// ```
    /// use multicalc::estimation::NominalState;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::SO3;
    ///
    /// let level = SO3::<f64>::identity();
    /// let state = NominalState::at_rest(level);
    ///
    /// let step_east = 2.0;
    /// let mut error = Vector::zeros();
    /// error[NominalState::<f64>::POSITION_ERROR_INDEX] = step_east;
    ///
    /// let corrected = state.plus_error(error);
    /// assert_eq!(corrected.position()[0], step_east);
    /// ```
    #[must_use]
    pub fn plus_error(self, error: Vector<ERROR_DIMENSION, T>) -> Self {
        let position_error = Vector::new([error[0], error[1], error[2]]);
        let velocity_error = Vector::new([error[3], error[4], error[5]]);
        let rotation_error = Vector::new([error[6], error[7], error[8]]);
        let gyroscope_bias_error = Vector::new([error[9], error[10], error[11]]);
        let accelerometer_bias_error = Vector::new([error[12], error[13], error[14]]);
        NominalState {
            position: self.position + position_error,
            velocity: self.velocity + velocity_error,
            orientation: self.orientation * SO3::exp(rotation_error),
            gyroscope_bias: self.gyroscope_bias + gyroscope_bias_error,
            accelerometer_bias: self.accelerometer_bias + accelerometer_bias_error,
        }
    }

    /// The error that, folded into `reference`, gives this state back.
    ///
    /// The exact inverse of [`plus_error`](Self::plus_error), so the two round-trip for any two
    /// states within a half turn of each other.
    ///
    /// ```
    /// use multicalc::estimation::NominalState;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::SO3;
    ///
    /// let level = SO3::<f64>::identity();
    /// let reference = NominalState::at_rest(level);
    ///
    /// let position = Vector::new([1.0, -2.0, 0.5]);
    /// let velocity = Vector::new([0.1, 0.2, -0.3]);
    /// let quarter_turn = Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]);
    /// let gyroscope_bias = Vector::new([0.01, 0.02, 0.03]);
    /// let accelerometer_bias = Vector::new([0.04, 0.05, 0.06]);
    /// let other = NominalState::new(
    ///     position,
    ///     velocity,
    ///     SO3::exp(quarter_turn),
    ///     gyroscope_bias,
    ///     accelerometer_bias,
    /// );
    ///
    /// let rebuilt = reference.plus_error(other.error_from(reference));
    /// assert!((rebuilt.position() - other.position()).norm() < 1e-12);
    /// assert!((rebuilt.orientation().log() - quarter_turn).norm() < 1e-12);
    /// ```
    pub fn error_from(self, reference: Self) -> Vector<ERROR_DIMENSION, T> {
        let position_error = self.position - reference.position;
        let velocity_error = self.velocity - reference.velocity;
        let rotation_error = (reference.orientation.inverse() * self.orientation).log();
        let gyroscope_bias_error = self.gyroscope_bias - reference.gyroscope_bias;
        let accelerometer_bias_error = self.accelerometer_bias - reference.accelerometer_bias;
        Vector::new([
            position_error[0],
            position_error[1],
            position_error[2],
            velocity_error[0],
            velocity_error[1],
            velocity_error[2],
            rotation_error[0],
            rotation_error[1],
            rotation_error[2],
            gyroscope_bias_error[0],
            gyroscope_bias_error[1],
            gyroscope_bias_error[2],
            accelerometer_bias_error[0],
            accelerometer_bias_error[1],
            accelerometer_bias_error[2],
        ])
    }

    /// The state one IMU step later.
    ///
    /// Each reading has its sensor's steady offset taken off first. What is left of the push
    /// reading is turned into world axes and gravity is added, because the push sensor reads only
    /// what the body is being pushed by and not the fall it is not resisting. The biases carry over
    /// untouched; only a correction moves them.
    ///
    /// ```
    /// use multicalc::estimation::NominalState;
    /// use multicalc::linear_algebra::Vector;
    /// use multicalc::spatial::SO3;
    ///
    /// let level = SO3::<f64>::identity();
    /// let state = NominalState::at_rest(level);
    ///
    /// // Held still against gravity, the push sensor reads a full gravity upward.
    /// let gravity_strength = 9.81;
    /// let gravity = Vector::new([0.0, 0.0, -gravity_strength]);
    /// let gyroscope_reading = Vector::new([0.0, 0.0, 0.0]);
    /// let accelerometer_reading = Vector::new([0.0, 0.0, gravity_strength]);
    /// let timestep = 0.01;
    ///
    /// let next = state.propagated(gyroscope_reading, accelerometer_reading, timestep, gravity);
    /// assert!(next.position().norm() < 1e-15);
    /// assert!(next.velocity().norm() < 1e-15);
    /// ```
    #[must_use]
    pub fn propagated(
        self,
        gyroscope_reading: Vector3D<T>,
        accelerometer_reading: Vector3D<T>,
        timestep: T,
        gravity: Vector3D<T>,
    ) -> Self {
        let turn_rate = gyroscope_reading - self.gyroscope_bias;
        let proper_push = accelerometer_reading - self.accelerometer_bias;
        let world_push = self.orientation.act(proper_push) + gravity;
        NominalState {
            position: self.position
                + self.velocity * timestep
                + world_push * (timestep * timestep * T::HALF),
            velocity: self.velocity + world_push * timestep,
            orientation: self.orientation * SO3::exp(turn_rate * timestep),
            gyroscope_bias: self.gyroscope_bias,
            accelerometer_bias: self.accelerometer_bias,
        }
    }

    /// The same state rebuilt with every number carrying a derivative slot.
    ///
    /// The filter uses this to read a sensor model's sensitivities without the model ever being
    /// written twice. It is public because a caller checking their own model against a hand
    /// derivative wants the same lift.
    #[must_use]
    pub fn lifted_to_dual(self) -> NominalState<Dual<T>> {
        let lift_vector = |vector: Vector3D<T>| {
            Vector::new([
                Dual::constant(vector[0]),
                Dual::constant(vector[1]),
                Dual::constant(vector[2]),
            ])
        };
        let [w, x, y, z] = self.orientation.quaternion().as_array();
        NominalState {
            position: lift_vector(self.position),
            velocity: lift_vector(self.velocity),
            orientation: SO3::from_quaternion(Quaternion::new(
                Dual::constant(w),
                Dual::constant(x),
                Dual::constant(y),
                Dual::constant(z),
            )),
            gyroscope_bias: lift_vector(self.gyroscope_bias),
            accelerometer_bias: lift_vector(self.accelerometer_bias),
        }
    }

    /// True when every number in the state is a real, finite one.
    #[must_use]
    pub fn is_finite(self) -> bool {
        let [w, x, y, z] = self.orientation.quaternion().as_array();
        self.position.is_finite()
            && self.velocity.is_finite()
            && self.gyroscope_bias.is_finite()
            && self.accelerometer_bias.is_finite()
            && w.is_finite()
            && x.is_finite()
            && y.is_finite()
            && z.is_finite()
    }
}

/// A sensor model: what a reading would be, given where the body is and which way it faces.
///
/// Written once and evaluated at whatever kind of number the filter needs — plain values for the
/// predicted reading, and derivative-carrying ones for the sensitivities it needs alongside. **No
/// derivative is ever coded by hand.**
///
/// ```
/// use multicalc::estimation::{NominalState, NominalStateFn};
/// use multicalc::scalar::Numeric;
///
/// // A tracker in the room reports where the drone is, and nothing else.
/// struct RoomTracker;
/// impl NominalStateFn<3> for RoomTracker {
///     fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 3] {
///         *state.position().as_array()
///     }
/// }
/// ```
pub trait NominalStateFn<const MEASUREMENT_DIMENSION: usize> {
    /// The reading this sensor would produce from `state`.
    fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; MEASUREMENT_DIMENSION];
}

/// How noisy an IMU is, in the figures a datasheet quotes.
///
/// The first two say how much each reading jitters from one sample to the next. The last two say
/// how fast each sensor's steady offset wanders over minutes and hours, which is what the filter
/// has to keep chasing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuNoise<T = f64> {
    /// How much the turn-rate reading jitters, in rad/s per square root hertz.
    pub gyroscope_noise_density: T,
    /// How much the push reading jitters, in m/s² per square root hertz.
    pub accelerometer_noise_density: T,
    /// How fast the turn-rate sensor's steady offset wanders, in rad/s² per square root hertz.
    pub gyroscope_bias_random_walk: T,
    /// How fast the push sensor's steady offset wanders, in m/s³ per square root hertz.
    pub accelerometer_bias_random_walk: T,
}

/// An IMU-driven filter that tracks where a body is, how it is moving, which way it faces, and what
/// its own sensors are getting steadily wrong.
///
/// It tracks the *correction* to a running guess rather than the guess itself. That is what lets
/// the facing live on the rotation group, where it can turn any distance without wrapping or
/// needing renormalization, while the uncertainty stays a plain flat fifteen numbers that ordinary
/// matrix arithmetic can carry forward. After every correction the error is folded back into the
/// guess and reset to zero, so it never grows large enough for the flat treatment to strain.
///
/// The fifteen numbers run in this order:
/// index  0  1  2    3   4   5    6   7   8    9    10   11    12   13   14
///        δpx δpy δpz δvx δvy δvz  δθx δθy δθz  δbgx δbgy δbgz  δbax δbay δbaz
/// δp — how far off the position is, in world axes, metres
/// δv — how far off the speed is, in world axes, m/s
/// δθ — a small turn taking the estimated facing to the true one, radians
/// δbg — the turn-rate sensor's steady offset, rad/s
/// δba — the push sensor's steady offset, m/s²
///
/// The turn-rate and push readings drive [`predict`](Self::predict); corrections come from any
/// sensor a caller can write as a [`NominalStateFn`].
///
/// **Two generic parameters, where [`ExtendedKalmanFilter`](crate::estimation::ExtendedKalmanFilter)
/// has four.** The state width is fixed at fifteen by the formulation, so it is not a parameter.
/// There is no pluggable differentiation backend either: a stepped difference over an error state
/// is not meaningful, because the error is identically zero and a finite step would move a point on
/// the rotation group by an amount the sensor model cannot tell from real signal. The Jacobian is
/// always taken exactly, with [`Dual`].
///
/// Cost: `predict` is two 15-cubed matrix products, with the transition written in closed form
/// rather than differentiated. `update` adds fifteen model evaluations at twice the scalar width,
/// one `MEASUREMENT_DIMENSION`-square Cholesky factorization, and the reset product. Corrections
/// are expected at a few hertz against predictions at a kilohertz, which is what makes that
/// asymmetry the right trade.
///
/// # Examples
/// ```
/// use multicalc::estimation::{ErrorStateKalmanFilter, ImuNoise, NominalState, NominalStateFn};
/// use multicalc::linear_algebra::{Matrix, Vector};
/// use multicalc::scalar::Numeric;
/// use multicalc::spatial::SO3;
/// # fn main() -> Result<(), multicalc::error::EstimationError> {
/// // A tracker in the room reports where the drone is, and nothing else.
/// struct RoomTracker;
/// impl NominalStateFn<3> for RoomTracker {
///     fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 3] {
///         *state.position().as_array()
///     }
/// }
///
/// let level = SO3::<f64>::identity();
/// let initial_covariance = Matrix::from_diagonal([0.1; 15]);
/// let imu_noise = ImuNoise {
///     gyroscope_noise_density: 0.02,
///     accelerometer_noise_density: 0.05,
///     gyroscope_bias_random_walk: 1e-4,
///     accelerometer_bias_random_walk: 1e-3,
/// };
/// let tracker_spread = 0.03;
/// let measurement_noise = Matrix::from_diagonal([tracker_spread * tracker_spread; 3]);
/// let mut filter = ErrorStateKalmanFilter::<3>::new(
///     NominalState::at_rest(level),
///     initial_covariance,
///     imu_noise,
///     measurement_noise,
/// );
///
/// // Sitting still, the push sensor reads a full gravity upward.
/// let gravity_strength = 9.81;
/// let gyroscope_reading = Vector::new([0.0, 0.0, 0.0]);
/// let accelerometer_reading = Vector::new([0.0, 0.0, gravity_strength]);
/// let timestep = 0.001;
/// filter.predict(gyroscope_reading, accelerometer_reading, timestep)?;
///
/// // The tracker says the drone is a little east of where the filter has it.
/// let step_east = 0.1;
/// let fix = Vector::new([step_east, 0.0, 0.0]);
/// filter.update(&RoomTracker, fix)?;
/// assert!(filter.nominal_state().position()[0] > 0.0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorStateKalmanFilter<const MEASUREMENT_DIMENSION: usize, T: Numeric = f64> {
    nominal_state: NominalState<T>,
    covariance: Matrix<ERROR_DIMENSION, ERROR_DIMENSION, T>,
    imu_noise: ImuNoise<T>,
    measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    gravity: Vector3D<T>,
    innovation: Vector<MEASUREMENT_DIMENSION, T>,
    innovation_covariance: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    covariance_update: CovarianceUpdate,
}

impl<const MEASUREMENT_DIMENSION: usize, T: Numeric>
    ErrorStateKalmanFilter<MEASUREMENT_DIMENSION, T>
{
    /// Builds a filter around a starting guess and how much it is trusted.
    ///
    /// Gravity starts at `(0, 0, −9.81)`, matching a world frame with z pointing up; change it with
    /// [`with_gravity`](Self::with_gravity). The covariance update starts at
    /// [`Joseph`](CovarianceUpdate::Joseph); change it with
    /// [`with_covariance_update`](Self::with_covariance_update).
    pub fn new(
        initial_state: NominalState<T>,
        initial_covariance: Matrix<ERROR_DIMENSION, ERROR_DIMENSION, T>,
        imu_noise: ImuNoise<T>,
        measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    ) -> Self {
        const {
            assert!(
                MEASUREMENT_DIMENSION > 0,
                "ErrorStateKalmanFilter: MEASUREMENT_DIMENSION must be non-zero"
            )
        };
        ErrorStateKalmanFilter {
            nominal_state: initial_state,
            covariance: initial_covariance,
            imu_noise,
            measurement_noise,
            gravity: Vector::new([T::ZERO, T::ZERO, T::from_f64(-9.81)]),
            innovation: Vector::zeros(),
            innovation_covariance: Matrix::zeros(),
            covariance_update: CovarianceUpdate::Joseph,
        }
    }

    /// Selects how [`update`](Self::update) recomputes the covariance.
    #[must_use]
    pub const fn with_covariance_update(mut self, covariance_update: CovarianceUpdate) -> Self {
        self.covariance_update = covariance_update;
        self
    }

    /// Replaces the pull of gravity in world axes. Starts at `(0, 0, −9.81)`.
    #[must_use]
    pub fn with_gravity(mut self, gravity: Vector3D<T>) -> Self {
        self.gravity = gravity;
        self
    }

    /// Replaces the running guess, for a caller who has re-initialized it from scratch.
    pub fn set_nominal_state(&mut self, nominal_state: NominalState<T>) {
        self.nominal_state = nominal_state;
    }

    /// Replaces the IMU noise settings, for a re-tuning or a changed sample rate.
    pub fn set_imu_noise(&mut self, imu_noise: ImuNoise<T>) {
        self.imu_noise = imu_noise;
    }

    /// Replaces the measurement noise.
    pub fn set_measurement_noise(
        &mut self,
        measurement_noise: Matrix<MEASUREMENT_DIMENSION, MEASUREMENT_DIMENSION, T>,
    ) {
        self.measurement_noise = measurement_noise;
    }

    // ----- Predict -----

    /// Rolls the guess and its spread forward by one IMU sample.
    ///
    /// Returns [`NonFinite`](EstimationError::NonFinite) when either reading, the timestep, the
    /// propagated state, or the resulting covariance holds an infinity or NaN.
    /// [`Diff`](EstimationError::Diff) is never returned: this step has no derivative to take.
    pub fn predict(
        &mut self,
        gyroscope_reading: Vector3D<T>,
        accelerometer_reading: Vector3D<T>,
        timestep: T,
    ) -> Result<(), EstimationError> {
        if !gyroscope_reading.is_finite()
            || !accelerometer_reading.is_finite()
            || !timestep.is_finite()
        {
            return Err(EstimationError::NonFinite);
        }

        // Built before the guess moves, because it describes the step from where the guess is now.
        let transition =
            self.error_state_transition(gyroscope_reading, accelerometer_reading, timestep);
        let propagated = self.nominal_state.propagated(
            gyroscope_reading,
            accelerometer_reading,
            timestep,
            self.gravity,
        );
        if !propagated.is_finite() || !transition.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        self.nominal_state = propagated;
        self.propagate_covariance(transition, timestep);
        if !self.covariance.is_finite() {
            return Err(EstimationError::NonFinite);
        }
        Ok(())
    }

    /// How an error in the current estimate carries into the next step.
    ///
    /// Rows and columns run in the error order `[position, velocity, rotation, gyroscope bias,
    /// accelerometer bias]`, three each. [`predict`](Self::predict) uses this to carry the
    /// covariance forward; it is public so a caller doing their own smoothing can reach the same
    /// matrix.
    ///
    /// It is exact in the velocity and rotation rows and first-order in the timestep elsewhere: the
    /// position row leaves out a half-timestep-squared term, and the rotation row's dependence on
    /// the turn-rate bias leaves out a similar one. Both are the standard error-state form.
    pub fn error_state_transition(
        &self,
        gyroscope_reading: Vector3D<T>,
        accelerometer_reading: Vector3D<T>,
        timestep: T,
    ) -> Matrix<ERROR_DIMENSION, ERROR_DIMENSION, T> {
        let turn_rate = gyroscope_reading - self.nominal_state.gyroscope_bias();
        let proper_push = accelerometer_reading - self.nominal_state.accelerometer_bias();
        let rotation = self.nominal_state.orientation().to_matrix();

        // A tilt in the estimated facing points the measured push the wrong way, and a wrong push
        // offset feeds straight through the same turn into the world.
        let push_sensitivity = -(rotation * SO3::hat(proper_push)).scale(timestep);
        let bias_sensitivity = -rotation.scale(timestep);
        // The step's own turn, taken backwards: an error measured before the turn is measured
        // against a differently-pointed frame after it.
        let turn_transfer = SO3::exp(-turn_rate * timestep).to_matrix();

        let mut transition = Matrix::<ERROR_DIMENSION, ERROR_DIMENSION, T>::identity();
        write_block(&mut transition, 0, 3, Matrix3D::identity().scale(timestep));
        write_block(&mut transition, 3, 6, push_sensitivity);
        write_block(&mut transition, 3, 12, bias_sensitivity);
        write_block(&mut transition, 6, 6, turn_transfer);
        write_block(&mut transition, 6, 9, Matrix3D::identity().scale(-timestep));
        transition
    }

    /// Carries the spread through the step and adds what the IMU's own noise put into it.
    fn propagate_covariance(
        &mut self,
        transition: Matrix<ERROR_DIMENSION, ERROR_DIMENSION, T>,
        timestep: T,
    ) {
        self.covariance = transition * self.covariance * transition.transpose();

        // The noise goes straight onto four diagonal blocks: what carries sensor noise into the
        // error only picks blocks out, so there is no product to form. Jitter in a reading is
        // integrated once over the step, so it lands as `(density · Δt)²`, while an offset wandering
        // is itself a random walk, so it lands as `density² · Δt`.
        let velocity_noise = self.imu_noise.accelerometer_noise_density * timestep;
        let rotation_noise = self.imu_noise.gyroscope_noise_density * timestep;
        let gyroscope_walk = self.imu_noise.gyroscope_bias_random_walk;
        let accelerometer_walk = self.imu_noise.accelerometer_bias_random_walk;
        for offset in 0..3 {
            self.covariance[(3 + offset, 3 + offset)] += velocity_noise * velocity_noise;
            self.covariance[(6 + offset, 6 + offset)] += rotation_noise * rotation_noise;
            self.covariance[(9 + offset, 9 + offset)] += gyroscope_walk * gyroscope_walk * timestep;
            self.covariance[(12 + offset, 12 + offset)] +=
                accelerometer_walk * accelerometer_walk * timestep;
        }

        self.covariance = self.covariance.symmetrized();
    }

    // ----- Update -----

    /// Folds `measurement` into the estimate, forming the residual as `measurement − h(state)`.
    ///
    /// Use [`update_with_residual`](Self::update_with_residual) when any measurement component is
    /// an angle: plain subtraction is wrong across the ±π wrap.
    ///
    /// Returns [`NonFinite`](EstimationError::NonFinite) when the measurement, the residual, the
    /// measurement Jacobian, or the formed innovation covariance holds an infinity or NaN, and
    /// [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) when the innovation covariance
    /// cannot be factorized — the gain is undefined.
    /// [`Diff`](EstimationError::Diff) is never returned: the Jacobian is always exact.
    pub fn update<MeasurementModel>(
        &mut self,
        measurement_model: &MeasurementModel,
        measurement: Vector<MEASUREMENT_DIMENSION, T>,
    ) -> Result<(), EstimationError>
    where
        MeasurementModel: NominalStateFn<MEASUREMENT_DIMENSION>,
    {
        if !measurement.is_finite() {
            return Err(EstimationError::NonFinite);
        }
        let predicted = Vector::new(measurement_model.eval(&self.nominal_state));
        self.update_with_residual(measurement_model, measurement - predicted)
    }

    /// [`update`](Self::update) with a caller-formed residual, for measurements that plain
    /// subtraction cannot difference correctly.
    ///
    /// A bearing residual must be wrapped to (−π, π] before it reaches the filter: unwrapped, an
    /// error near ±π reads as most of a full turn, and the gain drives the estimate hard the wrong
    /// way — silently, since nothing about the arithmetic is invalid. The filter cannot do this
    /// itself; which components of a `MEASUREMENT_DIMENSION`-vector are angular is not something
    /// the type records.
    ///
    /// ```
    /// use multicalc::estimation::{ErrorStateKalmanFilter, ImuNoise, NominalState, NominalStateFn};
    /// use multicalc::linear_algebra::{Matrix, Vector};
    /// use multicalc::scalar::Numeric;
    /// use multicalc::spatial::SO3;
    /// # fn main() -> Result<(), multicalc::error::EstimationError> {
    /// // A compass reads which way the body faces about the vertical.
    /// struct Compass;
    /// impl NominalStateFn<1> for Compass {
    ///     fn eval<S: Numeric>(&self, state: &NominalState<S>) -> [S; 1] {
    ///         let forward = state.orientation().act(Vector::new([S::ONE, S::ZERO, S::ZERO]));
    ///         [forward[1].atan2(forward[0])]
    ///     }
    /// }
    ///
    /// let heading = 3.1;
    /// let facing = SO3::exp(Vector::new([0.0, 0.0, heading]));
    /// let imu_noise = ImuNoise {
    ///     gyroscope_noise_density: 0.02,
    ///     accelerometer_noise_density: 0.05,
    ///     gyroscope_bias_random_walk: 1e-4,
    ///     accelerometer_bias_random_walk: 1e-3,
    /// };
    /// let compass_spread = 0.05;
    /// let mut filter = ErrorStateKalmanFilter::<1>::new(
    ///     NominalState::at_rest(facing),
    ///     Matrix::from_diagonal([0.1; 15]),
    ///     imu_noise,
    ///     Matrix::from_diagonal([compass_spread * compass_spread; 1]),
    /// );
    ///
    /// // The compass reads just the other side of the half turn: a small error, not most of one.
    /// let reading = -3.1;
    /// let predicted = Compass.eval(&filter.nominal_state());
    /// let residual = Vector::new([(reading - predicted[0]).wrap_to_pi()]);
    /// filter.update_with_residual(&Compass, residual)?;
    ///
    /// // The estimate steps a little further round the way the compass pointed, rather than most
    /// // of the way back around the circle.
    /// let corrected = Compass.eval(&filter.nominal_state());
    /// let moved = (corrected[0] - predicted[0]).wrap_to_pi();
    /// assert!(moved > 0.0 && moved < residual[0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_with_residual<MeasurementModel>(
        &mut self,
        measurement_model: &MeasurementModel,
        residual: Vector<MEASUREMENT_DIMENSION, T>,
    ) -> Result<(), EstimationError>
    where
        MeasurementModel: NominalStateFn<MEASUREMENT_DIMENSION>,
    {
        let innovation_covariance =
            self.fold_in(measurement_model, residual, self.measurement_noise)?;
        self.innovation = residual;
        self.innovation_covariance = innovation_covariance;
        Ok(())
    }

    /// [`update_with_residual`](Self::update_with_residual) for a sensor of a different width,
    /// folded in through the same gain and the same reset.
    ///
    /// A vehicle usually has more than one kind of correction — a three-number position fix and a
    /// one-number heading aid, say — and only one of them can set the width the filter is declared
    /// with. This takes the other, with its own noise, per call.
    ///
    /// [`innovation`](Self::innovation) and
    /// [`innovation_covariance`](Self::innovation_covariance) are not written here: their width is
    /// fixed by the type, so they keep reporting the last correction of the declared width.
    pub fn update_other<const OTHER_DIMENSION: usize, MeasurementModel>(
        &mut self,
        measurement_model: &MeasurementModel,
        residual: Vector<OTHER_DIMENSION, T>,
        measurement_noise: Matrix<OTHER_DIMENSION, OTHER_DIMENSION, T>,
    ) -> Result<(), EstimationError>
    where
        MeasurementModel: NominalStateFn<OTHER_DIMENSION>,
    {
        // The innovation covariance is dropped: it is this sensor's width, not the filter's, so
        // there is nowhere on the type to report it.
        let _ = self.fold_in(measurement_model, residual, measurement_noise)?;
        Ok(())
    }

    /// The correction itself, at whatever width the sensor has. Returns the innovation covariance
    /// it formed, which only a caller of the filter's own declared width keeps.
    fn fold_in<const WIDTH: usize, MeasurementModel>(
        &mut self,
        measurement_model: &MeasurementModel,
        residual: Vector<WIDTH, T>,
        measurement_noise: Matrix<WIDTH, WIDTH, T>,
    ) -> Result<Matrix<WIDTH, WIDTH, T>, EstimationError>
    where
        MeasurementModel: NominalStateFn<WIDTH>,
    {
        if !residual.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        let measurement_jacobian = self.measurement_jacobian(measurement_model);
        if !measurement_jacobian.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        let innovation_covariance =
            measurement_jacobian * self.covariance * measurement_jacobian.transpose()
                + measurement_noise;
        if !innovation_covariance.is_finite() {
            return Err(EstimationError::NonFinite);
        }

        // Kᵀ = S⁻¹·H·Pᵀ, solved rather than inverted. Pᵀ is written out so an asymmetric
        // caller-seeded covariance still gives the exact gain.
        let projected = measurement_jacobian * self.covariance.transpose();
        let kalman_gain = innovation_covariance
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?
            .solve_matrix::<ERROR_DIMENSION>(projected)
            .transpose();

        let error = kalman_gain * residual;

        let residual_transfer = Matrix::<ERROR_DIMENSION, ERROR_DIMENSION, T>::identity()
            - kalman_gain * measurement_jacobian;
        self.covariance = match self.covariance_update {
            CovarianceUpdate::Joseph => {
                residual_transfer * self.covariance * residual_transfer.transpose()
                    + kalman_gain * measurement_noise * kalman_gain.transpose()
            }
            CovarianceUpdate::Naive => residual_transfer * self.covariance,
        };

        self.inject_error_and_reset(error);
        self.covariance = self.covariance.symmetrized();
        Ok(innovation_covariance)
    }

    /// Fifteen seeded passes through the caller's model, one per error direction, each reading how
    /// every output moves. The error is zero at this point, so each pass asks what a small nudge in
    /// one direction would do to the reading.
    fn measurement_jacobian<const WIDTH: usize, MeasurementModel>(
        &self,
        measurement_model: &MeasurementModel,
    ) -> Matrix<WIDTH, ERROR_DIMENSION, T>
    where
        MeasurementModel: NominalStateFn<WIDTH>,
    {
        let lifted = self.nominal_state.lifted_to_dual();
        let mut jacobian = Matrix::<WIDTH, ERROR_DIMENSION, T>::zeros();
        for column in 0..ERROR_DIMENSION {
            let mut error = [Dual::constant(T::ZERO); ERROR_DIMENSION];
            error[column] = Dual::variable(T::ZERO);
            let outputs = measurement_model.eval(&lifted.plus_error(Vector::new(error)));
            for row in 0..WIDTH {
                jacobian[(row, column)] = outputs[row].deriv;
            }
        }
        jacobian
    }

    /// Folds an error into the running estimate and shrinks the spread to match.
    ///
    /// [`update`](Self::update) calls this itself, so a caller never has to. It is public so the
    /// step can be exercised directly with a known error.
    ///
    /// Folding the error in moves the orientation, and the leftover spread was measured about the
    /// old one, so it is carried across by `I − ½[δθ]×` on the rotation block — close to the
    /// identity for a small correction, and the thing that keeps the spread honest when the
    /// correction is not small.
    pub fn inject_error_and_reset(&mut self, error: Vector<ERROR_DIMENSION, T>) {
        self.nominal_state = self.nominal_state.plus_error(error);

        let rotation_error = Vector::new([error[6], error[7], error[8]]);
        let rotation_block = Matrix3D::<T>::identity() - SO3::hat(rotation_error).scale(T::HALF);
        let mut reset = Matrix::<ERROR_DIMENSION, ERROR_DIMENSION, T>::identity();
        write_block(&mut reset, 6, 6, rotation_block);

        self.covariance = reset * self.covariance * reset.transpose();
    }

    /// Evens the spread out and lifts any direction that has gone negative back to
    /// `minimum_eigenvalue`.
    ///
    /// Rounding over a long run can leave the spread claiming negative uncertainty in some
    /// direction, which makes the next gain meaningless. This repairs it. It is deliberately not
    /// automatic: working out the directions costs far more than a filter step does, so a caller on
    /// a fast loop runs it on a slow schedule — once a second, or on a health check — rather than
    /// every tick. Evening the spread out *is* automatic, on every predict and update; only the
    /// lifting is left to the caller.
    ///
    /// Returns [`NonFinite`](EstimationError::NonFinite) when the covariance holds an infinity or
    /// NaN.
    pub fn condition_covariance(&mut self, minimum_eigenvalue: T) -> Result<(), EstimationError> {
        let evened = self.covariance.symmetrized();
        // The decomposition fails only on a non-finite entry or a lopsided matrix, and the second
        // cannot happen after evening the halves out, so both collapse to the same report.
        let directions = evened
            .symmetric_eigendecomposition()
            .map_err(|_| EstimationError::NonFinite)?;
        self.covariance = directions.clamped(minimum_eigenvalue);
        Ok(())
    }

    // ----- Accessors -----

    /// The current running guess.
    #[must_use]
    pub fn nominal_state(&self) -> NominalState<T> {
        self.nominal_state
    }

    /// The current spread over the fifteen error numbers.
    pub fn covariance(&self) -> Matrix<ERROR_DIMENSION, ERROR_DIMENSION, T> {
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

    /// How far the estimate is from a known truth, measured against its own claimed spread.
    ///
    /// Only a test or a simulation has the truth to pass in. Over many runs the average of this
    /// should sit near fifteen, one for each error number: much larger means the filter is more
    /// wrong than it admits, much smaller means it is needlessly unsure. The facing's share is
    /// taken as the turn between the two orientations rather than a subtraction, which is a
    /// difference only the filter can form correctly.
    ///
    /// Returns [`NotPositiveDefinite`](EstimationError::NotPositiveDefinite) if the covariance
    /// cannot be factorized.
    pub fn normalized_estimation_error_squared(
        &self,
        true_state: NominalState<T>,
    ) -> Result<T, EstimationError> {
        let error = true_state.error_from(self.nominal_state);
        let weighted = self
            .covariance
            .cholesky()
            .map_err(|_| EstimationError::NotPositiveDefinite)?
            .solve(error);
        Ok(error.dot(weighted))
    }
}

/// Writes a 3×3 block into a bigger square matrix, top-left corner first.
fn write_block<const N: usize, T: Numeric>(
    target: &mut Matrix<N, N, T>,
    top_row: usize,
    left_column: usize,
    block: Matrix3D<T>,
) {
    for row in 0..3 {
        for column in 0..3 {
            target[(top_row + row, left_column + column)] = block[(row, column)];
        }
    }
}
