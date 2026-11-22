//! A light attitude filter: a body's facing, from a turn-rate sensor pulled gently onto what a
//! push sensor and a magnetometer say, with the turn-rate sensor's steady offset learned as it
//! goes.
//!
//! Non-finite policy: every operation is checked. [`MahonyFilter::step`] returns
//! [`NonFinite`](EstimationError::NonFinite) when any reading, the timestep, or the resulting
//! facing or offset holds an infinity or NaN, and leaves the filter untouched when it does.

use crate::error::EstimationError;
use crate::estimation::attitude_correction;
use crate::linear_algebra::Vector3D;
use crate::scalar::Numeric;
use crate::spatial::SO3;

/// A body's facing and its turn-rate sensor's steady offset, worked out from a turn-rate sensor, a
/// push sensor, and optionally a magnetometer.
///
/// A turn-rate sensor alone gives a smooth facing that slowly wanders. A push sensor alone says
/// which way is down but jumps about whenever the body is pushed, and a magnetometer alone says
/// which way is north and is easily disturbed. This filter takes the turn rate as the answer and
/// nudges it, every tick, by however far the other two say it is off. How hard it nudges is one
/// gain; how much of the running total of those nudges it blames on a sensor offset is the other.
///
/// It carries a facing and a three-number offset and nothing else — no place, no speed, and no
/// spread. When a spread is wanted, that is
/// [`ErrorStateKalmanFilter`](crate::estimation::ErrorStateKalmanFilter)'s job, at roughly a
/// hundred times the arithmetic. This one is a handful of cross products and one exponential a
/// tick, the same work whatever the readings are.
///
/// The world it works in has z up and x north by default; change that with
/// [`with_reference_directions`](Self::with_reference_directions). The starting facing usually
/// comes from [`SO3::from_two_direction_pairs`] on a still body.
///
/// # Examples
/// ```
/// use multicalc::estimation::MahonyFilter;
/// use multicalc::linear_algebra::Vector;
/// use multicalc::spatial::SO3;
/// # fn main() -> Result<(), multicalc::error::EstimationError> {
/// // Starting off level by about a tenth of a radian, on a body that is in fact still and level.
/// let tilt = Vector::new([0.1, -0.05, 0.0]);
/// let mut filter = MahonyFilter::new(SO3::exp(tilt));
///
/// let gravity_strength = 9.81;
/// let not_turning = Vector::new([0.0, 0.0, 0.0]);
/// let one_gravity_up = Vector::new([0.0, 0.0, gravity_strength]);
/// let field_north = Vector::new([1.0, 0.0, 0.0]);
/// let timestep = 0.005;
/// let ticks = 12_000; // a minute at 200 Hz
/// for _ in 0..ticks {
///     filter.step(not_turning, one_gravity_up, Some(field_north), timestep)?;
/// }
///
/// // It has found level. The lean is gone within a few seconds; the rest of the minute is the
/// // running total of the nudges unwinding, since the sensor turned out to have no offset.
/// assert!(filter.orientation().log().norm() < 1e-3);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MahonyFilter<T: Numeric = f64> {
    /// Which way the body is facing.
    orientation: SO3<T>,
    /// What the turn-rate sensor reads when the body is still.
    gyroscope_bias: Vector3D<T>,
    /// How hard each tick's nudge pulls the facing.
    proportional_gain: T,
    /// How much of the running total of those nudges is blamed on the sensor's offset.
    integral_gain: T,
    /// Which way is up, in world axes.
    upward_reference: Vector3D<T>,
    /// Which way is north, in world axes.
    north_reference: Vector3D<T>,
}

impl<T: Numeric> MahonyFilter<T> {
    /// Builds a filter around a starting facing.
    ///
    /// The offset starts at zero, the gains at 1 and 0.1, and the world at z up and x north;
    /// change them with [`with_proportional_gain`](Self::with_proportional_gain),
    /// [`with_integral_gain`](Self::with_integral_gain), and
    /// [`with_reference_directions`](Self::with_reference_directions).
    pub fn new(initial_orientation: SO3<T>) -> Self {
        MahonyFilter {
            orientation: initial_orientation,
            gyroscope_bias: Vector3D::zeros(),
            proportional_gain: T::ONE,
            integral_gain: T::from_f64(0.1),
            upward_reference: attitude_correction::upward_reference(),
            north_reference: attitude_correction::north_reference(),
        }
    }

    /// Replaces how hard each tick's nudge pulls the facing. Starts at 1.
    #[must_use]
    pub fn with_proportional_gain(mut self, proportional_gain: T) -> Self {
        self.proportional_gain = proportional_gain;
        self
    }

    /// Replaces how much of the nudging is blamed on the sensor's offset. Starts at 0.1.
    #[must_use]
    pub fn with_integral_gain(mut self, integral_gain: T) -> Self {
        self.integral_gain = integral_gain;
        self
    }

    /// Replaces which way is up and which way is north, in world axes. Starts at `(0, 0, 1)` and
    /// `(1, 0, 0)`.
    ///
    /// North is squared up against up before it is stored, so the two need only be roughly at right
    /// angles. A pair that does not describe two usable directions leaves both as they were.
    #[must_use]
    pub fn with_reference_directions(
        mut self,
        upward_reference: Vector3D<T>,
        north_reference: Vector3D<T>,
    ) -> Self {
        let Some(upward) = upward_reference.try_normalized() else {
            return self;
        };
        let levelled = north_reference - upward * north_reference.dot(upward);
        let Some(north) = levelled.try_normalized() else {
            return self;
        };
        self.upward_reference = upward;
        self.north_reference = north;
        self
    }

    /// Replaces the facing, for a caller re-seeding it from a still-body fix.
    pub fn set_orientation(&mut self, orientation: SO3<T>) {
        self.orientation = orientation;
    }

    /// Replaces the turn-rate sensor's offset, for a caller restoring a saved one at start-up.
    pub fn set_gyroscope_bias(&mut self, gyroscope_bias: Vector3D<T>) {
        self.gyroscope_bias = gyroscope_bias;
    }

    /// Rolls the facing forward one tick and pulls it toward what the sensors say.
    ///
    /// Pass `None` for the magnetometer when there is not one, or when its reading is not to be
    /// trusted this tick; the heading then rides on the turn-rate sensor alone and slowly wanders,
    /// while the lean stays pinned by the push sensor.
    ///
    /// Returns [`NonFinite`](EstimationError::NonFinite) when any reading, the timestep, or the
    /// result holds an infinity or NaN, and leaves the filter untouched when it does.
    pub fn step(
        &mut self,
        gyroscope_reading: Vector3D<T>,
        accelerometer_reading: Vector3D<T>,
        magnetometer_reading: Option<Vector3D<T>>,
        timestep: T,
    ) -> Result<(), EstimationError> {
        if !attitude_correction::readings_are_finite(
            gyroscope_reading,
            accelerometer_reading,
            magnetometer_reading,
            timestep,
        ) {
            return Err(EstimationError::NonFinite);
        }

        let correction = attitude_correction::correction(
            self.orientation,
            accelerometer_reading,
            magnetometer_reading,
            self.upward_reference,
            self.north_reference,
        );

        // The offset moves first, so the turn rate is corrected by the offset this tick worked out.
        let gyroscope_bias = self.gyroscope_bias - correction * self.integral_gain * timestep;
        let corrected_rate =
            gyroscope_reading - gyroscope_bias + correction * self.proportional_gain;
        let orientation =
            attitude_correction::stepped_orientation(self.orientation, corrected_rate, timestep);

        if !gyroscope_bias.is_finite() || !attitude_correction::orientation_is_finite(orientation) {
            return Err(EstimationError::NonFinite);
        }

        self.gyroscope_bias = gyroscope_bias;
        self.orientation = orientation;
        Ok(())
    }

    /// Rolls the facing forward one tick with no magnetometer.
    ///
    /// The lean stays pinned by the push sensor; the heading rides on the turn-rate sensor alone
    /// and slowly wanders.
    pub fn step_without_magnetometer(
        &mut self,
        gyroscope_reading: Vector3D<T>,
        accelerometer_reading: Vector3D<T>,
        timestep: T,
    ) -> Result<(), EstimationError> {
        self.step(gyroscope_reading, accelerometer_reading, None, timestep)
    }

    /// Which way the body is facing, as worked out so far.
    #[inline]
    #[must_use]
    pub fn orientation(&self) -> SO3<T> {
        self.orientation
    }

    /// What the turn-rate sensor is reading when the body is not turning, rad/s.
    #[inline]
    pub fn gyroscope_bias(&self) -> Vector3D<T> {
        self.gyroscope_bias
    }
}
