//! A light attitude filter: a body's facing, from a turn-rate sensor nudged a fixed amount each
//! tick toward what a push sensor and a magnetometer say, with the turn-rate sensor's steady
//! offset learned as it goes.
//!
//! Non-finite policy: every operation is checked. [`MadgwickFilter::step`] returns
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
/// Where [`MahonyFilter`](crate::estimation::MahonyFilter) nudges harder the more wrong it is, this
/// one always nudges by the same amount and only picks the direction from the readings. That makes
/// one gain, in radians per second, the whole tuning story: it is how fast the filter is willing to
/// walk toward the sensors, and it does not change whether the facing is a degree out or ninety.
/// Recovering from a badly wrong start therefore takes a predictable amount of time, and a burst of
/// nonsense from a knocked sensor can only move the facing by that same rate.
///
/// A second gain says how much of that walking to blame on the turn-rate sensor having a steady
/// offset. Set it to zero for the published filter's behaviour, which learns no offset.
///
/// **How the facing is carried forward differs from the published version.** The nudge direction
/// and the meaning of the gain are the same, but the facing is turned by composing the turn it
/// makes over the step rather than by adding a rate to four loose numbers and scaling them back.
/// What comes back is a true rotation with nothing to repair, and the two agree to the square of
/// the step size.
///
/// The world it works in has z up and x north by default; change that with
/// [`with_reference_directions`](Self::with_reference_directions). The starting facing usually
/// comes from [`SO3::from_two_direction_pairs`] on a still body.
///
/// # Examples
/// ```
/// use multicalc::estimation::MadgwickFilter;
/// use multicalc::linear_algebra::Vector;
/// use multicalc::spatial::SO3;
/// # fn main() -> Result<(), multicalc::error::EstimationError> {
/// // Starting off level by about a tenth of a radian, on a body that is in fact still and level.
/// let tilt = Vector::new([0.1, -0.05, 0.0]);
/// let mut filter = MadgwickFilter::new(SO3::exp(tilt));
///
/// let gravity_strength = 9.81;
/// let not_turning = Vector::new([0.0, 0.0, 0.0]);
/// let one_gravity_up = Vector::new([0.0, 0.0, gravity_strength]);
/// let field_north = Vector::new([1.0, 0.0, 0.0]);
/// let timestep = 0.005;
/// let ticks = 2_000; // ten seconds at 200 Hz
/// for _ in 0..ticks {
///     filter.step(not_turning, one_gravity_up, Some(field_north), timestep)?;
/// }
///
/// // It has found level, to within the one step of walking it takes at a time.
/// assert!(filter.orientation().log().norm() < 1e-3);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MadgwickFilter<T: Numeric = f64> {
    /// Which way the body is facing.
    orientation: SO3<T>,
    /// What the turn-rate sensor reads when the body is still.
    gyroscope_bias: Vector3D<T>,
    /// How fast the filter walks toward the sensors, rad/s.
    correction_gain: T,
    /// How much of that walking is blamed on the sensor's offset.
    bias_gain: T,
    /// Which way is up, in world axes.
    upward_reference: Vector3D<T>,
    /// Which way is north, in world axes.
    north_reference: Vector3D<T>,
}

impl<T: Numeric> MadgwickFilter<T> {
    /// Builds a filter around a starting facing.
    ///
    /// The offset starts at zero, the gains at 0.1 and 0.01, and the world at z up and x north;
    /// change them with [`with_correction_gain`](Self::with_correction_gain),
    /// [`with_bias_gain`](Self::with_bias_gain), and
    /// [`with_reference_directions`](Self::with_reference_directions).
    pub fn new(initial_orientation: SO3<T>) -> Self {
        MadgwickFilter {
            orientation: initial_orientation,
            gyroscope_bias: Vector3D::zeros(),
            correction_gain: T::from_f64(0.1),
            bias_gain: T::from_f64(0.01),
            upward_reference: attitude_correction::upward_reference(),
            north_reference: attitude_correction::north_reference(),
        }
    }

    /// Replaces how fast the filter walks toward the sensors, rad/s. Starts at 0.1.
    #[must_use]
    pub fn with_correction_gain(mut self, correction_gain: T) -> Self {
        self.correction_gain = correction_gain;
        self
    }

    /// Replaces how much of the walking is blamed on the sensor's offset. Starts at 0.01; set it to
    /// zero to leave the offset alone.
    #[must_use]
    pub fn with_bias_gain(mut self, bias_gain: T) -> Self {
        self.bias_gain = bias_gain;
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

    /// Rolls the facing forward one tick and walks it toward what the sensors say.
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
        // Only the direction is kept, so the pull is the same rate however wrong the facing is.
        let direction = correction.try_normalized().unwrap_or_else(Vector3D::zeros);

        // The offset moves first, so the turn rate is corrected by the offset this tick worked out.
        let gyroscope_bias = self.gyroscope_bias - direction * self.bias_gain * timestep;
        let corrected_rate = gyroscope_reading - gyroscope_bias + direction * self.correction_gain;
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
