//! Ready-made process and measurement models for the filters.
//!
//! - [`ConstantTurnAndSpeed`] — a vehicle rolling along the arc it is turning through.
//! - [`DirectMeasurement`] — a sensor reading part of the state straight off.
//! - [`residual_with_wrapped_angles`] — a reading minus its prediction, angles folded into range.

use crate::error::EstimationError;
use crate::linear_algebra::Vector;
use crate::scalar::{Numeric, VectorFn};

/// Rolls a vehicle's state `[x, y, heading, speed, turn rate]` forward one tick.
///
/// The speed and turn rate are taken to hold, so the vehicle traces an arc; the filter learns
/// otherwise from its measurements. Tracking work calls this the coordinated-turn model.
///
/// ```
/// use multicalc::estimation::ConstantTurnAndSpeed;
/// use multicalc::scalar::VectorFn;
///
/// let timestep = 0.1;
/// let speed = 2.0;
/// let motion = ConstantTurnAndSpeed { timestep };
///
/// // Facing along x and not turning: it covers speed times the tick, straight ahead.
/// let straight = [0.0, 0.0, 0.0, speed, 0.0];
/// let moved = motion.eval(&straight);
/// assert!((moved[0] - speed * timestep).abs() < 1e-12);
///
/// // Turning left: the heading moves on and the path curves upward.
/// let turn_rate = 1.0;
/// let turning = [0.0, 0.0, 0.0, speed, turn_rate];
/// let swung = motion.eval(&turning);
/// assert!((swung[2] - turn_rate * timestep).abs() < 1e-12);
/// assert!(swung[1] > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantTurnAndSpeed {
    /// How long one tick lasts.
    pub timestep: f64,
}

impl VectorFn<5, 5> for ConstantTurnAndSpeed {
    fn eval<S: Numeric>(&self, state: &[S; 5]) -> [S; 5] {
        let [x, y, heading, speed, turn_rate] = *state;
        let timestep = S::from_f64(self.timestep);
        let next_heading = heading + turn_rate * timestep;
        // Straighten the arc when the turn rate is tiny, so the radius cannot blow up.
        let (next_x, next_y) = if turn_rate.abs() > S::from_f64(1e-6) {
            let radius = speed / turn_rate;
            (
                x + radius * (next_heading.sin() - heading.sin()),
                y + radius * (heading.cos() - next_heading.cos()),
            )
        } else {
            (
                x + speed * heading.cos() * timestep,
                y + speed * heading.sin() * timestep,
            )
        };
        [next_x, next_y, next_heading.wrap_to_pi(), speed, turn_rate]
    }
}

/// A sensor reading state components straight off, such as a position fix reading `[x, y]`.
///
/// The components come back in the order they are listed.
///
/// ```
/// use multicalc::estimation::DirectMeasurement;
/// use multicalc::scalar::VectorFn;
///
/// // A state of [x, y, heading, speed, turn rate].
/// let state = [3.0, 4.0, 0.5, 2.0, 0.1];
///
/// let position_fix = DirectMeasurement::<5, 2>::try_new([0, 1])?;
/// assert_eq!(position_fix.eval(&state), [3.0, 4.0]);
///
/// let encoders = DirectMeasurement::<5, 2>::try_new([3, 4])?;
/// assert_eq!(encoders.eval(&state), [2.0, 0.1]);
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectMeasurement<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize> {
    indices: [usize; MEASUREMENT_DIMENSION],
}

impl<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize>
    DirectMeasurement<STATE_DIMENSION, MEASUREMENT_DIMENSION>
{
    /// A model reading the listed state components, in the order given.
    ///
    /// Returns [`EstimationError::StateIndexOutOfRange`] if any index is past the end of the state.
    ///
    /// ```
    /// use multicalc::error::EstimationError;
    /// use multicalc::estimation::DirectMeasurement;
    ///
    /// // A five-component state has nothing at index five.
    /// assert_eq!(
    ///     DirectMeasurement::<5, 2>::try_new([0, 5]).unwrap_err(),
    ///     EstimationError::StateIndexOutOfRange
    /// );
    /// ```
    pub fn try_new(indices: [usize; MEASUREMENT_DIMENSION]) -> Result<Self, EstimationError> {
        for index in indices {
            if index >= STATE_DIMENSION {
                return Err(EstimationError::StateIndexOutOfRange);
            }
        }
        Ok(DirectMeasurement { indices })
    }

    /// The state components this model reads.
    #[must_use]
    pub fn indices(&self) -> [usize; MEASUREMENT_DIMENSION] {
        self.indices
    }
}

impl<const STATE_DIMENSION: usize, const MEASUREMENT_DIMENSION: usize>
    VectorFn<STATE_DIMENSION, MEASUREMENT_DIMENSION>
    for DirectMeasurement<STATE_DIMENSION, MEASUREMENT_DIMENSION>
{
    fn eval<S: Numeric>(&self, state: &[S; STATE_DIMENSION]) -> [S; MEASUREMENT_DIMENSION] {
        // `try_new` rejects an index past the end, so the fallback never runs.
        core::array::from_fn(|position| {
            self.indices
                .get(position)
                .and_then(|&index| state.get(index))
                .copied()
                .unwrap_or(S::ZERO)
        })
    }
}

/// A reading minus its prediction, with the components listed in `angular_components` folded back
/// into (-π, π].
///
/// Without folding, a heading either side of the half turn reads as nearly a whole turn of error,
/// and the filter lurches to correct it.
///
/// ```
/// use multicalc::estimation::residual_with_wrapped_angles;
/// use multicalc::linear_algebra::Vector;
///
/// // A heading of 3 rad against one of -3 rad: a short step apart, not six radians.
/// let measured = Vector::new([3.0_f64, 10.0]);
/// let predicted = Vector::new([-3.0, 4.0]);
/// let heading_only = [0];
/// let residual = residual_with_wrapped_angles(measured, predicted, &heading_only);
/// assert!(residual[0].abs() < 0.3);
///
/// // The second component is no angle, so it is a plain subtraction.
/// assert!((residual[1] - 6.0).abs() < 1e-12);
/// ```
pub fn residual_with_wrapped_angles<const MEASUREMENT_DIMENSION: usize, T: Numeric>(
    measured: Vector<MEASUREMENT_DIMENSION, T>,
    predicted: Vector<MEASUREMENT_DIMENSION, T>,
    angular_components: &[usize],
) -> Vector<MEASUREMENT_DIMENSION, T> {
    Vector::from_fn(|component| {
        let difference = measured[component] - predicted[component];
        if angular_components.contains(&component) {
            difference.wrap_to_pi()
        } else {
            difference
        }
    })
}
