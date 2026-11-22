//! The endpoint measurement model of *Probabilistic Robotics* §6.4.

use crate::scalar::Numeric;

/// Scores a scan by how close its endpoints fall to mapped obstacles.
///
/// Each beam contributes
/// `ln((1 − w)·exp(−d² / 2σ²) + w / maximum_range)`, for `d` the endpoint's distance to the nearest
/// obstacle, `σ` the [`measurement_deviation`](Self::measurement_deviation) and `w` the
/// [`random_measurement_weight`](Self::random_measurement_weight).
///
/// It replaces one ray cast per beam per particle with one interpolated field lookup, and is
/// smoother in the pose than a beam model, whose likelihood is jagged because it depends on map
/// resolution.
///
/// It ignores occlusion, so a pose can score highly by seeing through a wall. Keep
/// [`BeamModel`](crate::estimation::BeamModel) where that matters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LikelihoodFieldModel<T: Numeric = f64> {
    /// Spread of the Gaussian on an endpoint's distance to the nearest obstacle.
    pub measurement_deviation: T,
    /// Mixture weight for a reading that is pure noise, in zero to one.
    pub random_measurement_weight: T,
}

impl<T: Numeric> Default for LikelihoodFieldModel<T> {
    fn default() -> Self {
        LikelihoodFieldModel {
            measurement_deviation: T::from_f64(0.20),
            random_measurement_weight: T::from_f64(0.05),
        }
    }
}
