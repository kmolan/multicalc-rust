//! Pure-pursuit path-following law.

use crate::error::ControlError;
use crate::kinematics::BodyTwist;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;
use crate::spatial::SE2;

/// A path curvature, the reciprocal of the turning radius, in units of 1/m. Positive curves left.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Curvature<T: Numeric> {
    value: T,
}

impl<T: Numeric> Curvature<T> {
    /// Wraps a curvature value.
    #[inline]
    #[must_use]
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// The curvature value in 1/m.
    #[inline]
    #[must_use]
    pub fn value(self) -> T {
        self.value
    }

    /// The body twist that follows this curvature at a given forward speed: linear speed forward and
    /// angular rate `forward_speed * curvature`.
    #[inline]
    #[must_use]
    pub fn to_body_twist(self, forward_speed: T) -> BodyTwist<T> {
        BodyTwist::new(forward_speed, forward_speed * self.value)
    }
}

/// The pure-pursuit curvature that steers a robot toward a lookahead point.
///
/// `pose` is the robot pose in the world frame and `lookahead_point` is the target in the world
/// frame. `lookahead_distance` is the pursuit distance `L_d`. The result is the exact
/// `κ = 2·sin(α)/L_d` written in body-frame coordinates, which holds when `L_d` equals the distance
/// from the robot to the target point.
///
/// Returns [`ControlError::NonPositiveLookaheadDistance`] if `lookahead_distance` is not finite or not
/// strictly positive, or [`ControlError::NonFinite`] if the lookahead point is not finite.
///
/// ```
/// use multicalc::control::pure_pursuit_curvature;
/// use multicalc::spatial::SE2;
/// use multicalc::linear_algebra::Vector;
///
/// let pose = SE2::identity();
/// // A point straight ahead needs no turn.
/// let ahead = pure_pursuit_curvature(pose, Vector::new([2.0_f64, 0.0]), 2.0).unwrap();
/// assert!(ahead.value().abs() < 1e-12);
/// // A point to the left curves left (positive curvature).
/// let left = pure_pursuit_curvature(pose, Vector::new([2.0_f64, 1.0]), 2.0).unwrap();
/// assert!(left.value() > 0.0);
/// ```
pub fn pure_pursuit_curvature<T: Numeric>(
    pose: SE2<T>,
    lookahead_point: Vector<2, T>,
    lookahead_distance: T,
) -> Result<Curvature<T>, ControlError> {
    if !lookahead_distance.is_finite() || lookahead_distance <= T::ZERO {
        return Err(ControlError::NonPositiveLookaheadDistance);
    }
    if !lookahead_point.is_finite() {
        return Err(ControlError::NonFinite);
    }
    let [_forward, lateral] = pose.inverse().act(lookahead_point).into_array();
    let curvature = (T::TWO * lateral) / (lookahead_distance * lookahead_distance);
    Ok(Curvature::new(curvature))
}
