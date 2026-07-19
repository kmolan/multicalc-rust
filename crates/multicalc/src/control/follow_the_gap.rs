//! Follow-the-Gap reactive obstacle avoidance over a forward range scan.
#![deny(clippy::indexing_slicing)]

use crate::error::ControlError;
use crate::kinematics::BodyTwist;
use crate::scalar::Numeric;

/// A reactive gap-follower over a forward range scan of `BEAMS` beams.
///
/// Beam bearings are measured from the forward (+x) axis in the robot body frame, positive
/// counter-clockwise, and the beams are spaced uniformly across `field_of_view`. The scan runs from
/// the beam at `-field_of_view/2` to the beam at `+field_of_view/2`.
///
/// Gaps are measured in metres, not beams. A run of free beams is only usable if the two returns
/// bounding it are at least `chassis_width` apart, and the aim inside it is held off each bounded
/// edge by the angle the robot's half-width subtends at that edge's range. The same angular gap is
/// therefore passable at four metres and impassable at forty centimetres.
///
/// A run that reaches either end of the field of view has no bounding return on that side and
/// counts as open. The sensor saw nothing out there, and inventing a wall would stop the robot on
/// no evidence.
///
/// The method is purely reactive: it plans from the current scan alone and keeps no state. In a
/// three-sided concave pocket it can dither between two gaps rather than back out, which is a
/// property of the algorithm and not a defect.
///
/// A scan offering no gap the robot fits through yields a stopped plan with
/// [`GapPlan::is_blocked`] set, never a rotation. The recovery policy is left to the caller, so the
/// follower never commands a heading it did not derive from the scan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FollowTheGap<const BEAMS: usize, T: Numeric> {
    /// Total angular span of the scan, in radians.
    field_of_view: T,
    /// Sensor maximum range, in metres.
    maximum_range: T,
    /// Full width of the robot, in metres.
    chassis_width: T,
    /// A beam is free when its range is at least this, in metres.
    gap_threshold: T,
    /// Forward speed with clear space ahead, in metres per second.
    cruise_speed: T,
    /// Yaw rate per radian of heading error, in 1/s.
    turn_gain: T,
    /// Weight on goal alignment in gap scoring, dimensionless.
    goal_bias: T,
    /// Frontal clearance at or below which forward speed is zero, in metres.
    stopping_distance: T,
    /// Frontal clearance at or above which forward speed is `cruise_speed`, in metres.
    clear_distance: T,
    /// Half-width of the arc that counts as frontal, in radians.
    frontal_half_angle: T,
}

impl<const BEAMS: usize, T: Numeric> FollowTheGap<BEAMS, T> {
    /// A gap-follower over `BEAMS` beams spanning `field_of_view` radians.
    ///
    /// The remaining settings take defaults: `turn_gain` is `1.5`, `goal_bias` is `0.5`,
    /// `stopping_distance` is half the chassis width, `clear_distance` is `maximum_range`, and
    /// `frontal_half_angle` is a quarter of `field_of_view`. The builders override them.
    ///
    /// Returns [`ControlError::InvalidBeamCount`] if `BEAMS` is below two,
    /// [`ControlError::NonFinite`] if any argument is infinite or NaN,
    /// [`ControlError::InvalidFieldOfView`] if `field_of_view` is outside `(0, 2π]`,
    /// [`ControlError::NonPositiveRange`] if `maximum_range` or `gap_threshold` is not strictly
    /// positive or the threshold exceeds the range, [`ControlError::NonPositiveChassisWidth`] if
    /// `chassis_width` is not strictly positive or half of it reaches `maximum_range`, and
    /// [`ControlError::NonPositiveSpeed`] if `cruise_speed` is not strictly positive.
    pub fn try_new(
        field_of_view: T,
        maximum_range: T,
        chassis_width: T,
        gap_threshold: T,
        cruise_speed: T,
    ) -> Result<Self, ControlError> {
        if BEAMS < 2 {
            return Err(ControlError::InvalidBeamCount);
        }
        if !field_of_view.is_finite()
            || !maximum_range.is_finite()
            || !chassis_width.is_finite()
            || !gap_threshold.is_finite()
            || !cruise_speed.is_finite()
        {
            return Err(ControlError::NonFinite);
        }
        if field_of_view <= T::ZERO || field_of_view > T::TWO_PI {
            return Err(ControlError::InvalidFieldOfView);
        }
        if maximum_range <= T::ZERO || gap_threshold <= T::ZERO || gap_threshold > maximum_range {
            return Err(ControlError::NonPositiveRange);
        }
        // Half the chassis is the default stopping distance, so bounding it below the maximum range
        // keeps the speed-scaling span in `plan` strictly positive.
        if chassis_width <= T::ZERO || chassis_width * T::HALF >= maximum_range {
            return Err(ControlError::NonPositiveChassisWidth);
        }
        if cruise_speed <= T::ZERO {
            return Err(ControlError::NonPositiveSpeed);
        }
        Ok(Self {
            field_of_view,
            maximum_range,
            chassis_width,
            gap_threshold,
            cruise_speed,
            turn_gain: T::from_f64(1.5),
            goal_bias: T::from_f64(0.5),
            stopping_distance: chassis_width * T::HALF,
            clear_distance: maximum_range,
            frontal_half_angle: field_of_view / T::from_f64(4.0),
        })
    }

    /// Sets the yaw rate commanded per radian of heading error, in 1/s.
    ///
    /// Returns [`ControlError::NonFinite`] if `turn_gain` is not finite, or
    /// [`ControlError::NonPositiveSpeed`] if it is not strictly positive.
    pub fn with_turn_gain(mut self, turn_gain: T) -> Result<Self, ControlError> {
        if !turn_gain.is_finite() {
            return Err(ControlError::NonFinite);
        }
        if turn_gain <= T::ZERO {
            return Err(ControlError::NonPositiveSpeed);
        }
        self.turn_gain = turn_gain;
        Ok(self)
    }

    /// Sets the weight on goal alignment when scoring gaps. Zero picks the widest gap outright.
    ///
    /// Returns [`ControlError::NonFinite`] if `goal_bias` is not finite, or
    /// [`ControlError::NegativeGoalBias`] if it is negative.
    pub fn with_goal_bias(mut self, goal_bias: T) -> Result<Self, ControlError> {
        if !goal_bias.is_finite() {
            return Err(ControlError::NonFinite);
        }
        if goal_bias < T::ZERO {
            return Err(ControlError::NegativeGoalBias);
        }
        self.goal_bias = goal_bias;
        Ok(self)
    }

    /// Sets the frontal clearances between which forward speed ramps from zero to `cruise_speed`.
    ///
    /// Returns [`ControlError::NonFinite`] if either argument is not finite, or
    /// [`ControlError::InvalidSpeedScaling`] if `stopping_distance` is negative or is not strictly
    /// less than `clear_distance`.
    pub fn with_speed_scaling(
        mut self,
        stopping_distance: T,
        clear_distance: T,
    ) -> Result<Self, ControlError> {
        if !stopping_distance.is_finite() || !clear_distance.is_finite() {
            return Err(ControlError::NonFinite);
        }
        if stopping_distance < T::ZERO || stopping_distance >= clear_distance {
            return Err(ControlError::InvalidSpeedScaling);
        }
        self.stopping_distance = stopping_distance;
        self.clear_distance = clear_distance;
        Ok(self)
    }

    /// Sets the half-width of the arc whose beams count as frontal, in radians.
    ///
    /// Returns [`ControlError::NonFinite`] if `frontal_half_angle` is not finite, or
    /// [`ControlError::InvalidFieldOfView`] if it is not strictly positive or exceeds half the
    /// field of view.
    pub fn with_frontal_half_angle(mut self, frontal_half_angle: T) -> Result<Self, ControlError> {
        if !frontal_half_angle.is_finite() {
            return Err(ControlError::NonFinite);
        }
        if frontal_half_angle <= T::ZERO || frontal_half_angle > self.field_of_view * T::HALF {
            return Err(ControlError::InvalidFieldOfView);
        }
        self.frontal_half_angle = frontal_half_angle;
        Ok(self)
    }

    /// The body-frame bearing of beam `index`, in radians, or `None` if the index is out of range.
    #[inline]
    #[must_use]
    pub fn beam_bearing(&self, index: usize) -> Option<T> {
        (index < BEAMS).then(|| self.beam_bearing_unchecked(index))
    }

    /// The bearing formula, for callers that have already bounded the index.
    #[inline]
    fn beam_bearing_unchecked(&self, index: usize) -> T {
        // `try_new` rejects fewer than two beams and is the only constructor, so this cannot wrap.
        let span = T::from_usize(BEAMS - 1);
        -self.field_of_view * T::HALF + self.field_of_view * T::from_usize(index) / span
    }
}
