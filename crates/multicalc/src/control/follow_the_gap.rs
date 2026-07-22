//! Follow-the-Gap reactive obstacle avoidance over a forward range scan.
#![deny(clippy::indexing_slicing)]

use crate::error::ControlError;
use crate::kinematics::BodyTwist;
use crate::scalar::Numeric;

/// Steers a robot through the widest safe gap it can see in a forward range scan of `BEAMS` beams.
///
/// Each beam points in a fixed direction and reports how far away the nearest obstacle is. The
/// beams fan out evenly across `field_of_view`, from `-field_of_view/2` on the right to
/// `+field_of_view/2` on the left, with straight ahead at zero and angles growing to the left.
///
/// The follower finds stretches of beams that see open space and picks one to drive toward. A gap
/// only counts if the robot fits through it: the two obstacles at its edges must be at least
/// `chassis_width` apart, measured in metres. Metres are what matter here — the same spread of
/// beams can be a real gap up close and too narrow far away.
///
/// A stretch of open beams that runs off the edge of the scan still counts as open: the sensor
/// simply saw nothing out there, so the follower does not invent a wall to stop for.
///
/// It reacts to the latest scan only and keeps no memory. In a dead-end pocket it may flip between
/// two gaps instead of backing out; that is how the method works, not a bug.
///
/// When nothing is wide enough, the result is a full stop with [`FollowTheGapOutput::is_blocked`] set — the
/// follower never spins in place. What to do then, such as reversing, is left to the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FollowTheGap<const BEAMS: usize, T: Numeric> {
    /// How wide an angle the scan covers, in radians.
    field_of_view: T,
    /// How far the sensor can see, in metres.
    maximum_range: T,
    /// The robot's full width, in metres.
    chassis_width: T,
    /// A beam counts as open space when it reads at least this far, in metres.
    free_range_threshold: T,
    /// Forward speed when the way ahead is clear, in metres per second.
    cruise_speed: T,
    /// How sharply the robot turns toward its chosen heading, in 1/s.
    steering_gain: T,
    /// How strongly to favour gaps pointing toward the goal; higher means more, dimensionless.
    goal_bias: T,
    /// With this much space ahead or less, the robot stops, in metres.
    stopping_distance: T,
    /// With this much space ahead or more, the robot drives at full `cruise_speed`, in metres.
    clear_distance: T,
    /// How wide an angle counts as "ahead" when checking space to slow down for, half-angle in radians.
    frontal_half_angle: T,
}

/// The result of one reactive step: the motion command plus context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FollowTheGapOutput<T: Numeric> {
    body_twist: BodyTwist<T>,
    heading: T,
    gap_start_index: usize,
    gap_end_index: usize,
    minimum_clearance: T,
    blocked: bool,
}

impl<T: Numeric> FollowTheGapOutput<T> {
    /// A full stop, used when no gap is wide enough for the robot.
    #[inline]
    fn stopped(minimum_clearance: T) -> Self {
        FollowTheGapOutput {
            body_twist: BodyTwist::new(T::ZERO, T::ZERO),
            heading: T::ZERO,
            gap_start_index: 0,
            gap_end_index: 0,
            minimum_clearance,
            blocked: true,
        }
    }

    /// The commanded forward speed and turn rate. Both are zero when blocked.
    #[inline]
    #[must_use]
    pub fn body_twist(&self) -> BodyTwist<T> {
        self.body_twist
    }

    /// The direction the robot should steer, in radians from straight ahead, positive to the left.
    /// Zero when blocked.
    #[inline]
    #[must_use]
    pub fn heading(&self) -> T {
        self.heading
    }

    /// The first beam index of the selected gap. Zero when blocked.
    #[inline]
    #[must_use]
    pub fn gap_start_index(&self) -> usize {
        self.gap_start_index
    }

    /// The last beam index of the selected gap, inclusive. Zero when blocked.
    #[inline]
    #[must_use]
    pub fn gap_end_index(&self) -> usize {
        self.gap_end_index
    }

    /// The distance to the nearest obstacle anywhere in the scan, in metres.
    #[inline]
    #[must_use]
    pub fn minimum_clearance(&self) -> T {
        self.minimum_clearance
    }

    /// True when no gap was both open and wide enough, so the result is a full stop.
    #[inline]
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.blocked
    }
}

impl<const BEAMS: usize, T: Numeric> FollowTheGap<BEAMS, T> {
    /// A gap-follower over `BEAMS` beams spanning `field_of_view` radians.
    ///
    /// The remaining settings take defaults: `steering_gain` is `1.5`, `goal_bias` is `0.5`,
    /// `stopping_distance` is half the chassis width, `clear_distance` is `maximum_range`, and
    /// `frontal_half_angle` is a quarter of `field_of_view`. The builders override them.
    ///
    /// Returns [`ControlError::InvalidBeamCount`] if `BEAMS` is below two,
    /// [`ControlError::NonFinite`] if any argument is infinite or NaN,
    /// [`ControlError::InvalidFieldOfView`] if `field_of_view` is outside `(0, 2π]`,
    /// [`ControlError::NonPositiveRange`] if `maximum_range` or `free_range_threshold` is not
    /// strictly positive or the threshold exceeds the range, [`ControlError::NonPositiveChassisWidth`] if
    /// `chassis_width` is not strictly positive or half of it reaches `maximum_range`, and
    /// [`ControlError::NonPositiveSpeed`] if `cruise_speed` is not strictly positive.
    pub fn try_new(
        field_of_view: T,
        maximum_range: T,
        chassis_width: T,
        free_range_threshold: T,
        cruise_speed: T,
    ) -> Result<Self, ControlError> {
        if BEAMS < 2 {
            return Err(ControlError::InvalidBeamCount);
        }
        if !field_of_view.is_finite()
            || !maximum_range.is_finite()
            || !chassis_width.is_finite()
            || !free_range_threshold.is_finite()
            || !cruise_speed.is_finite()
        {
            return Err(ControlError::NonFinite);
        }
        if field_of_view <= T::ZERO || field_of_view > T::TWO_PI {
            return Err(ControlError::InvalidFieldOfView);
        }
        if maximum_range <= T::ZERO
            || free_range_threshold <= T::ZERO
            || free_range_threshold > maximum_range
        {
            return Err(ControlError::NonPositiveRange);
        }

        // The default stopping distance is half the chassis width. Keeping that below the maximum
        // range means the speed ramp in `compute` never divides by zero.
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
            free_range_threshold,
            cruise_speed,
            steering_gain: T::from_f64(1.5),
            goal_bias: T::from_f64(0.5),
            stopping_distance: chassis_width * T::HALF,
            clear_distance: maximum_range,
            frontal_half_angle: field_of_view / T::from_f64(4.0),
        })
    }

    /// Sets how sharply the robot turns toward its chosen heading, in 1/s.
    ///
    /// Returns [`ControlError::NonFinite`] if `steering_gain` is not finite, or
    /// [`ControlError::NonPositiveSpeed`] if it is not strictly positive.
    pub fn with_steering_gain(mut self, steering_gain: T) -> Result<Self, ControlError> {
        if !steering_gain.is_finite() {
            return Err(ControlError::NonFinite);
        }
        if steering_gain <= T::ZERO {
            return Err(ControlError::NonPositiveSpeed);
        }
        self.steering_gain = steering_gain;
        Ok(self)
    }

    /// Sets how strongly gaps pointing toward the goal are favoured. Zero just picks the widest gap.
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

    /// Sets the two space-ahead thresholds between which speed ramps from a stop up to `cruise_speed`.
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

    /// Sets how wide an angle counts as "ahead" when measuring the space to slow down for, in radians.
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

    /// The direction beam `index` points, in radians from straight ahead, or `None` if the index is
    /// out of range.
    #[inline]
    #[must_use]
    pub fn beam_angle(&self, index: usize) -> Option<T> {
        (index < BEAMS).then(|| self.beam_angle_unchecked(index))
    }

    /// Works out a speed and turn command from one range scan.
    ///
    /// `beam_ranges` holds one distance per beam in metres, ordered from the beam at
    /// `-field_of_view/2` to the beam at `+field_of_view/2`. `goal_angle` is the direction you want
    /// to head, measured from straight ahead, with `0` meaning straight ahead.
    ///
    /// A beam that is invalid or zero-or-negative is treated as empty space at `maximum_range`: a
    /// missed reading means the sensor saw nothing there, not that something is close.
    ///
    /// The turn command is `steering_gain × heading`. The speed ramps smoothly from a stop when the
    /// space ahead is `stopping_distance` up to `cruise_speed` when it reaches `clear_distance`,
    /// judged only from the beams pointing forward.
    ///
    /// Returns [`ControlError::NonFinite`] if `goal_angle` is not finite.
    ///
    /// ```
    /// use multicalc::control::FollowTheGap;
    ///
    /// // 31 beams over 120°, 4 m range, a 0.5 m robot, 0.5 m open-space threshold, 0.4 m/s cruise.
    /// let follower: FollowTheGap<31, f64> =
    ///     FollowTheGap::try_new(2.0 * core::f64::consts::PI / 3.0, 4.0, 0.5, 0.5, 0.4).unwrap();
    ///
    /// // Nothing in the way: drive straight ahead at cruise speed.
    /// let output = follower.compute(&[4.0; 31], 0.0).unwrap();
    /// assert!(output.heading().abs() < 1e-12);
    /// assert!((output.body_twist().linear() - 0.4).abs() < 1e-12);
    ///
    /// // A wall all round: stop, and say so.
    /// let blocked = follower.compute(&[0.2; 31], 0.0).unwrap();
    /// assert!(blocked.is_blocked());
    /// assert_eq!(blocked.body_twist().linear(), 0.0);
    /// ```
    pub fn compute(
        &self,
        beam_ranges: &[T; BEAMS],
        goal_angle: T,
    ) -> Result<FollowTheGapOutput<T>, ControlError> {
        if !goal_angle.is_finite() {
            return Err(ControlError::NonFinite);
        }

        // Treat invalid or zero-or-negative readings as empty space at `maximum_range`.
        let sanitized_ranges: [T; BEAMS] =
            core::array::from_fn(|index| match beam_ranges.get(index) {
                Some(&range) if range.is_finite() && range > T::ZERO => {
                    range.min(self.maximum_range)
                }
                _ => self.maximum_range,
            });

        // Track the nearest obstacle across the whole scan, reported in the output.
        let mut minimum_clearance = self.maximum_range;
        for &range in &sanitized_ranges {
            if range < minimum_clearance {
                minimum_clearance = range;
            }
        }

        // Sweep across the beams once. Each time a stretch of open beams ends, score it. The extra
        // step past the last beam reads nothing and counts as blocked, which closes off a stretch
        // still open at the very edge of the scan.
        let mut best_score = T::NEG_INFINITY;
        let mut best_gap: Option<(usize, usize, T)> = None;
        let mut run_start: Option<usize> = None;

        for index in 0..=BEAMS {
            let is_free = match sanitized_ranges.get(index) {
                Some(&range) => range >= self.free_range_threshold,
                None => false,
            };
            match (is_free, run_start) {
                // A free beam with nothing open yet begins a new stretch.
                (true, None) => run_start = Some(index),
                (false, Some(start)) => {
                    // Safe: we only reach here with a stretch that started at an earlier beam.
                    let end = index - 1;
                    run_start = None; // Reset for the next stretch.

                    // Drop this stretch if it is too narrow for the robot to fit through.
                    if self
                        .gap_width(&sanitized_ranges, start, end)
                        .is_some_and(|width| width < self.chassis_width)
                    {
                        continue;
                    }

                    // Choose where to aim inside this gap.
                    let (low, high) = self.aim_bounds(&sanitized_ranges, start, end);
                    let aim = if low > high {
                        // The safety margins from both edges overlap, so aim at the middle — the
                        // spot farthest from either obstacle.
                        (self.beam_angle_unchecked(start) + self.beam_angle_unchecked(end))
                            * T::HALF
                    } else {
                        // Head for the goal, pulled just inside the safe edges.
                        goal_angle.max(low).min(high)
                    };
                    // Reward a wider gap, penalise an aim that points away from the goal.
                    let score =
                        (high - low).max(T::ZERO) - self.goal_bias * (aim - goal_angle).abs();

                    // Ties are common because beam angles come in fixed steps. Breaking a tie
                    // toward the straighter aim gives the same result whichever way we scan; a
                    // perfectly symmetric tie goes to the earlier beam.
                    let wins = match best_gap {
                        None => true,
                        Some((_, _, best_aim)) => {
                            score > best_score
                                || (score == best_score && aim.abs() < best_aim.abs())
                        }
                    };
                    if wins {
                        best_score = score;
                        best_gap = Some((start, end, aim));
                    }
                }
                _ => {}
            }
        }

        // Take the highest-scoring gap, or stop if none was wide enough.
        let (gap_start_index, gap_end_index, heading) = match best_gap {
            Some(gap) => gap,
            None => return Ok(FollowTheGapOutput::stopped(minimum_clearance)),
        };

        // Speed depends only on the beams pointing forward, so something off to the side does not
        // slow the robot until it comes around to the front.
        let mut frontal_clearance = self.maximum_range;
        for (index, &range) in sanitized_ranges.iter().enumerate() {
            if self.beam_angle_unchecked(index).abs() <= self.frontal_half_angle
                && range < frontal_clearance
            {
                frontal_clearance = range;
            }
        }

        // Both `try_new` and `with_speed_scaling` keep `stopping_distance` below `clear_distance`,
        // so this is always above zero.
        let span = self.clear_distance - self.stopping_distance;
        let speed_scale = ((frontal_clearance - self.stopping_distance) / span)
            .max(T::ZERO)
            .min(T::ONE);

        // Turn toward the chosen aim, and drive at the speed the room ahead allows.
        Ok(FollowTheGapOutput {
            body_twist: BodyTwist::new(
                self.cruise_speed * speed_scale,
                self.steering_gain * heading,
            ),
            heading,
            gap_start_index,
            gap_end_index,
            minimum_clearance,
            blocked: false,
        })
    }

    /// The angle for a beam whose index is already known to be in range.
    #[inline]
    fn beam_angle_unchecked(&self, index: usize) -> T {
        // `try_new` is the only way to build this and rejects fewer than two beams, so `BEAMS - 1`
        // never underflows.
        let span = T::from_usize(BEAMS - 1);
        -self.field_of_view * T::HALF + self.field_of_view * T::from_usize(index) / span
    }

    /// The straight-line distance in metres between the two obstacles on either side of the open
    /// run `[start, end]`, or `None` if the run reaches the edge of the scan and has no obstacle on
    /// that side.
    ///
    /// The two obstacles and the sensor form a triangle: we know both nearby sides (the obstacle
    /// ranges) and the angle between them, and the third side is the gap. Measuring it in metres is
    /// what matters — the same spread of beams is a wide gap at four metres but a tight one at forty
    /// centimetres.
    fn gap_width(&self, beam_ranges: &[T; BEAMS], start: usize, end: usize) -> Option<T> {
        let before = start.checked_sub(1)?;
        let after = end.checked_add(1).filter(|&index| index < BEAMS)?;
        let range_a = *beam_ranges.get(before)?;
        let range_b = *beam_ranges.get(after)?;
        let separation = self.beam_angle_unchecked(after) - self.beam_angle_unchecked(before);
        let squared =
            range_a * range_a + range_b * range_b - T::TWO * range_a * range_b * separation.cos();
        // A very thin triangle can round to a tiny negative here, which would make `sqrt` NaN.
        Some(squared.max(T::ZERO).sqrt())
    }

    /// The range of directions the robot may aim within the open run `[start, end]`. Each edge that
    /// has an obstacle is pulled inward just enough to keep the robot's half-width clear of it at
    /// that obstacle's distance. In other implementations this is achieved using a safety multiplier
    /// added to the minimum required gap.
    ///
    /// An edge that runs off the scan has no obstacle, so it gets no inward margin, matching
    /// [`Self::gap_width`]. If the two margins overlap, the low bound ends up above the high one;
    /// the caller handles that case.
    fn aim_bounds(&self, beam_ranges: &[T; BEAMS], start: usize, end: usize) -> (T, T) {
        let half_width = self.chassis_width * T::HALF;
        let inset = |index: usize| match beam_ranges.get(index) {
            Some(&range) if range > T::ZERO => (half_width / range).atan(),
            _ => T::ZERO,
        };
        let low = self.beam_angle_unchecked(start) + start.checked_sub(1).map_or(T::ZERO, inset);
        let high = self.beam_angle_unchecked(end)
            - end
                .checked_add(1)
                .filter(|&index| index < BEAMS)
                .map_or(T::ZERO, inset);
        (low, high)
    }
}
