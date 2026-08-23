//! Point-to-point motion profiles
#![deny(clippy::indexing_slicing)]

use crate::error::MotionError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// Phases in every profile. A trapezoid is the same seven with the four jerk phases at zero
/// duration.
pub const PROFILE_PHASE_COUNT: usize = 7;

/// Kinematic limits: velocity, acceleration, and optionally jerk.
///
/// A jerk limit selects the seven-phase S-curve; without one the profile is trapezoidal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileLimits<T: Numeric = f64> {
    speed: T,
    acceleration: T,
    jerk: Option<T>,
}

impl<T: Numeric> ProfileLimits<T> {
    /// Validated limits, with jerk optional.
    ///
    /// Returns [`MotionError::LimitNotPositive`] unless every limit supplied is finite and strictly
    /// positive.
    pub fn try_new(speed: T, acceleration: T, jerk: Option<T>) -> Result<Self, MotionError> {
        let positive = |value: T| value.is_finite() && value > T::ZERO;
        if !positive(speed) || !positive(acceleration) {
            return Err(MotionError::LimitNotPositive);
        }
        if let Some(jerk_limit) = jerk {
            if !positive(jerk_limit) {
                return Err(MotionError::LimitNotPositive);
            }
        }
        Ok(Self {
            speed,
            acceleration,
            jerk,
        })
    }

    /// Velocity limit.
    #[inline]
    #[must_use]
    pub fn speed(&self) -> T {
        self.speed
    }

    /// Acceleration limit.
    #[inline]
    #[must_use]
    pub fn acceleration(&self) -> T {
        self.acceleration
    }

    /// Jerk limit, if the profile is bounded in jerk.
    #[inline]
    #[must_use]
    pub fn jerk(&self) -> Option<T> {
        self.jerk
    }
}

/// Profile shape to plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProfileStrategy {
    /// Read the shape off the limits: S-curve with a jerk limit, trapezoidal without.
    #[default]
    Automatic,
    /// Three phases: accelerate, cruise, decelerate. Any jerk limit is ignored.
    Trapezoidal,
    /// Seven phases with bounded jerk. Requires a jerk limit.
    JerkLimited,
}

/// Single-axis kinematic state at one instant.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ProfileState<T: Numeric = f64> {
    /// Position.
    pub position: T,
    /// Velocity.
    pub velocity: T,
    /// Acceleration.
    pub acceleration: T,
    /// Jerk.
    pub jerk: T,
}

/// One constant-jerk phase and the state it starts from.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct ProfilePhase<T: Numeric> {
    duration: T,
    jerk: T,
    start_position: T,
    start_velocity: T,
    start_acceleration: T,
}

impl<T: Numeric> ProfilePhase<T> {
    /// A zero-duration phase at rest.
    #[inline]
    #[must_use]
    fn empty() -> Self {
        Self {
            duration: T::ZERO,
            jerk: T::ZERO,
            start_position: T::ZERO,
            start_velocity: T::ZERO,
            start_acceleration: T::ZERO,
        }
    }
}

/// A time-optimal rest-to-rest move within its limits, evaluable at any time.
///
/// Planning is one-off; [`state_at`](MotionProfile::state_at) is the per-tick cost, bounded at
/// seven phase steps and one cubic evaluation with no allocation, so it is real-time safe.
///
/// A negative displacement mirrors a positive one exactly.
///
/// Under a `Dual` scalar the derivative follows the branch the solve took, so it is one-sided at a
/// branch boundary, where the profile is not differentiable.
///
/// `MotionProfile<f64>` is 296 bytes: seven phases of five values, plus duration and displacement.
/// Half that at `f32`.
///
/// ```
/// use multicalc::motion::{MotionProfilePlanner, ProfileLimits};
///
/// // 10 m, v_max 2 m/s, a_max 1 m/s².
/// let limits = ProfileLimits::<f64>::try_new(2.0, 1.0, None).unwrap();
/// let profile = MotionProfilePlanner::new(limits).plan(10.0).unwrap();
///
/// // 2 s accelerating, 3 s cruising, 2 s decelerating.
/// assert!((profile.duration() - 7.0).abs() < 1e-12);
///
/// // Mid-cruise, at the velocity limit.
/// let state = profile.state_at(3.5).unwrap();
/// assert!((state.velocity - 2.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionProfile<T: Numeric = f64> {
    phases: [ProfilePhase<T>; PROFILE_PHASE_COUNT],
    duration: T,
    distance: T,
}

impl<T: Numeric> MotionProfile<T> {
    /// Integrates a phase schedule forward from rest into stored phases.
    ///
    /// `accelerations` holds each phase's initial acceleration: the source of the trapezoid's step
    /// changes, and redundant with integrating `jerks` for the S-curve. `sign` is `-1` for a move in
    /// the opposite direction, negating every jerk, velocity and acceleration.
    #[must_use]
    fn from_phase_plan(
        durations: [T; PROFILE_PHASE_COUNT],
        jerks: [T; PROFILE_PHASE_COUNT],
        accelerations: [T; PROFILE_PHASE_COUNT],
        sign: T,
    ) -> Self {
        let mut phases = [ProfilePhase::empty(); PROFILE_PHASE_COUNT];
        let mut total_duration = T::ZERO;
        let mut position = T::ZERO;
        let mut velocity = T::ZERO;

        let plan = durations.iter().zip(jerks.iter()).zip(accelerations.iter());
        for (((duration, jerk), acceleration), phase) in plan.zip(phases.iter_mut()) {
            let start_acceleration = *acceleration * sign;
            let signed_jerk = *jerk * sign;
            *phase = ProfilePhase {
                duration: *duration,
                jerk: signed_jerk,
                start_position: position,
                start_velocity: velocity,
                start_acceleration,
            };

            let elapsed = *duration;
            position = position
                + velocity * elapsed
                + start_acceleration * elapsed * elapsed * T::HALF
                + signed_jerk * elapsed * elapsed * elapsed / T::from_usize(6);
            velocity =
                velocity + start_acceleration * elapsed + signed_jerk * elapsed * elapsed * T::HALF;
            total_duration += elapsed;
        }

        Self {
            phases,
            duration: total_duration,
            distance: position,
        }
    }

    /// Total move time.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> T {
        self.duration
    }

    /// Signed displacement.
    #[inline]
    #[must_use]
    pub fn distance(&self) -> T {
        self.distance
    }

    /// Cruise velocity, signed. Zero for a move of no displacement.
    #[must_use]
    pub fn peak_speed(&self) -> T {
        self.phases
            .get(3)
            .map_or(T::ZERO, |cruise| cruise.start_velocity)
    }

    /// Phase durations in order: jerk up, accelerate, jerk down, cruise, jerk down, decelerate,
    /// jerk up.
    #[must_use]
    pub fn phase_durations(&self) -> [T; PROFILE_PHASE_COUNT] {
        core::array::from_fn(|index| {
            self.phases
                .get(index)
                .map_or(T::ZERO, |phase| phase.duration)
        })
    }

    /// The same profile time-scaled to a longer duration.
    ///
    /// For `s = duration / self.duration`, phase durations scale by `s`, velocities by `1/s`,
    /// accelerations by `1/s²` and jerks by `1/s³`. The spatial path is unchanged and no limit that
    /// held can break. A duration that is not finite, not positive, or not longer than the current
    /// one returns `*self`.
    ///
    /// Time-scaling is not a re-solve: the result is not time-optimal for the new duration, which
    /// would take a lower peak velocity and a different phase split, and a search to find.
    #[must_use]
    pub fn stretched_to(&self, duration: T) -> Self {
        if !duration.is_finite() || duration <= self.duration || self.duration <= T::ZERO {
            return *self;
        }

        let scale = duration / self.duration;
        let mut stretched = *self;
        for phase in stretched.phases.iter_mut() {
            phase.duration *= scale;
            phase.start_velocity /= scale;
            phase.start_acceleration /= scale * scale;
            phase.jerk /= scale * scale * scale;
        }
        stretched.duration = duration;
        stretched
    }

    /// State at `time`, clamped to `[0, duration]`.
    ///
    /// Evaluation is right-continuous, so at a step in acceleration it reads the commanded value,
    /// not the one being left: a trapezoid reports full acceleration at `t = 0`. Returns
    /// [`MotionError::NonFinite`] for a time that is not finite.
    pub fn state_at(&self, time: T) -> Result<ProfileState<T>, MotionError> {
        if !time.is_finite() {
            return Err(MotionError::NonFinite);
        }
        let clamped = time.max(T::ZERO).min(self.duration);

        let mut phase_start = T::ZERO;
        for phase in self.phases.iter() {
            let phase_end = phase_start + phase.duration;
            if phase_end > clamped {
                let elapsed = clamped - phase_start;
                return Ok(ProfileState {
                    position: phase.start_position
                        + phase.start_velocity * elapsed
                        + phase.start_acceleration * elapsed * elapsed * T::HALF
                        + phase.jerk * elapsed * elapsed * elapsed / T::from_usize(6),
                    velocity: phase.start_velocity
                        + phase.start_acceleration * elapsed
                        + phase.jerk * elapsed * elapsed * T::HALF,
                    acceleration: phase.start_acceleration + phase.jerk * elapsed,
                    jerk: phase.jerk,
                });
            }
            phase_start = phase_end;
        }

        // Past the final phase: on target, at rest.
        Ok(ProfileState {
            position: self.distance,
            velocity: T::ZERO,
            acceleration: T::ZERO,
            jerk: T::ZERO,
        })
    }
}

/// Zero-displacement profile: zero duration, every phase empty.
#[must_use]
fn plan_still<T: Numeric>(sign: T) -> MotionProfile<T> {
    MotionProfile::from_phase_plan(
        [T::ZERO; PROFILE_PHASE_COUNT],
        [T::ZERO; PROFILE_PHASE_COUNT],
        [T::ZERO; PROFILE_PHASE_COUNT],
        sign,
    )
}

/// Three-phase solve at constant acceleration.
///
/// `distance` is non-negative; `sign` carries the direction.
#[must_use]
fn plan_trapezoidal<T: Numeric>(
    distance: T,
    limits: &ProfileLimits<T>,
    sign: T,
) -> MotionProfile<T> {
    let acceleration_limit = limits.acceleration();

    // Peak velocity the displacement allows, from d = v²/a_max in the triangular case.
    let speed_reachable = (distance * acceleration_limit).sqrt();
    let peak_speed = limits.speed().min(speed_reachable);
    if peak_speed <= T::ZERO {
        return plan_still(sign);
    }

    let acceleration_time = peak_speed / acceleration_limit;
    let cruise_time = ((distance - peak_speed * acceleration_time) / peak_speed).max(T::ZERO);

    MotionProfile::from_phase_plan(
        [
            T::ZERO,
            acceleration_time,
            T::ZERO,
            cruise_time,
            T::ZERO,
            acceleration_time,
            T::ZERO,
        ],
        [T::ZERO; PROFILE_PHASE_COUNT],
        [
            T::ZERO,
            acceleration_limit,
            T::ZERO,
            T::ZERO,
            T::ZERO,
            -acceleration_limit,
            T::ZERO,
        ],
        sign,
    )
}

/// Jerk-phase and constant-acceleration durations to reach `peak_speed` from rest.
///
/// Acceleration either saturates at its limit and holds, or peaks below it and reverses at once,
/// in which case the hold is zero.
#[must_use]
fn acceleration_ramp<T: Numeric>(peak_speed: T, acceleration_limit: T, jerk_limit: T) -> (T, T) {
    if peak_speed * jerk_limit >= acceleration_limit * acceleration_limit {
        let jerk_time = acceleration_limit / jerk_limit;
        let hold_time = (peak_speed / acceleration_limit - jerk_time).max(T::ZERO);
        (jerk_time, hold_time)
    } else {
        ((peak_speed / jerk_limit).sqrt(), T::ZERO)
    }
}

/// Seven-phase solve with bounded jerk.
///
/// `distance` is non-negative; `sign` carries the direction.
#[must_use]
fn plan_jerk_limited<T: Numeric>(
    distance: T,
    limits: &ProfileLimits<T>,
    jerk_limit: T,
    sign: T,
) -> MotionProfile<T> {
    let acceleration_limit = limits.acceleration();

    // Peak velocity the displacement allows, one expression per ramp shape. They agree at
    // v = a_max² / j_max, so the branch is continuous.
    let speed_triangular = (distance * distance * jerk_limit / T::from_usize(4)).cbrt();
    let speed_reachable = if speed_triangular * jerk_limit < acceleration_limit * acceleration_limit
    {
        speed_triangular
    } else {
        let ratio = acceleration_limit * acceleration_limit / jerk_limit;
        (-ratio + (ratio * ratio + acceleration_limit * distance * T::from_usize(4)).sqrt())
            * T::HALF
    };

    let peak_speed = limits.speed().min(speed_reachable);
    if peak_speed <= T::ZERO {
        return plan_still(sign);
    }

    let (jerk_time, hold_time) = acceleration_ramp(peak_speed, acceleration_limit, jerk_limit);
    let ramp_time = jerk_time + jerk_time + hold_time;
    let cruise_time = ((distance - peak_speed * ramp_time) / peak_speed).max(T::ZERO);
    let peak_acceleration = jerk_limit * jerk_time;

    MotionProfile::from_phase_plan(
        [
            jerk_time,
            hold_time,
            jerk_time,
            cruise_time,
            jerk_time,
            hold_time,
            jerk_time,
        ],
        [
            jerk_limit,
            T::ZERO,
            -jerk_limit,
            T::ZERO,
            -jerk_limit,
            T::ZERO,
            jerk_limit,
        ],
        [
            T::ZERO,
            peak_acceleration,
            peak_acceleration,
            T::ZERO,
            T::ZERO,
            -peak_acceleration,
            -peak_acceleration,
        ],
        sign,
    )
}

/// Plans point-to-point moves against a fixed set of limits.
///
/// [`ProfileStrategy`] selects the shape, by default whichever the limits imply. The solve is
/// closed form — a fixed count of square and cube roots, no iteration — so planning cost is
/// constant.
///
/// ```
/// use multicalc::motion::{MotionProfilePlanner, ProfileLimits, ProfileStrategy};
///
/// // 10 m, v_max 2 m/s, a_max 1 m/s², j_max 2 m/s³.
/// let limits = ProfileLimits::<f64>::try_new(2.0, 1.0, Some(2.0)).unwrap();
/// let planner = MotionProfilePlanner::new(limits);
///
/// // A jerk limit is present, so the default plans the S-curve.
/// let eased = planner.plan(10.0).unwrap();
/// assert!((eased.duration() - 7.5).abs() < 1e-12);
///
/// // Trapezoidal ignores the jerk limit and finishes sooner.
/// let plain = planner
///     .with_strategy(ProfileStrategy::Trapezoidal)
///     .plan(10.0)
///     .unwrap();
/// assert!((plain.duration() - 7.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionProfilePlanner<T: Numeric = f64> {
    limits: ProfileLimits<T>,
    strategy: ProfileStrategy,
}

impl<T: Numeric> MotionProfilePlanner<T> {
    /// A planner against these limits, with the shape read from them.
    #[inline]
    #[must_use]
    pub fn new(limits: ProfileLimits<T>) -> Self {
        Self {
            limits,
            strategy: ProfileStrategy::Automatic,
        }
    }

    /// The same planner with an explicit shape.
    #[inline]
    #[must_use]
    pub fn with_strategy(mut self, strategy: ProfileStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// The limits it plans against.
    #[inline]
    #[must_use]
    pub fn limits(&self) -> ProfileLimits<T> {
        self.limits
    }

    /// The shape it plans.
    #[inline]
    #[must_use]
    pub fn strategy(&self) -> ProfileStrategy {
        self.strategy
    }

    /// The time-optimal profile for this displacement.
    ///
    /// A negative displacement moves the other way; zero gives a profile of zero duration. Returns
    /// [`MotionError::NonFinite`] for a displacement that is not finite, and
    /// [`MotionError::JerkLimitRequired`] for [`ProfileStrategy::JerkLimited`] against limits with
    /// no jerk bound.
    pub fn plan(&self, distance: T) -> Result<MotionProfile<T>, MotionError> {
        if !distance.is_finite() {
            return Err(MotionError::NonFinite);
        }

        let magnitude = distance.abs();
        let sign = if distance < T::ZERO { -T::ONE } else { T::ONE };

        match (self.strategy, self.limits.jerk()) {
            (ProfileStrategy::JerkLimited, None) => Err(MotionError::JerkLimitRequired),
            (ProfileStrategy::Trapezoidal, _) | (ProfileStrategy::Automatic, None) => {
                Ok(plan_trapezoidal(magnitude, &self.limits, sign))
            }
            (ProfileStrategy::Automatic | ProfileStrategy::JerkLimited, Some(jerk_limit)) => {
                Ok(plan_jerk_limited(magnitude, &self.limits, jerk_limit, sign))
            }
        }
    }
}

/// Per-axis kinematic state at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynchronizedState<const DIMENSION: usize, T: Numeric = f64> {
    /// Position per axis.
    pub position: Vector<DIMENSION, T>,
    /// Velocity per axis.
    pub velocity: Vector<DIMENSION, T>,
    /// Acceleration per axis.
    pub acceleration: Vector<DIMENSION, T>,
    /// Jerk per axis.
    pub jerk: Vector<DIMENSION, T>,
}

/// Per-axis profiles time-scaled to a common finish time.
///
/// Built from already-planned profiles, so each axis keeps its own limits and its own shape — a
/// seven-joint arm with a different velocity limit per joint needs nothing extra.
/// [`from_profiles`](SynchronizedProfile::from_profiles) paces to the slowest axis;
/// [`try_from_profiles_over`](SynchronizedProfile::try_from_profiles_over) paces to a requested
/// duration, raised to the slowest axis's if it is shorter rather than refused.
///
/// Only the pacing axis is time-optimal; the rest are time-scaled, which is exact and search-free.
///
/// ```
/// use multicalc::motion::{MotionProfilePlanner, ProfileLimits, SynchronizedProfile};
///
/// // Two joints with different limits, same displacement.
/// let fast = MotionProfilePlanner::new(ProfileLimits::<f64>::try_new(2.0, 1.0, None).unwrap());
/// let slow = MotionProfilePlanner::new(ProfileLimits::<f64>::try_new(0.5, 1.0, None).unwrap());
/// let synchronized = SynchronizedProfile::from_profiles([
///     fast.plan(1.0).unwrap(),
///     slow.plan(1.0).unwrap(),
/// ]);
///
/// // The slower joint sets the pace; both land on target.
/// let end = synchronized.state_at(synchronized.duration()).unwrap();
/// assert!((end.position[0] - 1.0).abs() < 1e-12);
/// assert!((end.position[1] - 1.0).abs() < 1e-12);
/// assert!(end.velocity.norm() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynchronizedProfile<const DIMENSION: usize, T: Numeric = f64> {
    axes: [MotionProfile<T>; DIMENSION],
    duration: T,
}

impl<const DIMENSION: usize, T: Numeric> SynchronizedProfile<DIMENSION, T> {
    /// Axes time-scaled to the slowest axis's duration.
    ///
    /// A zero-duration axis is left alone.
    #[must_use]
    pub fn from_profiles(axes: [MotionProfile<T>; DIMENSION]) -> Self {
        let slowest = slowest_duration(&axes);
        Self {
            axes: stretched_axes(axes, slowest),
            duration: slowest,
        }
    }

    /// Axes time-scaled to `duration`, or to the slowest axis's duration if that is longer.
    ///
    /// Read [`duration`](SynchronizedProfile::duration) back to see which applied. Returns
    /// [`MotionError::DurationNotPositive`] if `duration` is zero, negative or not finite.
    pub fn try_from_profiles_over(
        axes: [MotionProfile<T>; DIMENSION],
        duration: T,
    ) -> Result<Self, MotionError> {
        if !duration.is_finite() || duration <= T::ZERO {
            return Err(MotionError::DurationNotPositive);
        }

        let target = duration.max(slowest_duration(&axes));
        Ok(Self {
            axes: stretched_axes(axes, target),
            duration: target,
        })
    }

    /// Common finish time.
    #[inline]
    #[must_use]
    pub fn duration(&self) -> T {
        self.duration
    }

    /// One axis's profile, or `None` past the last axis.
    #[inline]
    #[must_use]
    pub fn axis(&self, index: usize) -> Option<&MotionProfile<T>> {
        self.axes.get(index)
    }

    /// Every axis's state at `time`, clamped to `[0, duration]`.
    ///
    /// Returns [`MotionError::NonFinite`] for a time that is not finite.
    pub fn state_at(&self, time: T) -> Result<SynchronizedState<DIMENSION, T>, MotionError> {
        let mut position = [T::ZERO; DIMENSION];
        let mut velocity = [T::ZERO; DIMENSION];
        let mut acceleration = [T::ZERO; DIMENSION];
        let mut jerk = [T::ZERO; DIMENSION];

        let axes = self.axes.iter();
        let outputs = position
            .iter_mut()
            .zip(velocity.iter_mut())
            .zip(acceleration.iter_mut())
            .zip(jerk.iter_mut());
        for (axis, (((axis_position, axis_velocity), axis_acceleration), axis_jerk)) in
            axes.zip(outputs)
        {
            let state = axis.state_at(time)?;
            *axis_position = state.position;
            *axis_velocity = state.velocity;
            *axis_acceleration = state.acceleration;
            *axis_jerk = state.jerk;
        }

        Ok(SynchronizedState {
            position: Vector::new(position),
            velocity: Vector::new(velocity),
            acceleration: Vector::new(acceleration),
            jerk: Vector::new(jerk),
        })
    }
}

/// The longest duration among these axes.
#[must_use]
fn slowest_duration<const DIMENSION: usize, T: Numeric>(axes: &[MotionProfile<T>; DIMENSION]) -> T {
    axes.iter()
        .fold(T::ZERO, |longest, axis| longest.max(axis.duration()))
}

/// Every axis time-scaled to finish at `target`.
#[must_use]
fn stretched_axes<const DIMENSION: usize, T: Numeric>(
    axes: [MotionProfile<T>; DIMENSION],
    target: T,
) -> [MotionProfile<T>; DIMENSION] {
    let mut axes = axes;
    for axis in axes.iter_mut() {
        *axis = axis.stretched_to(target);
    }
    axes
}
