//! Blocks that shape a signal without filtering it: ignoring small values, switching with a gap,
//! and limiting how fast a value may change.

use crate::error::SignalError;
use crate::scalar::Numeric;

/// Treats values near zero as zero, so a resting sensor or a centred stick reads as no command.
///
/// Two forms. The plain one passes anything outside the band through untouched, which means the
/// output jumps from zero to the threshold the moment the band is left. The re-centered one slides
/// values outside the band back toward zero by the threshold, so the output leaves zero smoothly —
/// that is the one to use where a jump would become a jolt, such as a stick position or a torque
/// command.
///
/// ```
/// use multicalc::signal_processing::Deadband;
///
/// let plain = Deadband::plain(0.1_f64).unwrap();
/// assert!(plain.apply(0.05).abs() < 1e-12);
/// // Outside the band the input comes through as it is, so the output starts at 0.5.
/// assert!((plain.apply(0.5) - 0.5).abs() < 1e-12);
/// assert!((plain.apply(-0.5) + 0.5).abs() < 1e-12);
///
/// let recentered = Deadband::recentered(0.1_f64).unwrap();
/// assert!(recentered.apply(0.05).abs() < 1e-12);
/// // The same input slides back by the threshold, so the output leaves zero smoothly.
/// assert!((recentered.apply(0.5) - 0.4).abs() < 1e-12);
/// assert!((recentered.apply(-0.5) + 0.4).abs() < 1e-12);
/// ```
///
/// The band holds no state, so a non-finite value cannot spoil it, it falls straight through to
/// the output and is gone by the next call. `apply_checked` reports the bad reading instead of
/// passing it on to whatever consumes the output.
///
/// ```
/// use multicalc::signal_processing::Deadband;
///
/// let plain = Deadband::plain(0.1_f64).unwrap();
///
/// // Unchecked, a non-finite value goes straight through, one sample at a time
/// assert!(plain.apply(f64::NAN).is_nan());
/// assert!(plain.apply(f64::INFINITY).is_infinite());
///
/// // Nothing is carried over, so the very next value behaves normally
/// assert!(plain.apply(0.05).abs() < 1e-12);
///
/// // Checked, the bad reading is reported rather than passed on
/// for signal in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
///     assert!(plain.apply_checked(signal).is_err());
/// }
/// assert!(plain.apply_checked(0.5).is_ok());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Deadband<T: Numeric = f64> {
    /// How far from zero a value has to be before it counts.
    threshold: T,
    /// Whether values outside the band are shifted back toward zero.
    recentered: bool,
}

impl<T: Numeric> Deadband<T> {
    /// Builds a deadband that passes anything outside the band through untouched.
    ///
    /// Returns [`SignalError::NonFinite`] if `threshold` is not finite, or
    /// [`SignalError::NegativeThreshold`] if it is negative. A threshold of zero is allowed and
    /// makes the block a pass-through.
    pub fn plain(threshold: T) -> Result<Self, SignalError> {
        Self::build(threshold, false)
    }

    /// Builds a deadband that slides values outside the band back toward zero by the threshold.
    ///
    /// Returns the same errors as [`plain`](Self::plain).
    pub fn recentered(threshold: T) -> Result<Self, SignalError> {
        Self::build(threshold, true)
    }

    /// Applies the band to one value.
    ///
    /// The band holds no state, so a non-finite value cannot spoil anything: the comparison against
    /// the threshold reads false and the value falls straight through to the output, one sample at a
    /// time. `apply_checked` refuses it instead, so a bad reading is caught here rather than passed
    /// on to whatever consumes the output.
    #[inline]
    #[must_use]
    pub fn apply(&self, input: T) -> T {
        if input.abs() <= self.threshold {
            T::ZERO
        } else if self.recentered {
            input - self.threshold.copysign(input)
        } else {
            input
        }
    }

    /// Alternative to `apply` with checked input.
    /// Returns `SignalError::NonFinite` in case of non-finite input.
    ///
    /// There is no state to protect here - this reports the bad reading rather than letting it
    /// through to the rest of the chain.
    #[inline]
    pub fn apply_checked(&self, input: T) -> Result<T, SignalError> {
        if input.is_finite() {
            Ok(self.apply(input))
        } else {
            Err(SignalError::NonFinite)
        }
    }

    /// How far from zero a value has to be before it counts.
    #[inline]
    #[must_use]
    pub fn threshold(&self) -> T {
        self.threshold
    }

    fn build(threshold: T, recentered: bool) -> Result<Self, SignalError> {
        if !threshold.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if threshold < T::ZERO {
            return Err(SignalError::NegativeThreshold);
        }
        Ok(Self {
            threshold,
            recentered,
        })
    }
}

/// A yes-or-no answer with a gap between the two thresholds, so a signal sitting near the switching
/// point does not chatter.
///
/// The answer turns yes only above the upper threshold and back to no only below the lower one; in
/// between it holds whatever it already was. It starts as no.
///
/// ```
/// use multicalc::error::SignalError;
/// use multicalc::signal_processing::Hysteresis;
///
/// let mut switch = Hysteresis::new(0.4_f64, 0.6).unwrap();
///
/// // It starts as no, and a value inside the gap leaves it there.
/// assert!(!switch.update(0.5));
/// // Above the upper threshold it turns yes, and stays yes back inside the gap.
/// assert!(switch.update(0.7));
/// assert!(switch.update(0.5));
/// // Only below the lower threshold does it turn no again.
/// assert!(!switch.update(0.3));
///
/// assert_eq!(
///     Hysteresis::new(0.6_f64, 0.4),
///     Err(SignalError::ThresholdsOutOfOrder)
/// );
/// ```
///
/// The answer is a yes or a no, so it cannot be corrupted the way a numeric state can. The risk
/// here is the opposite one: a NaN loses both comparisons and the switch silently holds, which
/// looks exactly like a steady signal parked inside the gap. `update_checked` tells the two apart.
///
/// ```
/// use multicalc::signal_processing::Hysteresis;
///
/// let mut switch = Hysteresis::new(0.4_f64, 0.6).unwrap();
///
/// // Turn it on with a real reading.
/// assert!(switch.update(0.7));
///
/// // A NaN changes nothing at all — the answer is held, not corrupted.
/// assert!(switch.update(f64::NAN));
/// assert!(switch.is_high());
///
/// // Infinities are ordinary extreme values and switch it as you would expect.
/// assert!(!switch.update(f64::NEG_INFINITY));
/// assert!(switch.update(f64::INFINITY));
///
/// // Checked, a non-finite reading is reported and the answer is left alone.
/// let before = switch.is_high();
/// assert!(switch.update_checked(f64::NAN).is_err());
/// assert_eq!(switch.is_high(), before);
/// assert_eq!(switch.update_checked(0.3), Ok(false));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hysteresis<T: Numeric = f64> {
    /// Fall below this and the answer becomes no.
    lower: T,
    /// Rise above this and the answer becomes yes.
    upper: T,
    /// The current answer.
    is_high: bool,
}

impl<T: Numeric> Hysteresis<T> {
    /// Builds a switch that starts as no.
    ///
    /// Returns [`SignalError::NonFinite`] if either threshold is not finite, or
    /// [`SignalError::ThresholdsOutOfOrder`] if `lower` is not below `upper`.
    pub fn new(lower: T, upper: T) -> Result<Self, SignalError> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if lower >= upper {
            return Err(SignalError::ThresholdsOutOfOrder);
        }
        Ok(Self {
            lower,
            upper,
            is_high: false,
        })
    }

    /// Feeds one value and returns the answer it leaves behind.
    ///
    /// The answer is a yes or a no, so nothing here can be spoiled the way a numeric state can.
    /// Infinities work as ordinary extreme values and switch it as you would expect. A NaN loses
    /// both comparisons, so the switch simply holds whatever it already was — a dead sensor
    /// reads exactly like a signal parked inside the gap, and nothing in the answer says otherwise.
    /// `update_checked` is the way to tell the two apart.
    #[inline]
    #[must_use]
    pub fn update(&mut self, input: T) -> bool {
        if input > self.upper {
            self.is_high = true;
        } else if input < self.lower {
            self.is_high = false;
        }
        self.is_high
    }

    /// Alternative to `update` with checked input.
    /// Returns `SignalError::NonFinite` in case of non-finite input, leaving the answer untouched.
    ///
    /// The answer cannot be corrupted, so this is about visibility: without it a run of NaN
    /// readings is indistinguishable from a steady signal inside the gap.
    #[inline]
    pub fn update_checked(&mut self, input: T) -> Result<bool, SignalError> {
        if !input.is_nan() {
            Ok(self.update(input))
        } else {
            Err(SignalError::NonFinite)
        }
    }

    /// Sets the answer back to no.
    #[inline]
    pub fn reset(&mut self) {
        self.is_high = false;
    }

    /// The current answer, without feeding a value.
    #[inline]
    #[must_use]
    pub fn is_high(&self) -> bool {
        self.is_high
    }
}

/// Follows a target without ever moving faster than the rates it was given, so a step in the target
/// comes out as a ramp.
///
/// Rising and falling have separate limits, since a machine that can speed up gently often has to
/// slow down harder. The first call jumps straight to its target rather than ramping up from zero.
///
/// ```
/// use multicalc::error::SignalError;
/// use multicalc::signal_processing::SlewRateLimiter;
///
/// // Climbing at one per second, falling at two, a tenth of a second at a time.
/// let mut limited = SlewRateLimiter::new(1.0_f64, 2.0, 0.1).unwrap();
///
/// // The first call goes straight to its target.
/// assert!(limited.filter(0.0).abs() < 1e-12);
///
/// // A jump to 10 comes out as a ramp: a tenth of a unit per call.
/// assert!((limited.filter(10.0) - 0.1).abs() < 1e-12);
/// for _ in 0..9 {
///     let _ = limited.filter(10.0);
/// }
/// assert!((limited.value() - 1.0).abs() < 1e-12);
///
/// // Turning around, it moves twice as fast.
/// assert!((limited.filter(-10.0) - 0.8).abs() < 1e-12);
///
/// assert_eq!(
///     SlewRateLimiter::new(0.0_f64, 1.0, 0.1),
///     Err(SignalError::NonPositiveRate)
/// );
/// ```
///
/// The `SlewRateLimiter` has another checked entry point for cases where the target could be
/// non-finite. The checked entry point refuses exactly the targets that would spoil the state, and
/// lets the harmless ones through.
///
/// ```
/// use multicalc::signal_processing::SlewRateLimiter;
///
/// let mut running = SlewRateLimiter::new(1.0_f64, 2.0, 0.1).unwrap();
///
/// let _ = running.filter(0.0);
/// let running_snapshot = running;
///
/// // A NaN target is always refused, and the limiter is left exactly where it was
/// let output = running.filter_checked(f64::NAN);
/// assert!(output.is_err());
/// assert_eq!(running, running_snapshot);
///
/// // Once seeded, an infinite target cannot spoil anything, so it is accepted and clamped
/// let output = running.filter_checked(f64::INFINITY);
/// assert!(output.is_ok());
/// assert!(output.unwrap().is_finite());
///
/// let output = running.filter_checked(0.1_f64);
/// assert!(output.is_ok());
/// assert!(output.unwrap().is_finite());
///
/// // NaN spoils the limiter ..
/// let _ = running.filter(f64::NAN);
/// let output = running.filter(1.0);
/// assert!(output.is_nan());
///
/// //.. till reset
/// running.reset();
/// let output = running.filter(1.0);
/// assert!(output.is_finite());
/// ```
///
/// An infinite target is a different story. Once the limiter has been seeded the rate clamp turns
/// it into one ordinary step, so it does no harm at all. Only an infinite *first* target sticks,
/// because the first call takes its target as the starting point without clamping it.
///
/// ```
/// use multicalc::signal_processing::SlewRateLimiter;
///
/// // Seeded first, an infinite target just moves one rise step.
/// let mut seeded = SlewRateLimiter::new(1.0_f64, 2.0, 0.1).unwrap();
/// let _ = seeded.filter(0.0);
/// assert!((seeded.filter(f64::INFINITY) - 0.1).abs() < 1e-12);
/// assert!(seeded.value().is_finite());
///
/// // As the very first target it becomes the starting point, and nothing brings it back.
/// let mut fresh = SlewRateLimiter::new(1.0_f64, 2.0, 0.1).unwrap();
/// assert!(fresh.filter(f64::INFINITY).is_infinite());
/// assert!(fresh.filter(1.0).is_infinite());
///
/// // That is the one case where the checked entry point refuses an infinity.
/// let mut guarded = SlewRateLimiter::new(1.0_f64, 2.0, 0.1).unwrap();
/// assert!(guarded.filter_checked(f64::INFINITY).is_err());
///
/// // Seeded with a finite target first, the same infinity is then accepted.
/// assert!(guarded.filter_checked(0.0).is_ok());
/// assert!(guarded.filter_checked(f64::INFINITY).is_ok());
/// assert!(guarded.value().is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlewRateLimiter<T: Numeric = f64> {
    /// Most the value may climb per second.
    rise_per_second: T,
    /// Most the value may fall per second.
    fall_per_second: T,
    /// Seconds between calls.
    timestep: T,
    /// Where the output currently sits.
    state: T,
    /// Whether the first call has seeded the output.
    initialized: bool,
}

impl<T: Numeric> SlewRateLimiter<T> {
    /// Builds a limiter with separate rise and fall rates, in units per second.
    ///
    /// Returns [`SignalError::NonFinite`] if any argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `timestep` is not strictly positive, or
    /// [`SignalError::NonPositiveRate`] if either rate is not strictly positive.
    pub fn new(rise_per_second: T, fall_per_second: T, timestep: T) -> Result<Self, SignalError> {
        if !rise_per_second.is_finite() || !fall_per_second.is_finite() || !timestep.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if timestep <= T::ZERO {
            return Err(SignalError::NonPositiveTimestep);
        }
        if rise_per_second <= T::ZERO || fall_per_second <= T::ZERO {
            return Err(SignalError::NonPositiveRate);
        }
        Ok(Self {
            rise_per_second,
            fall_per_second,
            timestep,
            state: T::ZERO,
            initialized: false,
        })
    }

    /// Builds a limiter that rises and falls at the same rate.
    ///
    /// Returns the same errors as [`new`](Self::new).
    pub fn symmetric(rate_per_second: T, timestep: T) -> Result<Self, SignalError> {
        Self::new(rate_per_second, rate_per_second, timestep)
    }

    /// Moves one step toward the target and returns where the output now sits.
    ///
    /// A NaN target spoils the state till the next reset. An infinite target is harmless once the
    /// limiter has been seeded with finite value, because the rate clamp turns it into one ordinary step.
    /// Only an infinite first target with uninitialized state could corrupt the state with NaN.
    /// These NaN and INFINITY inputs are better handled by `filter_checked`.
    #[inline]
    #[must_use]
    pub fn filter(&mut self, target: T) -> T {
        if !self.initialized {
            self.state = target;
            self.initialized = true;
            return self.state;
        }

        // How far the output may move this call, in each direction.
        let rise_step = self.rise_per_second * self.timestep;
        let fall_step = self.fall_per_second * self.timestep;

        let step = target - self.state;
        self.state += if step > rise_step {
            rise_step
        } else if step < -fall_step {
            -fall_step
        } else {
            step
        };
        self.state
    }

    /// Alternative to `filter` with checked input.
    /// Returns `SignalError::NonFinite` in case of:
    /// - NaN target
    /// - Infinity target and uninitialized state
    ///
    /// The state must be guarded from non-finite value at all costs and
    /// the use of `filter_checked` must be consistent. The entry point cannot prevent the damage
    /// once it is done by a previous `filter` call with non-finite input.
    #[inline]
    pub fn filter_checked(&mut self, target: T) -> Result<T, SignalError> {
        if target.is_nan() || (target.is_infinite() && !self.initialized) {
            Err(SignalError::NonFinite)
        } else {
            Ok(self.filter(target))
        }
    }

    /// Clears the output so the next call jumps straight to its target again.
    #[inline]
    pub fn reset(&mut self) {
        self.state = T::ZERO;
        self.initialized = false;
    }

    /// Where the output currently sits, without feeding a target.
    #[inline]
    #[must_use]
    pub fn value(&self) -> T {
        self.state
    }
}
