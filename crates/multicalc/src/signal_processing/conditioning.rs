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
/// There is no state here, so a non-finite value falls straight through and is gone by the next
/// call. [`apply_checked`](Self::apply_checked) reports it rather than passing it on.
///
/// ```
/// use multicalc::signal_processing::Deadband;
///
/// let plain = Deadband::plain(0.1_f64).unwrap();
///
/// assert!(plain.apply(f64::NAN).is_nan());
/// assert_eq!(plain.apply(0.05), 0.0); // nothing carried over
///
/// assert!(plain.apply_checked(f64::NAN).is_err());
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
    /// There is no state to spoil: a non-finite value falls straight through to the output. See
    /// [`apply_checked`](Self::apply_checked).
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

    /// [`apply`](Self::apply) with the value checked.
    ///
    /// Returns [`SignalError::NonFinite`] for a non-finite value. There is no state to protect
    /// here: this catches the bad reading rather than letting it through to the rest of the chain.
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
/// A yes-or-no answer cannot be corrupted the way a numeric state can. The risk is the opposite
/// one: a NaN loses both comparisons and the switch silently holds, which looks exactly like a
/// steady signal parked inside the gap. [`update_checked`](Self::update_checked) tells the two
/// apart. Infinities are ordinary extreme values and switch it normally.
///
/// ```
/// use multicalc::signal_processing::Hysteresis;
///
/// let mut switch = Hysteresis::new(0.4_f64, 0.6).unwrap();
/// assert!(switch.update(0.7));
///
/// // A NaN changes nothing at all — the answer is held, not corrupted.
/// assert!(switch.update(f64::NAN));
/// assert!(switch.update_checked(f64::NAN).is_err());
/// assert!(switch.is_high());
///
/// assert!(!switch.update(f64::NEG_INFINITY));
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
    /// A NaN loses both comparisons, so the switch silently holds — indistinguishable from a signal
    /// parked inside the gap. See [`update_checked`](Self::update_checked).
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

    /// [`update`](Self::update) with the value checked.
    ///
    /// Returns [`SignalError::NonFinite`] for a NaN, leaving the answer untouched. The answer
    /// cannot be corrupted, so this is about visibility rather than protection.
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
/// A NaN target latches until [`reset`](Self::reset). An infinite target is harmless once the
/// limiter is seeded, because the rate clamp turns it into one ordinary step — only an infinite
/// *first* target sticks, since the first call takes its target as the starting point unclamped.
/// [`filter_checked`](Self::filter_checked) refuses exactly those two cases.
///
/// ```
/// use multicalc::signal_processing::SlewRateLimiter;
///
/// // Unseeded, an infinite target becomes the starting point and nothing brings it back.
/// let mut fresh = SlewRateLimiter::new(1.0_f64, 2.0, 0.1).unwrap();
/// assert!(fresh.filter(f64::INFINITY).is_infinite());
/// assert!(fresh.filter(1.0).is_infinite());
///
/// let mut running = SlewRateLimiter::new(1.0_f64, 2.0, 0.1).unwrap();
/// assert!(running.filter_checked(f64::INFINITY).is_err());
///
/// // Seeded, the rate clamp bounds it, so the same target is accepted.
/// let _ = running.filter(0.0);
/// assert!((running.filter_checked(f64::INFINITY).unwrap() - 0.1).abs() < 1e-12);
///
/// // A NaN is refused whatever the state.
/// let untouched = running;
/// assert!(running.filter_checked(f64::NAN).is_err());
/// assert_eq!(running, untouched);
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
    /// A NaN target latches until [`reset`](Self::reset), as does an infinite *first* target. Once
    /// seeded, the rate clamp makes an infinite target harmless. See
    /// [`filter_checked`](Self::filter_checked).
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

    /// [`filter`](Self::filter) with the target checked.
    ///
    /// Returns [`SignalError::NonFinite`] for a NaN target, or for an infinite one while the
    /// limiter is unseeded. It cannot undo damage a previous [`filter`](Self::filter) call has
    /// already done, so pick one entry point and stay with it.
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
