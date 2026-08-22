//! The simplest low-pass filter: a weighted blend of the new sample and the running value.

use crate::error::SignalError;
use crate::scalar::Numeric;

/// A one-pole infinite-impulse-response low-pass filter.
///
/// The recurrence is `y_n = smoothing * x_n + (1 - smoothing) * y_{n-1}`, where the smoothing
/// coefficient (α) lies in the closed interval `[0, 1]`. A value of `1` is pass-through and smaller
/// values apply heavier smoothing. The first sample seeds the state directly, so there is no startup
/// transient from a zero initial state.
///
/// ```
/// use multicalc::signal_processing::OnePoleLowPass;
///
/// // A weight of 1 keeps all of the new sample, so the output reproduces the input exactly.
/// let keep_everything = 1.0_f64;
/// let mut passthrough = OnePoleLowPass::new(keep_everything).unwrap();
/// assert_eq!(passthrough.filter(3.0), 3.0);
/// assert_eq!(passthrough.filter(-2.0), -2.0);
///
/// // A smaller weight leans on the history, and a constant input converges to that constant.
/// let half_new_half_old = 0.5_f64;
/// let mut smoother = OnePoleLowPass::new(half_new_half_old).unwrap();
/// let steady_input = 10.0;
/// for _ in 0..64 {
///     let _ = smoother.filter(steady_input);
/// }
/// assert!((smoother.value() - steady_input).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnePoleLowPass<T: Numeric = f64> {
    smoothing: T,
    state: T,
    initialized: bool,
}

impl<T: Numeric> OnePoleLowPass<T> {
    /// Builds a filter from a smoothing coefficient in `[0, 1]`.
    ///
    /// Returns [`SignalError::NonFinite`] if `smoothing` is not finite, or
    /// [`SignalError::CoefficientOutOfRange`] if it lies outside `[0, 1]`.
    pub fn new(smoothing: T) -> Result<Self, SignalError> {
        if !smoothing.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if smoothing < T::ZERO || smoothing > T::ONE {
            return Err(SignalError::CoefficientOutOfRange);
        }
        Ok(Self {
            smoothing,
            state: T::ZERO,
            initialized: false,
        })
    }

    /// Builds a filter from a cutoff frequency in hertz and a timestep in seconds.
    ///
    /// The smoothing coefficient is `a / (a + 1)` with `a = 2 * pi * cutoff_hz * timestep`. Returns
    /// [`SignalError::NonFinite`] if either argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `timestep` is not strictly positive, or
    /// [`SignalError::FrequencyOutOfRange`] if `cutoff_hz` is negative.
    pub fn from_cutoff(cutoff_hz: T, timestep: T) -> Result<Self, SignalError> {
        if !cutoff_hz.is_finite() || !timestep.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if timestep <= T::ZERO {
            return Err(SignalError::NonPositiveTimestep);
        }
        if cutoff_hz < T::ZERO {
            return Err(SignalError::FrequencyOutOfRange);
        }
        let a = T::TWO * T::PI * cutoff_hz * timestep;
        let smoothing = a / (a + T::ONE);
        Self::new(smoothing)
    }

    /// Feeds one sample and returns the updated output.
    #[inline]
    #[must_use]
    pub fn filter(&mut self, input: T) -> T {
        if self.initialized {
            self.state = self.smoothing * input + (T::ONE - self.smoothing) * self.state;
        } else {
            self.state = input;
            self.initialized = true;
        }
        self.state
    }

    /// Clears the state so the next sample seeds the filter again.
    #[inline]
    pub fn reset(&mut self) {
        self.state = T::ZERO;
        self.initialized = false;
    }

    /// Returns the current output without feeding a sample.
    #[inline]
    #[must_use]
    pub fn value(&self) -> T {
        self.state
    }
}
