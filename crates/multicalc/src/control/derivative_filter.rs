//! Single-pole low-pass filter used to smooth a noisy signal.

use crate::error::ControlError;
use crate::scalar::Numeric;

/// A one-pole infinite-impulse-response low-pass filter.
///
/// The recurrence is `y_n = smoothing * x_n + (1 - smoothing) * y_{n-1}`, where the smoothing
/// coefficient (α) lies in the closed interval `[0, 1]`. A value of `1` is pass-through and smaller
/// values apply heavier smoothing. The first sample seeds the state directly, so there is no startup
/// transient from a zero initial state.
///
/// ```
/// use multicalc::control::OnePoleLowPass;
///
/// // Pass-through: the output reproduces the input exactly.
/// let mut passthrough = OnePoleLowPass::new(1.0_f64).unwrap();
/// assert_eq!(passthrough.filter(3.0), 3.0);
/// assert_eq!(passthrough.filter(-2.0), -2.0);
///
/// // A constant input converges to that constant.
/// let mut smoother = OnePoleLowPass::new(0.5_f64).unwrap();
/// for _ in 0..64 {
///     smoother.filter(10.0);
/// }
/// assert!((smoother.value() - 10.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnePoleLowPass<T: Numeric> {
    smoothing: T,
    state: T,
    initialized: bool,
}

impl<T: Numeric> OnePoleLowPass<T> {
    /// Builds a filter from a smoothing coefficient in `[0, 1]`.
    ///
    /// Returns [`ControlError::NonFinite`] if `smoothing` is not finite, or
    /// [`ControlError::FilterCoefficientOutOfRange`] if it lies outside `[0, 1]`.
    pub fn new(smoothing: T) -> Result<Self, ControlError> {
        if !smoothing.is_finite() {
            return Err(ControlError::NonFinite);
        }
        if smoothing < T::ZERO || smoothing > T::ONE {
            return Err(ControlError::FilterCoefficientOutOfRange);
        }
        Ok(Self {
            smoothing,
            state: T::ZERO,
            initialized: false,
        })
    }

    /// Builds a filter from a cutoff frequency in hertz and a timestep in seconds.
    ///
    /// The smoothing coefficient is `a / (a + 1)` with `a = 2 * pi * cutoff_hz * dt`. Returns
    /// [`ControlError::NonFinite`] if either argument is not finite,
    /// [`ControlError::NonPositiveTimestep`] if `dt` is not strictly positive, or
    /// [`ControlError::FilterCoefficientOutOfRange`] if `cutoff_hz` is negative.
    pub fn from_cutoff(cutoff_hz: T, dt: T) -> Result<Self, ControlError> {
        if !cutoff_hz.is_finite() || !dt.is_finite() {
            return Err(ControlError::NonFinite);
        }
        if dt <= T::ZERO {
            return Err(ControlError::NonPositiveTimestep);
        }
        if cutoff_hz < T::ZERO {
            return Err(ControlError::FilterCoefficientOutOfRange);
        }
        let a = T::TWO * T::PI * cutoff_hz * dt;
        let smoothing = a / (a + T::ONE);
        Self::new(smoothing)
    }

    /// Feeds one sample and returns the updated output.
    #[inline]
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
