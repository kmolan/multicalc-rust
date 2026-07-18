//! Fixed-timestep PID controller.

use crate::control::OnePoleLowPass;
use crate::error::ControlError;
use crate::scalar::Numeric;

/// A proportional-integral-derivative controller running at a fixed timestep.
///
/// The derivative acts on the error. That derivative is passed through a one-pole low-pass filter
/// that defaults to pass-through, so an unconfigured controller behaves like a textbook PID. Integral
/// wind-up is limited by conditional integration: while the output is saturated and the error would
/// drive it further into the active limit, the integral is held instead of accumulated. The output is
/// clamped to the configured limits, which default to unbounded.
///
/// Every operation is generic over [`Numeric`](crate::Numeric), so wrapping one `update` in a
/// [`Dual`](crate::Dual) differentiates the whole control law exactly.
///
/// ```
/// use multicalc::control::Pid;
///
/// // Drive a scalar integrator plant `x_next = x + dt * output` to a setpoint.
/// let dt = 0.01_f64;
/// let mut controller = Pid::new(2.0, 1.0, 0.0, dt).unwrap();
/// let setpoint = 1.0;
/// let mut measurement = 0.0;
/// for _ in 0..2000 {
///     let output = controller.update(setpoint, measurement);
///     measurement += dt * output;
/// }
/// assert!((measurement - setpoint).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pid<T: Numeric> {
    proportional_gain: T,
    integral_gain: T,
    derivative_gain: T,
    dt: T,
    output_minimum: T,
    output_maximum: T,
    integral: T,
    derivative_filter: OnePoleLowPass<T>,
    previous_error: T,
    has_previous_error: bool,
}

impl<T: Numeric> Pid<T> {
    /// Builds a controller from its three gains and a fixed timestep.
    ///
    /// Returns [`ControlError::NonFinite`] if any argument is not finite, or
    /// [`ControlError::NonPositiveTimestep`] if `dt` is not strictly positive. The output limits
    /// default to unbounded and the derivative filter defaults to pass-through.
    pub fn new(
        proportional_gain: T,
        integral_gain: T,
        derivative_gain: T,
        dt: T,
    ) -> Result<Self, ControlError> {
        if !proportional_gain.is_finite()
            || !integral_gain.is_finite()
            || !derivative_gain.is_finite()
            || !dt.is_finite()
        {
            return Err(ControlError::NonFinite);
        }
        if dt <= T::ZERO {
            return Err(ControlError::NonPositiveTimestep);
        }
        Ok(Self {
            proportional_gain,
            integral_gain,
            derivative_gain,
            dt,
            output_minimum: T::NEG_INFINITY,
            output_maximum: T::INFINITY,
            integral: T::ZERO,
            derivative_filter: OnePoleLowPass::new(T::ONE)?,
            previous_error: T::ZERO,
            has_previous_error: false,
        })
    }

    /// Sets the output saturation limits.
    ///
    /// An infinite limit means unbounded on that side. Returns [`ControlError::NonFinite`] if either
    /// limit is NaN, or [`ControlError::InvalidOutputLimits`] if `minimum` exceeds `maximum`.
    pub fn with_output_limits(mut self, minimum: T, maximum: T) -> Result<Self, ControlError> {
        if minimum.is_nan() || maximum.is_nan() {
            return Err(ControlError::NonFinite);
        }
        if minimum > maximum {
            return Err(ControlError::InvalidOutputLimits);
        }
        self.output_minimum = minimum;
        self.output_maximum = maximum;
        Ok(self)
    }

    /// Sets the smoothing coefficient of the derivative low-pass filter.
    ///
    /// Returns [`ControlError::NonFinite`] if `smoothing` is not finite, or
    /// [`ControlError::FilterCoefficientOutOfRange`] if it lies outside `[0, 1]`.
    pub fn with_derivative_filter(mut self, smoothing: T) -> Result<Self, ControlError> {
        self.derivative_filter = OnePoleLowPass::new(smoothing)?;
        Ok(self)
    }

    /// Advances the controller one timestep and returns the saturated output.
    pub fn update(&mut self, setpoint: T, measurement: T) -> T {
        let error = setpoint - measurement;
        let proportional_term = self.proportional_gain * error;

        let raw_derivative = if self.has_previous_error {
            (error - self.previous_error) / self.dt
        } else {
            T::ZERO
        };
        let derivative_term = self.derivative_gain * self.derivative_filter.filter(raw_derivative);
        self.previous_error = error;
        self.has_previous_error = true;

        let candidate_integral = self.integral + self.integral_gain * error * self.dt;
        let unsaturated = proportional_term + candidate_integral + derivative_term;
        let output = unsaturated
            .max(self.output_minimum)
            .min(self.output_maximum);

        let saturated_high = unsaturated > self.output_maximum;
        let saturated_low = unsaturated < self.output_minimum;
        let pushing_deeper =
            (saturated_high && error > T::ZERO) || (saturated_low && error < T::ZERO);
        if !pushing_deeper {
            self.integral = candidate_integral;
        }

        output
    }

    /// Clears the integral, derivative history, and filter state.
    pub fn reset(&mut self) {
        self.integral = T::ZERO;
        self.has_previous_error = false;
        self.previous_error = T::ZERO;
        self.derivative_filter.reset();
    }

    /// Returns the accumulated integral term.
    #[inline]
    #[must_use]
    pub fn integral(&self) -> T {
        self.integral
    }
}
