//! Fixed-timestep PID controller.

use crate::error::ControlError;
use crate::scalar::Numeric;
use crate::signal_processing::OnePoleLowPass;

/// A proportional-integral-derivative controller running at a fixed timestep.
///
/// The derivative acts on the measurement rather than on the error, so a jump in the setpoint does
/// not send a spike through the derivative gain; when the setpoint holds still the two are the same
/// thing. That derivative is passed through a one-pole low-pass filter
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
/// // Drive a scalar integrator plant `x_next = x + timestep * output` to a setpoint.
/// let proportional_gain = 2.0_f64;
/// let integral_gain = 1.0;
/// let derivative_gain = 0.0;
/// let timestep = 0.01;
///
/// let mut controller =
///     Pid::new(proportional_gain, integral_gain, derivative_gain, timestep).unwrap();
/// let setpoint = 1.0;
/// let mut measurement = 0.0;
/// for _ in 0..2000 {
///     let output = controller.update(setpoint, measurement);
///     measurement += timestep * output;
/// }
/// assert!((measurement - setpoint).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pid<T: Numeric = f64> {
    proportional_gain: T,
    integral_gain: T,
    derivative_gain: T,
    dt: T,
    output_minimum: T,
    output_maximum: T,
    integral: T,
    derivative_filter: OnePoleLowPass<T>,
    previous_measurement: T,
    previous_error: T,
    has_previous_measurement: bool,
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
            previous_measurement: T::ZERO,
            previous_error: T::ZERO,
            has_previous_measurement: false,
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
    /// Returns [`ControlError::Signal`] carrying
    /// [`SignalError::NonFinite`](crate::error::SignalError::NonFinite) if `smoothing` is not
    /// finite, or
    /// [`SignalError::CoefficientOutOfRange`](crate::error::SignalError::CoefficientOutOfRange) if
    /// it lies outside `[0, 1]`.
    pub fn with_derivative_filter(mut self, smoothing: T) -> Result<Self, ControlError> {
        self.derivative_filter = OnePoleLowPass::new(smoothing)?;
        Ok(self)
    }

    /// Changes the three gains without the output stepping.
    ///
    /// The stored integral is shifted by exactly as much as the new gains change the other two
    /// terms, so the command coming out of the next call is what the old gains would have given
    /// and the new gains take effect from there. Nothing is shifted before the first `update`,
    /// where there is no output to hold on to. The integral is stored with the integral gain
    /// already applied, so changing that gain never steps the output on its own.
    ///
    /// Returns [`ControlError::NonFinite`] if any gain is not finite.
    pub fn set_gains(
        &mut self,
        proportional_gain: T,
        integral_gain: T,
        derivative_gain: T,
    ) -> Result<(), ControlError> {
        if !proportional_gain.is_finite()
            || !integral_gain.is_finite()
            || !derivative_gain.is_finite()
        {
            return Err(ControlError::NonFinite);
        }
        if self.has_previous_measurement {
            let filtered_derivative = self.derivative_filter.value();
            self.integral = self.integral
                + (self.proportional_gain - proportional_gain) * self.previous_error
                + (self.derivative_gain - derivative_gain) * filtered_derivative;
        }
        self.proportional_gain = proportional_gain;
        self.integral_gain = integral_gain;
        self.derivative_gain = derivative_gain;
        Ok(())
    }

    /// Takes over from a command that was being driven some other way, without the output
    /// stepping.
    ///
    /// Give it the command currently going to the actuator along with the setpoint and measurement
    /// that go with it. The integral is set so that calling `update` with that same pair returns
    /// exactly that command, and the controller carries on from there. The measurement history is
    /// seeded too, so the first derivative is taken against the handover point rather than against
    /// nothing.
    ///
    /// Returns [`ControlError::NonFinite`] if any argument is not finite.
    pub fn resume_from(
        &mut self,
        output: T,
        setpoint: T,
        measurement: T,
    ) -> Result<(), ControlError> {
        if !output.is_finite() || !setpoint.is_finite() || !measurement.is_finite() {
            return Err(ControlError::NonFinite);
        }
        let error = setpoint - measurement;
        // The next call adds one step of integral action of its own and sees no change in the
        // measurement, so the integral is seeded short by that one step and the derivative
        // contributes nothing.
        self.integral =
            output - self.proportional_gain * error - self.integral_gain * error * self.dt;
        self.derivative_filter.reset();
        self.previous_measurement = measurement;
        self.previous_error = error;
        self.has_previous_measurement = true;
        Ok(())
    }

    /// Advances the controller one timestep and returns the saturated output.
    #[must_use]
    pub fn update(&mut self, setpoint: T, measurement: T) -> T {
        let error = setpoint - measurement;
        let proportional_term = self.proportional_gain * error;

        // The measurement falling is the same as the error rising, so the difference is taken the
        // other way round and the term keeps the sign a textbook PID gives it.
        let raw_derivative = if self.has_previous_measurement {
            (self.previous_measurement - measurement) / self.dt
        } else {
            T::ZERO
        };
        let derivative_term = self.derivative_gain * self.derivative_filter.filter(raw_derivative);
        self.previous_measurement = measurement;
        self.previous_error = error;
        self.has_previous_measurement = true;

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

    /// Clears the integral, measurement history, and filter state.
    pub fn reset(&mut self) {
        self.integral = T::ZERO;
        self.has_previous_measurement = false;
        self.previous_measurement = T::ZERO;
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
