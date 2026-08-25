//! A second-order filter: its shape, and the running filter that applies it.

use crate::error::SignalError;
use crate::scalar::Numeric;

/// The shape of a second-order filter: how much of the newest input and the two before it go into
/// the output, and how much of the two previous outputs come back.
///
/// The weights are stored already divided through by the leading output weight, and the output
/// weights are stored with the sign they have when written on the left-hand side — so the running
/// filter subtracts them. This was chosen because this is the ordering SciPy uses.
///
/// A filter is designed once from a frequency in hertz, a sharpness, and the seconds between
/// samples. The sharpness sets how abruptly the filter acts: around 0.7 gives the flattest low-pass,
/// higher values give a narrower notch or a peakier band.
///
/// ```
/// use multicalc::signal_processing::BiquadCoefficients;
///
/// // A 50 Hz low-pass, sampled every millisecond. A low-pass passes a steady input through
/// // untouched, so the input weights add up to whatever the output weights leave behind.
/// let low_pass = BiquadCoefficients::low_pass(50.0_f64, 0.70710678, 0.001).unwrap();
/// let input_weights = low_pass.feed_forward();
/// let output_weights = low_pass.feedback();
/// let from_input = input_weights[0] + input_weights[1] + input_weights[2];
/// let from_output = 1.0 + output_weights[0] + output_weights[1];
/// assert!((from_input - from_output).abs() < 1e-12);
///
/// // A notch removes one frequency and leaves a steady input alone as well, so the same
/// // identity holds for it.
/// let notch = BiquadCoefficients::notch(180.0_f64, 4.0, 0.001).unwrap();
/// let input_weights = notch.feed_forward();
/// let output_weights = notch.feedback();
/// let from_input = input_weights[0] + input_weights[1] + input_weights[2];
/// let from_output = 1.0 + output_weights[0] + output_weights[1];
/// assert!((from_input - from_output).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadCoefficients<T: Numeric = f64> {
    /// Weights on the newest input sample and the two before it.
    feed_forward: [T; 3],
    /// Weights on the two previous outputs.
    feedback: [T; 2],
    /// Seconds between samples.
    timestep: T,
}

/// Which of the four filter shapes a design builds. Private: the public way in is one of the
/// four named constructors.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Design {
    LowPass,
    HighPass,
    BandPass,
    Notch,
}

impl<T: Numeric> BiquadCoefficients<T> {
    /// Builds a filter shape from weights that are already divided through by the leading output
    /// weight, together with the seconds between samples.
    ///
    /// Returns [`SignalError::NonFinite`] if any weight or `timestep` is not finite, or
    /// [`SignalError::NonPositiveTimestep`] if `timestep` is not strictly positive.
    pub fn new(feed_forward: [T; 3], feedback: [T; 2], timestep: T) -> Result<Self, SignalError> {
        for weight in feed_forward.into_iter().chain(feedback) {
            if !weight.is_finite() {
                return Err(SignalError::NonFinite);
            }
        }
        if !timestep.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if timestep <= T::ZERO {
            return Err(SignalError::NonPositiveTimestep);
        }
        Ok(Self {
            feed_forward,
            feedback,
            timestep,
        })
    }

    /// Builds a low-pass, which keeps content below the cutoff and fades out what is above it.
    ///
    /// Returns [`SignalError::NonFinite`] if any argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `timestep` is not strictly positive,
    /// [`SignalError::NonPositiveQualityFactor`] if `quality_factor` is not strictly positive, or
    /// [`SignalError::FrequencyOutOfRange`] if `cutoff_hz` is not strictly positive or reaches half
    /// the sampling rate.
    ///
    /// ```
    /// use multicalc::signal_processing::BiquadCoefficients;
    ///
    /// let low_pass = BiquadCoefficients::low_pass(50.0_f64, 0.70710678, 0.001).unwrap();
    /// assert_eq!(low_pass.timestep(), 0.001);
    /// ```
    pub fn low_pass(cutoff_hz: T, quality_factor: T, timestep: T) -> Result<Self, SignalError> {
        Self::build(Design::LowPass, cutoff_hz, quality_factor, timestep)
    }

    /// Builds a high-pass, which keeps content above the cutoff and fades out what is below it.
    ///
    /// Returns [`SignalError::NonFinite`] if any argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `timestep` is not strictly positive,
    /// [`SignalError::NonPositiveQualityFactor`] if `quality_factor` is not strictly positive, or
    /// [`SignalError::FrequencyOutOfRange`] if `cutoff_hz` is not strictly positive or reaches half
    /// the sampling rate.
    ///
    /// ```
    /// use multicalc::signal_processing::BiquadCoefficients;
    ///
    /// // A high-pass blocks a steady input, so its input weights cancel out.
    /// let high_pass = BiquadCoefficients::high_pass(50.0_f64, 0.70710678, 0.001).unwrap();
    /// let weights = high_pass.feed_forward();
    /// assert!((weights[0] + weights[1] + weights[2]).abs() < 1e-12);
    /// ```
    pub fn high_pass(cutoff_hz: T, quality_factor: T, timestep: T) -> Result<Self, SignalError> {
        Self::build(Design::HighPass, cutoff_hz, quality_factor, timestep)
    }

    /// Builds a band-pass, which keeps a band of frequencies around the centre and fades out
    /// everything to either side of it.
    ///
    /// Returns [`SignalError::NonFinite`] if any argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `timestep` is not strictly positive,
    /// [`SignalError::NonPositiveQualityFactor`] if `quality_factor` is not strictly positive, or
    /// [`SignalError::FrequencyOutOfRange`] if `center_hz` is not strictly positive or reaches half
    /// the sampling rate.
    ///
    /// ```
    /// use multicalc::signal_processing::BiquadCoefficients;
    ///
    /// // A band-pass blocks a steady input, so its input weights cancel out.
    /// let band_pass = BiquadCoefficients::band_pass(180.0_f64, 4.0, 0.001).unwrap();
    /// let weights = band_pass.feed_forward();
    /// assert!((weights[0] + weights[1] + weights[2]).abs() < 1e-12);
    /// ```
    pub fn band_pass(center_hz: T, quality_factor: T, timestep: T) -> Result<Self, SignalError> {
        Self::build(Design::BandPass, center_hz, quality_factor, timestep)
    }

    /// Builds a notch, which removes a narrow band of frequencies around the centre and leaves
    /// everything to either side of it alone.
    ///
    /// Returns [`SignalError::NonFinite`] if any argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `timestep` is not strictly positive,
    /// [`SignalError::NonPositiveQualityFactor`] if `quality_factor` is not strictly positive, or
    /// [`SignalError::FrequencyOutOfRange`] if `center_hz` is not strictly positive or reaches half
    /// the sampling rate.
    ///
    /// ```
    /// use multicalc::signal_processing::BiquadCoefficients;
    ///
    /// let notch = BiquadCoefficients::notch(180.0_f64, 4.0, 0.001).unwrap();
    /// assert_eq!(notch.timestep(), 0.001);
    /// ```
    pub fn notch(center_hz: T, quality_factor: T, timestep: T) -> Result<Self, SignalError> {
        Self::build(Design::Notch, center_hz, quality_factor, timestep)
    }

    /// The weights on the newest input sample and the two before it.
    #[inline]
    #[must_use]
    pub fn feed_forward(&self) -> [T; 3] {
        self.feed_forward
    }

    /// The weights on the two previous outputs, with the sign they have when written on the
    /// left-hand side.
    #[inline]
    #[must_use]
    pub fn feedback(&self) -> [T; 2] {
        self.feedback
    }

    /// The seconds between samples this filter was designed for.
    #[inline]
    #[must_use]
    pub fn timestep(&self) -> T {
        self.timestep
    }

    /// How much of a steady oscillation at this frequency comes through, as a multiple of what
    /// went in. One means untouched, and a low-pass is down to about seven tenths at its cutoff.
    ///
    /// A negative frequency reports the same magnitude as its positive counterpart — a real
    /// filter's response is mirrored around zero, so there is nothing new to measure there.
    ///
    /// Returns [`SignalError::NonFinite`] if `frequency_hz` is not finite, or
    /// [`SignalError::FrequencyOutOfRange`] if its magnitude reaches half the sampling rate — past
    /// there, on either side of zero, the reading has aliased and no longer describes that
    /// frequency.
    ///
    /// ```
    /// use multicalc::signal_processing::BiquadCoefficients;
    ///
    /// let low_pass = BiquadCoefficients::low_pass(50.0_f64, 0.70710678, 0.001).unwrap();
    ///
    /// // At the cutoff, about seven tenths of the input survives.
    /// assert!((low_pass.magnitude_at(50.0).unwrap() - 1.0 / 2.0_f64.sqrt()).abs() < 0.02);
    ///
    /// // Well above it, almost nothing does.
    /// assert!(low_pass.magnitude_at(400.0).unwrap() < 0.05);
    ///
    /// // A negative frequency mirrors its positive counterpart.
    /// assert!(
    ///     (low_pass.magnitude_at(-50.0).unwrap() - low_pass.magnitude_at(50.0).unwrap()).abs()
    ///         < 1e-12
    /// );
    /// ```
    pub fn magnitude_at(&self, frequency_hz: T) -> Result<T, SignalError> {
        Self::check_frequency(frequency_hz, self.timestep)?;
        let (input_real, input_imaginary, output_real, output_imaginary) =
            self.response_parts(frequency_hz);
        Ok(input_real.hypot(input_imaginary) / output_real.hypot(output_imaginary))
    }

    /// The same figure as [`magnitude_at`](Self::magnitude_at), in decibels: zero means untouched
    /// and negative means reduced.
    ///
    /// A frequency the filter removes completely reports negative infinity, which is what a
    /// notch's centre gives.
    ///
    /// Returns the same errors as [`magnitude_at`](Self::magnitude_at).
    pub fn magnitude_in_decibels_at(&self, frequency_hz: T) -> Result<T, SignalError> {
        Ok(T::from_f64(20.0 / core::f64::consts::LN_10) * self.magnitude_at(frequency_hz)?.log())
    }

    /// How far a steady oscillation at this frequency is shifted along, in radians. Negative means
    /// the output trails the input.
    ///
    /// A notch's phase jumps by half a turn as the frequency crosses its centre, which is what the
    /// filter really does rather than an artifact of the calculation. At the centre itself there is
    /// no output left to have a phase, so the figure reported there is whatever the leftover
    /// rounding in a near-zero response happens to give — it carries no information despite looking
    /// like a number.
    ///
    /// A negative frequency reports the negative of its positive counterpart's phase, the mirror
    /// image [`magnitude_at`](Self::magnitude_at) has around zero.
    ///
    /// Returns the same errors as [`magnitude_at`](Self::magnitude_at).
    pub fn phase_at(&self, frequency_hz: T) -> Result<T, SignalError> {
        Self::check_frequency(frequency_hz, self.timestep)?;
        let (input_real, input_imaginary, output_real, output_imaginary) =
            self.response_parts(frequency_hz);
        // Two separate arctangents, subtracted. Each lands on the turn its own half of the
        // response belongs to, so the difference stays right even when it runs past half a turn.
        // One arctangent of the combined ratio would silently wrap instead.
        Ok(input_imaginary.atan2(input_real) - output_imaginary.atan2(output_real))
    }

    /// How far behind the input a steady oscillation at this frequency comes out, in seconds.
    ///
    /// This is the number that eats a control loop's stability margin, so it is worth checking at
    /// the frequency the loop crosses over. A frequency of zero reports zero.
    ///
    /// A negative frequency reports the same delay as its positive counterpart: the phase flips
    /// sign there, but so does the frequency dividing it, and the two cancel.
    ///
    /// Returns the same errors as [`magnitude_at`](Self::magnitude_at); zero is always accepted
    /// regardless.
    ///
    /// ```
    /// use multicalc::signal_processing::BiquadCoefficients;
    ///
    /// // A 50 Hz low-pass sampled every millisecond puts its own cutoff about five
    /// // milliseconds behind.
    /// let low_pass = BiquadCoefficients::low_pass(50.0_f64, 0.70710678, 0.001).unwrap();
    /// assert!(low_pass.delay_at(50.0).unwrap() > 0.0);
    /// assert!(low_pass.delay_at(50.0).unwrap() < 0.01);
    /// ```
    pub fn delay_at(&self, frequency_hz: T) -> Result<T, SignalError> {
        if frequency_hz == T::ZERO {
            return Ok(T::ZERO);
        }
        Ok(-self.phase_at(frequency_hz)? / (T::TWO * T::PI * frequency_hz))
    }

    /// Whether the filter settles rather than growing without bound.
    ///
    /// Anything from one of the four design functions is always stable; this is for weights handed
    /// in directly.
    ///
    /// ```
    /// use multicalc::signal_processing::BiquadCoefficients;
    ///
    /// assert!(BiquadCoefficients::low_pass(50.0_f64, 0.70710678, 0.001).unwrap().is_stable());
    ///
    /// // Feeding back more than the whole of the previous output makes it grow every step.
    /// let runaway = BiquadCoefficients::new([1.0_f64, 0.0, 0.0], [0.0, 1.5], 0.001).unwrap();
    /// assert!(!runaway.is_stable());
    /// ```
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.feedback[1].abs() < T::ONE && self.feedback[0].abs() < T::ONE + self.feedback[1]
    }

    /// The response at one frequency, as the real and imaginary part of the input side followed by
    /// the real and imaginary part of the output side.
    #[must_use]
    fn response_parts(&self, frequency_hz: T) -> (T, T, T, T) {
        let angle = T::TWO * T::PI * frequency_hz * self.timestep;
        let cosine = angle.cos();
        let sine = angle.sin();
        let double_cosine = (T::TWO * angle).cos();
        let double_sine = (T::TWO * angle).sin();

        // Each weight is named for the sample it multiplies.
        let [newest_input, previous_input, earlier_input] = self.feed_forward;
        let [previous_output, earlier_output] = self.feedback;

        (
            newest_input + previous_input * cosine + earlier_input * double_cosine,
            -(previous_input * sine + earlier_input * double_sine),
            T::ONE + previous_output * cosine + earlier_output * double_cosine,
            -(previous_output * sine + earlier_output * double_sine),
        )
    }

    /// Checks a frequency passed to a response query: rejects one that is not finite, and one
    /// whose magnitude reaches the Nyquist frequency `timestep` implies. A negative frequency
    /// mirrors its positive counterpart (see [`magnitude_at`](Self::magnitude_at)), so the limit
    /// is on its absolute value, the same way [`Self::check_design`] already limits a design's own
    /// cutoff or centre on the positive side.
    fn check_frequency(frequency_hz: T, timestep: T) -> Result<(), SignalError> {
        if !frequency_hz.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if frequency_hz.abs() * timestep >= T::HALF {
            return Err(SignalError::FrequencyOutOfRange);
        }
        Ok(())
    }

    /// Checks the arguments every design function shares.
    fn check_design(frequency_hz: T, quality_factor: T, timestep: T) -> Result<(), SignalError> {
        if !frequency_hz.is_finite() || !quality_factor.is_finite() || !timestep.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if timestep <= T::ZERO {
            return Err(SignalError::NonPositiveTimestep);
        }
        if quality_factor <= T::ZERO {
            return Err(SignalError::NonPositiveQualityFactor);
        }
        if frequency_hz <= T::ZERO || frequency_hz * timestep >= T::HALF {
            return Err(SignalError::FrequencyOutOfRange);
        }
        Ok(())
    }

    /// Checks the arguments every design shares, then builds one of the four shapes from them.
    fn build(
        design: Design,
        frequency_hz: T,
        quality_factor: T,
        timestep: T,
    ) -> Result<Self, SignalError> {
        Self::check_design(frequency_hz, quality_factor, timestep)?;

        let angle = T::TWO * T::PI * frequency_hz * timestep;
        let cosine = angle.cos();

        // The sharpness means a different thing in each pair, so each pair gets its own formula
        // and they are not interchangeable. For the low- and high-pass it says how heavily the
        // filter is damped; for the band-pass and notch it says how wide a band is affected.
        // Using one formula for all four moves the band-pass and notch weights by about 0.02,
        // which is a different filter rather than a rounding difference.
        let alpha = match design {
            Design::LowPass | Design::HighPass => angle.sin() / (T::TWO * quality_factor),
            Design::BandPass | Design::Notch => (angle / (T::TWO * quality_factor)).tan(),
        };

        let feed_forward = match design {
            Design::LowPass => {
                let shared = T::ONE - cosine;
                [shared * T::HALF, shared, shared * T::HALF]
            }
            Design::HighPass => {
                let shared = T::ONE + cosine;
                [shared * T::HALF, -shared, shared * T::HALF]
            }
            Design::BandPass => [alpha, T::ZERO, -alpha],
            Design::Notch => [T::ONE, -(T::TWO * cosine), T::ONE],
        };

        Ok(Self::from_unnormalized(
            feed_forward,
            [T::ONE + alpha, -(T::TWO * cosine), T::ONE - alpha],
            timestep,
        ))
    }

    /// Divides the six raw weights through by the leading output weight and keeps the five that
    /// are left. [`Self::build`] reaches this only after its arguments pass
    /// [`Self::check_design`], which leaves the divisor above one.
    #[must_use]
    fn from_unnormalized(feed_forward: [T; 3], feedback: [T; 3], timestep: T) -> Self {
        let leading = feedback[0];
        Self {
            feed_forward: [
                feed_forward[0] / leading,
                feed_forward[1] / leading,
                feed_forward[2] / leading,
            ],
            feedback: [feedback[1] / leading, feedback[2] / leading],
            timestep,
        }
    }
}

/// A second-order filter running on a stream of samples.
///
/// It starts at rest, so the first outputs settle towards the input rather than tracking it
/// straight away. Calling [`settle_to`](Self::settle_to) first skips that.
///
/// The weights can be replaced part-way through with [`set_coefficients`](Self::set_coefficients),
/// which leaves the filter's memory of recent samples alone — so a notch can follow a frequency
/// that moves without a step in the output.
///
/// ```
/// use multicalc::signal_processing::{Biquad, BiquadCoefficients};
///
/// // A notch on 180 Hz, sampled every millisecond, removes an oscillation at that frequency.
/// let mut filter = Biquad::new(BiquadCoefficients::notch(180.0_f64, 4.0, 0.001).unwrap());
/// let mut last_outputs = [0.0; 500];
/// for sample in 0..2000 {
///     let angle = 2.0 * core::f64::consts::PI * 180.0 * f64::from(sample) / 1000.0;
///     let output = filter.filter(angle.sin());
///     if sample >= 1500 {
///         last_outputs[(sample - 1500) as usize] = output;
///     }
/// }
/// assert!(last_outputs.iter().all(|output| output.abs() < 0.05));
/// ```
///
/// Swapping the weights keeps the memory, so the output carries on from where it was:
///
/// ```
/// use multicalc::signal_processing::{Biquad, BiquadCoefficients};
///
/// // A notch passes a steady input through, so this settles on 1.
/// let mut running = Biquad::new(BiquadCoefficients::notch(180.0_f64, 4.0, 0.001).unwrap());
/// for _ in 0..1000 {
///     let _ = running.filter(1.0);
/// }
///
/// // Moving the notch to 210 Hz barely disturbs the output.
/// let moved = BiquadCoefficients::notch(210.0_f64, 4.0, 0.001).unwrap();
/// running.set_coefficients(moved);
/// assert!((running.filter(1.0) - 1.0).abs() < 0.03);
///
/// // A filter built from scratch has no memory, so its first output is well short of 1.
/// let mut fresh = Biquad::new(moved);
/// assert!((fresh.filter(1.0) - 1.0).abs() > 0.05);
/// ```
///
/// The `Biquad` filter has another checked entry point for cases where input could be non-finite.
/// The checked entry point prevents a non-finite input to spoil the filter state.
///
/// ```
/// use multicalc::signal_processing::{Biquad, BiquadCoefficients};
///
/// let mut running = Biquad::new(BiquadCoefficients::notch(180.0_f64, 4.0, 0.001).unwrap());
/// let test_inputs = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
///
/// let _ = running.filter(1.0);
/// let running_snapshot = running;
///
/// for signal in test_inputs {
///     let output = running.filter_checked(signal);
///     assert!(output.is_err())
/// }
///
/// // The filter is not spoiled
/// assert_eq!(running, running_snapshot);
///
/// let output = running.filter_checked(0.1_f64);
/// assert!(output.is_ok());
/// assert!(output.unwrap().is_finite());
///
/// // NaN spoils the filter ..
/// let _ = running.filter(f64::NAN);
/// assert!(running.filter(1.0).is_nan());
///
/// //.. till reset
/// running.reset();
/// assert!(running.filter(1.0).is_finite());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Biquad<T: Numeric = f64> {
    coefficients: BiquadCoefficients<T>,
    first_state: T,
    second_state: T,
    last_output: T,
}

impl<T: Numeric> Biquad<T> {
    /// Builds a filter at rest from a set of weights.
    ///
    /// This cannot fail: the weights were checked when they were designed.
    #[must_use]
    pub fn new(coefficients: BiquadCoefficients<T>) -> Self {
        Self {
            coefficients,
            first_state: T::ZERO,
            second_state: T::ZERO,
            last_output: T::ZERO,
        }
    }

    /// Feeds one sample and returns the filtered output.
    /// Non-finite input spoils the filter till it is reset,
    /// and should be handled by `filter_checked` instead.
    #[inline]
    #[must_use]
    pub fn filter(&mut self, input: T) -> T {
        let feed_forward = self.coefficients.feed_forward();
        let feedback = self.coefficients.feedback();

        // Each line uses the one above it, so the order matters.
        let output = feed_forward[0] * input + self.first_state;
        self.first_state = feed_forward[1] * input - feedback[0] * output + self.second_state;
        self.second_state = feed_forward[2] * input - feedback[1] * output;
        self.last_output = output;
        output
    }

    /// Alternative to `filter` with checked input.
    /// Non-finite input cannot spoils the filter.
    /// Return `SignalError::NonFinite` in case of non-finite input.
    #[inline]
    pub fn filter_checked(&mut self, input: T) -> Result<T, SignalError> {
        if input.is_finite() {
            Ok(self.filter(input))
        } else {
            Err(SignalError::NonFinite)
        }
    }

    /// Replaces the weights and keeps the memory of recent samples.
    #[inline]
    pub fn set_coefficients(&mut self, coefficients: BiquadCoefficients<T>) {
        self.coefficients = coefficients;
    }

    /// Puts the filter where it would sit after a long run of `value`, so a loop does not open
    /// with a settling period.
    ///
    /// A high-pass or band-pass settles to zero whatever the value is, since neither passes a
    /// steady input. Weights whose output side sums to zero would divide by zero here; none of the
    /// four designs produces them.
    pub fn settle_to(&mut self, value: T) {
        let feed_forward = self.coefficients.feed_forward();
        let feedback = self.coefficients.feedback();

        let steady = value * (feed_forward[0] + feed_forward[1] + feed_forward[2])
            / (T::ONE + feedback[0] + feedback[1]);
        self.second_state = feed_forward[2] * value - feedback[1] * steady;
        self.first_state = feed_forward[1] * value - feedback[0] * steady + self.second_state;
        self.last_output = steady;
    }

    /// Clears the memory of recent samples, putting the filter back at rest.
    #[inline]
    pub fn reset(&mut self) {
        self.first_state = T::ZERO;
        self.second_state = T::ZERO;
        self.last_output = T::ZERO;
    }

    /// The weights the filter is running with.
    #[inline]
    #[must_use]
    pub fn coefficients(&self) -> BiquadCoefficients<T> {
        self.coefficients
    }

    /// The most recent output, without feeding a sample.
    #[inline]
    #[must_use]
    pub fn value(&self) -> T {
        self.last_output
    }
}
