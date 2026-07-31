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
    dt: T,
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
    /// Returns [`SignalError::NonFinite`] if any weight or `dt` is not finite, or
    /// [`SignalError::NonPositiveTimestep`] if `dt` is not strictly positive.
    pub fn new(feed_forward: [T; 3], feedback: [T; 2], dt: T) -> Result<Self, SignalError> {
        for weight in feed_forward.into_iter().chain(feedback) {
            if !weight.is_finite() {
                return Err(SignalError::NonFinite);
            }
        }
        if !dt.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if dt <= T::ZERO {
            return Err(SignalError::NonPositiveTimestep);
        }
        Ok(Self {
            feed_forward,
            feedback,
            dt,
        })
    }

    /// Builds a low-pass, which keeps content below the cutoff and fades out what is above it.
    ///
    /// Returns [`SignalError::NonFinite`] if any argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `dt` is not strictly positive,
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
    pub fn low_pass(cutoff_hz: T, quality_factor: T, dt: T) -> Result<Self, SignalError> {
        Self::build(Design::LowPass, cutoff_hz, quality_factor, dt)
    }

    /// Builds a high-pass, which keeps content above the cutoff and fades out what is below it.
    ///
    /// Returns [`SignalError::NonFinite`] if any argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `dt` is not strictly positive,
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
    pub fn high_pass(cutoff_hz: T, quality_factor: T, dt: T) -> Result<Self, SignalError> {
        Self::build(Design::HighPass, cutoff_hz, quality_factor, dt)
    }

    /// Builds a band-pass, which keeps a band of frequencies around the centre and fades out
    /// everything to either side of it.
    ///
    /// Returns [`SignalError::NonFinite`] if any argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `dt` is not strictly positive,
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
    pub fn band_pass(center_hz: T, quality_factor: T, dt: T) -> Result<Self, SignalError> {
        Self::build(Design::BandPass, center_hz, quality_factor, dt)
    }

    /// Builds a notch, which removes a narrow band of frequencies around the centre and leaves
    /// everything to either side of it alone.
    ///
    /// Returns [`SignalError::NonFinite`] if any argument is not finite,
    /// [`SignalError::NonPositiveTimestep`] if `dt` is not strictly positive,
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
    pub fn notch(center_hz: T, quality_factor: T, dt: T) -> Result<Self, SignalError> {
        Self::build(Design::Notch, center_hz, quality_factor, dt)
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
        self.dt
    }

    /// How much of a steady oscillation at this frequency comes through, as a multiple of what
    /// went in. One means untouched, and a low-pass is down to about seven tenths at its cutoff.
    ///
    /// ```
    /// use multicalc::signal_processing::BiquadCoefficients;
    ///
    /// let low_pass = BiquadCoefficients::low_pass(50.0_f64, 0.70710678, 0.001).unwrap();
    ///
    /// // At the cutoff, about seven tenths of the input survives.
    /// assert!((low_pass.magnitude_at(50.0) - 1.0 / 2.0_f64.sqrt()).abs() < 0.02);
    ///
    /// // Well above it, almost nothing does.
    /// assert!(low_pass.magnitude_at(400.0) < 0.05);
    /// ```
    #[must_use]
    pub fn magnitude_at(&self, frequency_hz: T) -> T {
        let (input_real, input_imaginary, output_real, output_imaginary) =
            self.response_parts(frequency_hz);
        input_real.hypot(input_imaginary) / output_real.hypot(output_imaginary)
    }

    /// The same figure as [`magnitude_at`](Self::magnitude_at), in decibels: zero means untouched
    /// and negative means reduced.
    ///
    /// A frequency the filter removes completely reports negative infinity, which is what a
    /// notch's centre gives.
    #[must_use]
    pub fn magnitude_in_decibels_at(&self, frequency_hz: T) -> T {
        T::from_f64(20.0 / core::f64::consts::LN_10) * self.magnitude_at(frequency_hz).ln()
    }

    /// How far a steady oscillation at this frequency is shifted along, in radians. Negative means
    /// the output trails the input.
    ///
    /// A notch's phase jumps by half a turn as the frequency crosses its centre, which is what the
    /// filter really does rather than an artifact of the calculation. At the centre itself there is
    /// no output left to have a phase, so the figure reported there is whatever the leftover
    /// rounding in a near-zero response happens to give — it carries no information despite looking
    /// like a number.
    #[must_use]
    pub fn phase_at(&self, frequency_hz: T) -> T {
        let (input_real, input_imaginary, output_real, output_imaginary) =
            self.response_parts(frequency_hz);
        // Two separate arctangents, subtracted. Each lands on the turn its own half of the
        // response belongs to, so the difference stays right even when it runs past half a turn.
        // One arctangent of the combined ratio would silently wrap instead.
        input_imaginary.atan2(input_real) - output_imaginary.atan2(output_real)
    }

    /// How far behind the input a steady oscillation at this frequency comes out, in seconds.
    ///
    /// This is the number that eats a control loop's stability margin, so it is worth checking at
    /// the frequency the loop crosses over. A frequency of zero reports zero.
    ///
    /// ```
    /// use multicalc::signal_processing::BiquadCoefficients;
    ///
    /// // A 50 Hz low-pass sampled every millisecond puts its own cutoff about five
    /// // milliseconds behind.
    /// let low_pass = BiquadCoefficients::low_pass(50.0_f64, 0.70710678, 0.001).unwrap();
    /// assert!(low_pass.delay_at(50.0) > 0.0);
    /// assert!(low_pass.delay_at(50.0) < 0.01);
    /// ```
    #[must_use]
    pub fn delay_at(&self, frequency_hz: T) -> T {
        if frequency_hz == T::ZERO {
            return T::ZERO;
        }
        -self.phase_at(frequency_hz) / (T::TWO * T::PI * frequency_hz)
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
    fn response_parts(&self, frequency_hz: T) -> (T, T, T, T) {
        let angle = T::TWO * T::PI * frequency_hz * self.dt;
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

    /// Checks the arguments every design function shares.
    fn check_design(frequency_hz: T, quality_factor: T, dt: T) -> Result<(), SignalError> {
        if !frequency_hz.is_finite() || !quality_factor.is_finite() || !dt.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if dt <= T::ZERO {
            return Err(SignalError::NonPositiveTimestep);
        }
        if quality_factor <= T::ZERO {
            return Err(SignalError::NonPositiveQualityFactor);
        }
        if frequency_hz <= T::ZERO || frequency_hz * dt >= T::HALF {
            return Err(SignalError::FrequencyOutOfRange);
        }
        Ok(())
    }

    /// Checks the arguments every design shares, then builds one of the four shapes from them.
    fn build(
        design: Design,
        frequency_hz: T,
        quality_factor: T,
        dt: T,
    ) -> Result<Self, SignalError> {
        Self::check_design(frequency_hz, quality_factor, dt)?;

        let angle = T::TWO * T::PI * frequency_hz * dt;
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
            dt,
        ))
    }

    /// Divides the six raw weights through by the leading output weight and keeps the five that
    /// are left. [`Self::build`] reaches this only after its arguments pass
    /// [`Self::check_design`], which leaves the divisor above one.
    fn from_unnormalized(feed_forward: [T; 3], feedback: [T; 3], dt: T) -> Self {
        let leading = feedback[0];
        Self {
            feed_forward: [
                feed_forward[0] / leading,
                feed_forward[1] / leading,
                feed_forward[2] / leading,
            ],
            feedback: [feedback[1] / leading, feedback[2] / leading],
            dt,
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
///     running.filter(1.0);
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
    pub fn new(coefficients: BiquadCoefficients<T>) -> Self {
        Self {
            coefficients,
            first_state: T::ZERO,
            second_state: T::ZERO,
            last_output: T::ZERO,
        }
    }

    /// Feeds one sample and returns the filtered output.
    #[inline]
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
