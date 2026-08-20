//! Several second-order filters in series, and filtering each channel of a vector.

use crate::error::SignalError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;
use crate::signal_processing::{Biquad, BiquadCoefficients};

/// Several second-order filters in series, the output of each feeding the next.
///
/// Two low-passes in a row cut twice as steeply as one, and a row of notches removes a frequency
/// along with the whole-number multiples above it.
///
/// ```
/// use multicalc::signal_processing::{BiquadCascade, BiquadCoefficients};
///
/// let section = BiquadCoefficients::low_pass(50.0_f64, 0.70710678, 0.001).unwrap();
/// let mut cascade = BiquadCascade::new([section; 2]);
///
/// // Two sections cut far deeper than one: a single section leaves 0.003 of a 400 Hz
/// // oscillation, two leave under a hundredth of that.
/// assert!(cascade.magnitude_at(400.0) < 0.001);
///
/// // A steady input still comes through untouched.
/// for _ in 0..500 {
///     let _ = cascade.filter(5.0);
/// }
/// assert!((cascade.value() - 5.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiquadCascade<const SECTIONS: usize, T: Numeric = f64> {
    sections: [Biquad<T>; SECTIONS],
}

impl<const SECTIONS: usize, T: Numeric> BiquadCascade<SECTIONS, T> {
    /// Builds a cascade at rest, one section per set of weights.
    #[must_use]
    pub fn new(coefficients: [BiquadCoefficients<T>; SECTIONS]) -> Self {
        Self {
            sections: core::array::from_fn(|index| Biquad::new(coefficients[index])),
        }
    }

    /// Feeds one sample through every section in order and returns what comes out the far end.
    #[inline]
    #[must_use]
    pub fn filter(&mut self, input: T) -> T {
        let mut value = input;
        for section in &mut self.sections {
            value = section.filter(value);
        }
        value
    }

    /// Replaces one section's weights, keeping that section's memory of recent samples.
    ///
    /// Returns [`SignalError::SectionIndexOutOfRange`] if there is no section at `index`.
    pub fn set_section(
        &mut self,
        index: usize,
        coefficients: BiquadCoefficients<T>,
    ) -> Result<(), SignalError> {
        if index >= SECTIONS {
            return Err(SignalError::SectionIndexOutOfRange);
        }
        self.sections[index].set_coefficients(coefficients);
        Ok(())
    }

    /// Puts the whole chain where a long run of `value` would leave it.
    ///
    /// Each section is settled to what the section before it settles on, rather than all of them
    /// to the original input, so the chain agrees with itself from the first sample.
    pub fn settle_to(&mut self, value: T) {
        let mut carried = value;
        for section in &mut self.sections {
            section.settle_to(carried);
            carried = section.value();
        }
    }

    /// Clears every section's memory, putting the chain back at rest.
    pub fn reset(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }

    /// How much of a steady oscillation at this frequency comes through the whole chain, as a
    /// multiple of what went in.
    #[must_use]
    pub fn magnitude_at(&self, frequency_hz: T) -> T {
        self.sections.iter().fold(T::ONE, |carried, section| {
            carried * section.coefficients().magnitude_at(frequency_hz)
        })
    }

    /// The same figure as [`magnitude_at`](Self::magnitude_at), in decibels: zero means untouched
    /// and negative means reduced.
    ///
    /// A frequency any one section removes completely reports negative infinity.
    #[must_use]
    pub fn magnitude_in_decibels_at(&self, frequency_hz: T) -> T {
        T::from_f64(20.0 / core::f64::consts::LN_10) * self.magnitude_at(frequency_hz).log()
    }

    /// How far a steady oscillation at this frequency is shifted along by the whole chain, in
    /// radians. Negative means the output trails the input.
    #[must_use]
    pub fn phase_at(&self, frequency_hz: T) -> T {
        // Adding the sections' phases keeps the total right even when it runs past half a turn,
        // because each section's figure already sits on the turn it belongs to.
        self.sections.iter().fold(T::ZERO, |carried, section| {
            carried + section.coefficients().phase_at(frequency_hz)
        })
    }

    /// How far behind the input a steady oscillation at this frequency comes out of the whole
    /// chain, in seconds. A frequency of zero reports zero.
    #[must_use]
    pub fn delay_at(&self, frequency_hz: T) -> T {
        if frequency_hz == T::ZERO {
            return T::ZERO;
        }
        -self.phase_at(frequency_hz) / (T::TWO * T::PI * frequency_hz)
    }

    /// Whether every section settles rather than growing without bound.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.sections
            .iter()
            .all(|section| section.coefficients().is_stable())
    }

    /// The most recent output of the last section, without feeding a sample.
    #[must_use]
    pub fn value(&self) -> T {
        self.sections.last().map_or(T::ZERO, Biquad::value)
    }
}

/// One filter shape applied to every component of a vector, each keeping its own memory.
///
/// This is what a three-axis rate sensor wants: one object, one call per tick.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// use multicalc::signal_processing::{BiquadCoefficients, MultiChannelBiquad};
///
/// let shape = BiquadCoefficients::low_pass(50.0_f64, 0.70710678, 0.001).unwrap();
/// let mut filter = MultiChannelBiquad::new(shape);
///
/// // Three axes held steady come through untouched, each on its own.
/// let reading = Vector::new([1.0, -2.0, 0.5]);
/// for _ in 0..500 {
///     let _ = filter.filter(reading);
/// }
/// let settled = filter.value();
/// for channel in 0..3 {
///     assert!((settled[channel] - reading[channel]).abs() < 1e-6);
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiChannelBiquad<const CHANNELS: usize, T: Numeric = f64> {
    channels: [Biquad<T>; CHANNELS],
}

impl<const CHANNELS: usize, T: Numeric> MultiChannelBiquad<CHANNELS, T> {
    /// Builds a filter at rest, giving every channel the same shape and its own memory.
    #[must_use]
    pub fn new(coefficients: BiquadCoefficients<T>) -> Self {
        Self {
            channels: core::array::from_fn(|_| Biquad::new(coefficients)),
        }
    }

    /// Feeds one sample per channel and returns the filtered output of each.
    #[inline]
    pub fn filter(&mut self, input: Vector<CHANNELS, T>) -> Vector<CHANNELS, T> {
        Vector::from_fn(|channel| self.channels[channel].filter(input[channel]))
    }

    /// Replaces the shape on every channel, keeping every channel's memory of recent samples.
    pub fn set_coefficients(&mut self, coefficients: BiquadCoefficients<T>) {
        for channel in &mut self.channels {
            channel.set_coefficients(coefficients);
        }
    }

    /// Puts each channel where a long run of its own component of `value` would leave it.
    pub fn settle_to(&mut self, value: Vector<CHANNELS, T>) {
        for (channel, filter) in self.channels.iter_mut().enumerate() {
            filter.settle_to(value[channel]);
        }
    }

    /// Clears every channel's memory, putting the filter back at rest.
    pub fn reset(&mut self) {
        for channel in &mut self.channels {
            channel.reset();
        }
    }

    /// The most recent output of every channel, without feeding a sample.
    pub fn value(&self) -> Vector<CHANNELS, T> {
        Vector::from_fn(|channel| self.channels[channel].value())
    }
}

/// Notch weights for a frequency and the whole-number multiples above it, one section each.
///
/// Section 0 sits on the given frequency, section 1 on twice it, and so on — which is the shape a
/// spinning motor's vibration takes.
///
/// Returns [`SignalError::WindowTooShort`] if no sections were asked for, and
/// [`SignalError::FrequencyOutOfRange`] if the highest multiple would reach half the sampling rate,
/// so a caller retuning as a motor speeds up finds out rather than getting a filter that does
/// something else. The remaining error paths are those of
/// [`BiquadCoefficients::notch`](crate::signal_processing::BiquadCoefficients::notch).
///
/// ```
/// use multicalc::error::SignalError;
/// use multicalc::signal_processing::{harmonic_notch_coefficients, BiquadCascade};
///
/// // 80 Hz and its next two multiples, sampled every millisecond.
/// let sections = harmonic_notch_coefficients::<3, f64>(80.0, 4.0, 0.001).unwrap();
/// let cascade = BiquadCascade::new(sections);
///
/// assert!(cascade.magnitude_at(80.0) < 0.05);
/// assert!(cascade.magnitude_at(160.0) < 0.05);
/// assert!(cascade.magnitude_at(240.0) < 0.05);
///
/// // Frequencies away from the notches come through.
/// assert!(cascade.magnitude_at(20.0) > 0.9);
/// assert!(cascade.magnitude_at(120.0) > 0.7);
///
/// // Three sections on 180 Hz would need a notch at 540 Hz, past half of a 1 kHz sampling rate.
/// assert_eq!(
///     harmonic_notch_coefficients::<3, f64>(180.0, 4.0, 0.001),
///     Err(SignalError::FrequencyOutOfRange)
/// );
/// ```
pub fn harmonic_notch_coefficients<const SECTIONS: usize, T: Numeric>(
    fundamental_hz: T,
    quality_factor: T,
    timestep: T,
) -> Result<[BiquadCoefficients<T>; SECTIONS], SignalError> {
    if SECTIONS == 0 {
        return Err(SignalError::WindowTooShort);
    }

    // Check the lowest and the highest section before building anything, so a caller that is too
    // close to half the sampling rate hears about it once rather than part-way through.
    let fundamental = BiquadCoefficients::notch(fundamental_hz, quality_factor, timestep)?;
    if fundamental_hz * T::from_usize(SECTIONS) * timestep >= T::HALF {
        return Err(SignalError::FrequencyOutOfRange);
    }

    let mut sections = [fundamental; SECTIONS];
    for (index, section) in sections.iter_mut().enumerate().skip(1) {
        *section = BiquadCoefficients::notch(
            fundamental_hz * T::from_usize(index + 1),
            quality_factor,
            timestep,
        )?;
    }
    Ok(sections)
}
