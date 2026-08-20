//! Smoothing by fitting a small curve across a window, which also gives clean rates.

use crate::error::SignalError;
use crate::linear_algebra::Matrix;
use crate::scalar::Numeric;

/// Smooths a stream by fitting a small curve across the last few samples, and reports that curve's
/// slope and bend as well — so one noisy position signal gives a usable rate and acceleration.
///
/// `POLYNOMIAL_TERMS` is how many terms the curve has: two fits a straight line, three fits a
/// curve. Fewer terms smooth harder; more terms follow the data more closely. The window has to be
/// at least as long as the number of terms.
///
/// Two ways to read the answer. [`latest`](Self::latest) reads it at the newest sample, so the
/// output describes the sample just fed in and adds no delay — the arrangement a live loop needs.
/// [`centered`](Self::centered) reads it at the middle of the window, which smooths considerably
/// better but means **the output describes a sample from half a window ago**: at a window of 11 and
/// a 1 kHz rate, five milliseconds behind. [`delay`](Self::delay) reports that lag in seconds so it
/// can be accounted for.
///
/// The bend is much less precise than the value or the slope, and at `f32` it is good to only about
/// two digits. Working it out means adding up numbers thousands of times larger than the answer, so
/// most of the precision cancels away — and worse the smaller the timestep, which enters squared.
/// Take the bend at `f64`, or over a longer timestep, wherever it has to be right.
///
/// ```
/// use multicalc::error::SignalError;
/// use multicalc::signal_processing::SavitzkyGolay;
///
/// // Three terms fit a curve exactly, so a curve is reproduced with its slope and bend.
/// let timestep = 0.001_f64;
/// let mut fitted = SavitzkyGolay::<11, 3, f64>::latest(timestep).unwrap();
/// let mut time = 0.0;
/// for sample in 0..200 {
///     time = sample as f64 * timestep;
///     let _ = fitted.filter(0.5 * time * time);
/// }
/// assert!((fitted.first_derivative() - time).abs() < 1e-6);
/// assert!((fitted.second_derivative() - 1.0).abs() < 1e-6);
///
/// // Reading at the newest sample costs no delay; reading at the middle costs half a window.
/// assert_eq!(fitted.delay(), 0.0);
/// assert!((SavitzkyGolay::<11, 3, f64>::centered(timestep).unwrap().delay() - 0.005).abs() < 1e-12);
///
/// // A window too short to fit that many terms is rejected.
/// assert_eq!(
///     SavitzkyGolay::<3, 5, f64>::latest(timestep),
///     Err(SignalError::PolynomialOrderTooHigh)
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SavitzkyGolay<const WINDOW: usize, const POLYNOMIAL_TERMS: usize, T: Numeric = f64> {
    /// The last WINDOW samples, oldest to newest by position modulo the write index.
    samples: [T; WINDOW],
    /// Where the next sample goes.
    next: usize,
    /// Whether the first sample has seeded the window.
    initialized: bool,
    /// What to multiply each sample by to get the smoothed value.
    smoothing_weights: [T; WINDOW],
    /// The same for the slope, already divided through by the timestep.
    first_derivative_weights: [T; WINDOW],
    /// The same for the bend, already divided through by the timestep squared.
    second_derivative_weights: [T; WINDOW],
    /// How far behind the newest sample the answer sits.
    delay_seconds: T,
}

impl<const WINDOW: usize, const POLYNOMIAL_TERMS: usize, T: Numeric>
    SavitzkyGolay<WINDOW, POLYNOMIAL_TERMS, T>
{
    /// Builds a filter reading its answer at the middle of the window: the best smoothing on
    /// offer, at the cost of describing a sample from half a window ago.
    ///
    /// Returns [`SignalError::WindowEvenLength`] if the window length is even, since there would be
    /// no middle sample to read at. The other error paths are [`Self::latest`]'s.
    pub fn centered(timestep: T) -> Result<Self, SignalError> {
        if WINDOW % 2 == 0 {
            return Err(SignalError::WindowEvenLength);
        }
        Self::build(timestep, (WINDOW - 1) / 2)
    }

    /// Builds a filter reading its answer at the newest sample, so the output describes the sample
    /// just fed in and adds no delay.
    ///
    /// Returns [`SignalError::NonFinite`] if `timestep` is not finite,
    /// [`SignalError::NonPositiveTimestep`] if it is not strictly positive,
    /// [`SignalError::WindowTooShort`] if the window or the number of terms is zero,
    /// [`SignalError::PolynomialOrderTooHigh`] if there are more terms than samples to fit them,
    /// and [`SignalError::Linalg`] if the fit cannot be worked out.
    pub fn latest(timestep: T) -> Result<Self, SignalError> {
        Self::build(timestep, WINDOW - 1)
    }

    /// Feeds one sample and returns the smoothed value of the window it now sits in.
    #[inline]
    #[must_use]
    pub fn filter(&mut self, input: T) -> T {
        if self.initialized {
            self.samples[self.next] = input;
            self.next = (self.next + 1) % WINDOW;
        } else {
            self.samples = [input; WINDOW];
            self.next = 0;
            self.initialized = true;
        }
        self.value()
    }

    /// Clears the window so the next sample seeds it again. The weights are left alone: they depend
    /// only on what the filter was built with.
    #[inline]
    pub fn reset(&mut self) {
        self.samples = [T::ZERO; WINDOW];
        self.next = 0;
        self.initialized = false;
    }

    /// The smoothed value, in the input's units.
    #[inline]
    #[must_use]
    pub fn value(&self) -> T {
        self.weighted_sum(&self.smoothing_weights)
    }

    /// How fast the input is changing, per second.
    #[inline]
    #[must_use]
    pub fn first_derivative(&self) -> T {
        self.weighted_sum(&self.first_derivative_weights)
    }

    /// How fast that rate is itself changing, per second squared.
    #[inline]
    #[must_use]
    pub fn second_derivative(&self) -> T {
        self.weighted_sum(&self.second_derivative_weights)
    }

    /// How far behind the newest sample the reported answer sits, in seconds. Zero for
    /// [`latest`](Self::latest), half a window for [`centered`](Self::centered).
    #[inline]
    #[must_use]
    pub fn delay(&self) -> T {
        self.delay_seconds
    }

    /// Works out the three weight rows from one fit across the window.
    ///
    /// `read_position` is where in the window the answer is read: the sample positions are measured
    /// from there, so the fitted curve's constant term is the answer at that point.
    fn build(timestep: T, read_position: usize) -> Result<Self, SignalError> {
        if !timestep.is_finite() {
            return Err(SignalError::NonFinite);
        }
        if timestep <= T::ZERO {
            return Err(SignalError::NonPositiveTimestep);
        }
        if WINDOW == 0 || POLYNOMIAL_TERMS == 0 {
            return Err(SignalError::WindowTooShort);
        }
        if POLYNOMIAL_TERMS > WINDOW {
            return Err(SignalError::PolynomialOrderTooHigh);
        }

        let fit = Matrix::<WINDOW, POLYNOMIAL_TERMS, T>::from_fn(|sample, term| {
            let offset = T::from_usize(sample) - T::from_usize(read_position);
            offset.powi(term as i32)
        });
        let inverse = fit.pseudo_inverse()?;

        let smoothing_weights = core::array::from_fn(|sample| inverse[(0, sample)]);
        let first_derivative_weights = core::array::from_fn(|sample| {
            if POLYNOMIAL_TERMS >= 2 {
                inverse[(1, sample)] / timestep
            } else {
                T::ZERO
            }
        });
        // The curve's third term is half the bend, so it doubles here.
        let second_derivative_weights = core::array::from_fn(|sample| {
            if POLYNOMIAL_TERMS >= 3 {
                T::TWO * inverse[(2, sample)] / (timestep * timestep)
            } else {
                T::ZERO
            }
        });

        Ok(Self {
            samples: [T::ZERO; WINDOW],
            next: 0,
            initialized: false,
            smoothing_weights,
            first_derivative_weights,
            second_derivative_weights,
            delay_seconds: T::from_usize(WINDOW - 1 - read_position) * timestep,
        })
    }

    /// The window against one weight row, walked oldest to newest from the write index.
    #[inline]
    #[must_use]
    fn weighted_sum(&self, weights: &[T; WINDOW]) -> T {
        let mut total = T::ZERO;
        #[allow(clippy::needless_range_loop)]
        for position in 0..WINDOW {
            total += weights[position] * self.samples[(self.next + position) % WINDOW];
        }
        total
    }
}
