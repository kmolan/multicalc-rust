//! Filters that work from a fixed window of recent samples.

use crate::error::SignalError;
use crate::scalar::Numeric;

/// The average of the last few samples.
///
/// The first sample fills the whole window, so the output starts at that sample rather than
/// climbing from zero.
///
/// The average is added up from the whole window on every sample rather than kept as a running
/// total. A running total drifts as values are added and subtracted from it, badly enough at single
/// precision to matter over a long run, and a window is small enough that adding it up is cheap.
/// The cost is work proportional to the window length per sample.
///
/// ```
/// use multicalc::signal_processing::MovingAverage;
///
/// let mut average = MovingAverage::<4, f64>::new().unwrap();
///
/// // The first sample fills the window, so it comes straight back out.
/// assert_eq!(average.filter(1.0), 1.0);
///
/// // The next one replaces a quarter of the window: [5, 1, 1, 1] averages to 2.
/// assert_eq!(average.filter(5.0), 2.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovingAverage<const WINDOW: usize, T: Numeric = f64> {
    /// The last WINDOW samples, oldest to newest by position modulo the write index.
    samples: [T; WINDOW],
    /// Where the next sample goes.
    next: usize,
    /// Whether the first sample has seeded the window.
    initialized: bool,
}

impl<const WINDOW: usize, T: Numeric> MovingAverage<WINDOW, T> {
    /// Builds an empty filter.
    ///
    /// Returns [`SignalError::WindowTooShort`] if the window has no room for a sample.
    pub fn new() -> Result<Self, SignalError> {
        if WINDOW == 0 {
            return Err(SignalError::WindowTooShort);
        }
        Ok(Self {
            samples: [T::ZERO; WINDOW],
            next: 0,
            initialized: false,
        })
    }

    /// Feeds one sample and returns the average of the window it now sits in.
    #[inline]
    #[must_use]
    pub fn filter(&mut self, input: T) -> T {
        push(
            &mut self.samples,
            &mut self.next,
            &mut self.initialized,
            input,
        );
        self.value()
    }

    /// Clears the window so the next sample seeds it again.
    #[inline]
    pub fn reset(&mut self) {
        self.samples = [T::ZERO; WINDOW];
        self.next = 0;
        self.initialized = false;
    }

    /// The average of the window, without feeding a sample.
    #[inline]
    #[must_use]
    pub fn value(&self) -> T {
        let mut total = T::ZERO;
        for sample in self.samples {
            total += sample;
        }
        total / T::from_usize(WINDOW)
    }
}

/// The middle value of the last few samples, sorted.
///
/// This is the one filter here that removes a single wild reading outright — a lidar return off a
/// speck of dust, say — rather than blending it in. An average would carry a fraction of it into
/// the output; the middle value never sees it.
///
/// The window length has to be odd so the middle value is a sample that was actually measured.
/// Averaging the two middle values of an even window would let half of a single bad reading
/// through, which is the thing this filter exists to stop.
///
/// The first sample fills the whole window, so the output starts at that sample rather than
/// climbing from zero.
///
/// Each sample sorts a copy of the window, so the work grows with the square of the window length.
/// Windows of five to eleven samples are the useful range; a much longer one belongs somewhere
/// other than a fast loop.
///
/// ```
/// use multicalc::error::SignalError;
/// use multicalc::signal_processing::RunningMedian;
///
/// let mut median = RunningMedian::<5, f64>::new().unwrap();
/// for reading in [1.0, 1.1, 0.9, 50.0, 1.05] {
///     let _ = median.filter(reading);
/// }
///
/// // The one wild reading does not move the answer at all.
/// assert_eq!(median.value(), 1.05);
///
/// // An even window has no single middle sample.
/// assert_eq!(
///     RunningMedian::<4, f64>::new(),
///     Err(SignalError::WindowEvenLength)
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunningMedian<const WINDOW: usize, T: Numeric = f64> {
    /// The last WINDOW samples, oldest to newest by position modulo the write index.
    samples: [T; WINDOW],
    /// Where the next sample goes.
    next: usize,
    /// Whether the first sample has seeded the window.
    initialized: bool,
}

impl<const WINDOW: usize, T: Numeric> RunningMedian<WINDOW, T> {
    /// Builds an empty filter.
    ///
    /// Returns [`SignalError::WindowTooShort`] if the window has no room for a sample, or
    /// [`SignalError::WindowEvenLength`] if the window length is even.
    pub fn new() -> Result<Self, SignalError> {
        if WINDOW == 0 {
            return Err(SignalError::WindowTooShort);
        }
        if WINDOW.is_multiple_of(2) {
            return Err(SignalError::WindowEvenLength);
        }
        Ok(Self {
            samples: [T::ZERO; WINDOW],
            next: 0,
            initialized: false,
        })
    }

    /// Feeds one sample and returns the middle value of the window it now sits in.
    #[inline]
    #[must_use]
    pub fn filter(&mut self, input: T) -> T {
        push(
            &mut self.samples,
            &mut self.next,
            &mut self.initialized,
            input,
        );
        self.value()
    }

    /// Clears the window so the next sample seeds it again.
    #[inline]
    pub fn reset(&mut self) {
        self.samples = [T::ZERO; WINDOW];
        self.next = 0;
        self.initialized = false;
    }

    /// The middle value of the window, without feeding a sample.
    #[inline]
    #[must_use]
    pub fn value(&self) -> T {
        let mut sorted = self.samples;
        // Sorted by hand: the standard sort needs a total order, which floating-point numbers do
        // not have, and asking for one back would mean unwrapping a comparison that can fail.
        for placed in 1..WINDOW {
            let moving = sorted[placed];
            let mut slot = placed;
            while slot > 0 && sorted[slot - 1] > moving {
                sorted[slot] = sorted[slot - 1];
                slot -= 1;
            }
            sorted[slot] = moving;
        }
        sorted[WINDOW / 2]
    }
}

/// Writes one sample into a window, filling the whole window on the first one.
#[inline]
fn push<const WINDOW: usize, T: Numeric>(
    samples: &mut [T; WINDOW],
    next: &mut usize,
    initialized: &mut bool,
    input: T,
) {
    if *initialized {
        samples[*next] = input;
        *next = (*next + 1) % WINDOW;
    } else {
        *samples = [input; WINDOW];
        *next = 0;
        *initialized = true;
    }
}
