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
///
/// There is no feedback here, so a non-finite sample is flushed out rather than latched: it spoils
/// the output for exactly as many samples as the window is long, and then the average is clean
/// again on its own. `filter_checked` keeps it out of the window altogether.
///
/// ```
/// use multicalc::signal_processing::MovingAverage;
///
/// let mut running = MovingAverage::<4, f64>::new().unwrap();
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
/// // The window is not spoiled
/// assert_eq!(running, running_snapshot);
///
/// let output = running.filter_checked(0.1_f64);
/// assert!(output.is_ok());
/// assert!(output.unwrap().is_finite());
///
/// // Unchecked, a NaN sits in the window and spoils the average ..
/// let mut spoiled = MovingAverage::<4, f64>::new().unwrap();
/// let _ = spoiled.filter(1.0);
/// assert!(spoiled.filter(f64::NAN).is_nan());
/// for _ in 0..3 {
///     assert!(spoiled.filter(1.0).is_nan());
/// }
///
/// //.. till the window has moved past it, four samples later
/// assert!((spoiled.filter(1.0) - 1.0).abs() < 1e-12);
/// ```
///
/// Two infinities of opposite sign landing in the same window are worse than either was on its
/// own: the running total cancels them into a NaN.
///
/// ```
/// use multicalc::signal_processing::MovingAverage;
///
/// let mut running = MovingAverage::<4, f64>::new().unwrap();
/// let _ = running.filter(1.0);
///
/// // One infinity alone carries straight through the sum.
/// assert!(running.filter(f64::INFINITY).is_infinite());
///
/// // Its opposite in the same window turns the total into a NaN.
/// assert!(running.filter(f64::NEG_INFINITY).is_nan());
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
    ///
    /// A non-finite input could spoil the calculation for a whole window,
    /// till it is overwritten by a finite input window-size samples later.
    /// That case should be handled by `filter_checked`
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

    /// Alternative to `filter` with checked input.
    /// Returns `SignalError::NonFinite` in case of non-finite input, leaving the window untouched.
    #[inline]
    pub fn filter_checked(&mut self, input: T) -> Result<T, SignalError> {
        if input.is_finite() {
            Ok(self.filter(input))
        } else {
            Err(SignalError::NonFinite)
        }
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
/// // An infinity is an ordinary wild reading here, and is rejected just as cleanly.
/// let mut with_infinity = RunningMedian::<5, f64>::new().unwrap();
/// for reading in [1.0, 1.1, 0.9, f64::INFINITY, 1.05] {
///     let _ = with_infinity.filter(reading);
/// }
/// assert_eq!(with_infinity.value(), 1.05);
///
/// // An even window has no single middle sample.
/// assert_eq!(
///     RunningMedian::<4, f64>::new(),
///     Err(SignalError::WindowEvenLength)
/// );
/// ```
///
/// A NaN is a different matter, and this is the one filter whose failure is silent. The sort
/// compares with `>`, which reads false in both directions against a NaN, so the NaN becomes a wall
/// that elements cannot be moved past and the middle slot no longer holds the middle value. What
/// comes out is an ordinary finite number that is simply **wrong** — nothing downstream can tell.
/// `filter_checked` refuses a NaN for exactly this reason.
///
/// ```
/// use multicalc::signal_processing::RunningMedian;
///
/// // With a wild but finite reading, the filter does its job: 50.0 is rejected.
/// let mut control = RunningMedian::<5, f64>::new().unwrap();
/// for reading in [3.0, 50.0, 1.0, 2.0, 5.0] {
///     let _ = control.filter(reading);
/// }
/// assert_eq!(control.value(), 3.0);
///
/// // Swap that reading for a NaN and the answer moves, with nothing to show for it.
/// let mut spoiled = RunningMedian::<5, f64>::new().unwrap();
/// for reading in [3.0, f64::NAN, 1.0, 2.0, 5.0] {
///     let _ = spoiled.filter(reading);
/// }
/// assert_eq!(spoiled.value(), 2.0);
/// assert!(spoiled.value().is_finite());
///
/// // The checked entry point keeps the NaN out and leaves the window as it was.
/// let mut running = RunningMedian::<5, f64>::new().unwrap();
/// let _ = running.filter(1.0);
/// let running_snapshot = running;
///
/// assert!(running.filter_checked(f64::NAN).is_err());
/// assert_eq!(running, running_snapshot);
///
/// // Infinities are let through, because the median handles them correctly.
/// assert!(running.filter_checked(f64::INFINITY).is_ok());
/// assert_eq!(running.value(), 1.0);
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
    ///
    /// A NaN value could stay in the window forever and spoils the returned middle value.
    /// An INFINITY value still work.
    /// Better use `filter_checked` in that NaN input case.
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

    /// Alternative to `filter` with input checked.
    /// Returns `SignalError::NonFinite` in case of NaN input.
    #[inline]
    pub fn filter_checked(&mut self, input: T) -> Result<T, SignalError> {
        if !input.is_nan() {
            Ok(self.filter(input))
        } else {
            Err(SignalError::NonFinite)
        }
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
