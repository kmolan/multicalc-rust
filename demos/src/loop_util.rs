//! Shared harness for the 1 kHz showcase examples: a pacer and a latency-percentile ring.
//!
//! The demos own their own solve timing and viz logging; this only paces the loop and
//! summarizes jitter for the hud.

use std::time::{Duration, Instant};

/// One millisecond: the target period of the 1 kHz demos.
pub const TICK: Duration = Duration::from_millis(1);

/// Paces a loop to 1 kHz boundaries.
///
/// `thread::sleep` overshoots by 50–100 µs on Linux, so [`wait`](Pacer::wait) sleeps only to
/// ~300 µs before the boundary and spins the rest.
pub struct Pacer {
    next: Instant,
    overruns: u64,
}

impl Pacer {
    /// Starts the pacer; the first [`wait`](Pacer::wait) returns near one tick from now.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next: Instant::now() + TICK,
            overruns: 0,
        }
    }

    /// Waits for the next 1 ms boundary and returns how late it woke, in microseconds.
    ///
    /// If the loop fell more than 5 ms behind it resyncs to now instead of bursting through a run
    /// of catch-up ticks, counting the event as an overrun.
    #[must_use]
    pub fn wait(&mut self) -> i64 {
        if let Some(coarse) = self.next.checked_sub(Duration::from_micros(300)) {
            let now = Instant::now();
            if coarse > now {
                std::thread::sleep(coarse - now);
            }
        }
        while Instant::now() < self.next {}
        let late_us = Instant::now().duration_since(self.next).as_micros() as i64;
        if late_us > 5_000 {
            self.overruns += 1;
            self.next = Instant::now() + TICK; // resync, don't burst
        } else {
            self.next += TICK;
        }
        late_us
    }

    /// How many times the loop fell far enough behind to force a resync.
    #[must_use]
    pub fn overruns(&self) -> u64 {
        self.overruns
    }
}

impl Default for Pacer {
    fn default() -> Self {
        Self::new()
    }
}

/// Median, 99th percentile, and max over a latency window.
#[derive(Clone, Copy, Debug)]
pub struct Percentiles {
    /// 50th percentile.
    pub median: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Largest sample in the window.
    pub max: f64,
}

/// A fixed-capacity ring of recent samples, summarized once per second for the hud.
///
/// Keeps the most recent `cap` values, overwriting the oldest; [`summary`](LatencyRing::summary)
/// clones and sorts, so call it at 1 Hz, not per tick.
pub struct LatencyRing {
    buf: Vec<f64>,
    cap: usize,
    next: usize,
}

impl LatencyRing {
    /// A ring holding the most recent `cap` samples.
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            cap,
            next: 0,
        }
    }

    /// Records one sample, evicting the oldest once full.
    pub fn push(&mut self, sample: f64) {
        if self.buf.len() < self.cap {
            self.buf.push(sample);
        } else {
            self.buf[self.next] = sample;
            self.next = (self.next + 1) % self.cap;
        }
    }

    /// Percentiles over the current window, or `None` if no samples have been recorded.
    #[must_use]
    pub fn summary(&self) -> Option<Percentiles> {
        if self.buf.is_empty() {
            return None;
        }
        let mut sorted = self.buf.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let last = sorted.len() - 1;
        let pick = |quantile: f64| sorted[(last as f64 * quantile).round() as usize];
        Some(Percentiles {
            median: pick(0.5),
            p99: pick(0.99),
            max: sorted[last],
        })
    }
}

/// Formats an integer with thousands separators, e.g. `61204` -> `"61,204"` (for hud counts).
#[must_use]
pub fn commas(n: u64) -> String {
    let digits = n.to_string();
    let len = digits.len();
    let mut out = String::with_capacity(len + (len - 1) / 3);
    for (i, ch_char) in digits.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch_char);
    }
    out
}
