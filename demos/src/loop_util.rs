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
    pub fn new(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            cap,
            next: 0,
        }
    }

    /// Records one sample, evicting the oldest once full.
    pub fn push(&mut self, v: f64) {
        if self.buf.len() < self.cap {
            self.buf.push(v);
        } else {
            self.buf[self.next] = v;
            self.next = (self.next + 1) % self.cap;
        }
    }

    /// Percentiles over the current window, or `None` if no samples have been recorded.
    pub fn summary(&self) -> Option<Percentiles> {
        if self.buf.is_empty() {
            return None;
        }
        let mut sorted = self.buf.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let last = sorted.len() - 1;
        let pick = |q: f64| sorted[(last as f64 * q).round() as usize];
        Some(Percentiles {
            median: pick(0.5),
            p99: pick(0.99),
            max: sorted[last],
        })
    }
}

/// Formats an integer with thousands separators, e.g. `61204` -> `"61,204"` (for hud counts).
pub fn commas(n: u64) -> String {
    let s = n.to_string();
    let len = s.len();
    let mut out = String::with_capacity(len + (len - 1) / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// A CHROME `n×n`-cell unit grid on the z = 0 plane, centered at the origin, as 2-point strips.
/// Log once at tick 0 with
/// `line_strips3d("world/ground", &ground_grid(3.0, 1.0), &[CHROME], &[0.004])`.
pub fn ground_grid(half_extent: f64, step: f64) -> Vec<Vec<[f64; 3]>> {
    let mut strips = Vec::new();
    let n = (half_extent / step) as i64;
    for k in -n..=n {
        let c = k as f64 * step;
        strips.push(vec![[c, -half_extent, 0.0], [c, half_extent, 0.0]]);
        strips.push(vec![[-half_extent, c, 0.0], [half_extent, c, 0.0]]);
    }
    strips
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commas_groups_thousands() {
        assert_eq!(commas(0), "0");
        assert_eq!(commas(999), "999");
        assert_eq!(commas(1000), "1,000");
        assert_eq!(commas(61204), "61,204");
        assert_eq!(commas(412000), "412,000");
        assert_eq!(commas(1234567), "1,234,567");
    }

    #[test]
    fn empty_ring_has_no_summary() {
        assert!(LatencyRing::new(8).summary().is_none());
    }

    #[test]
    fn summary_over_a_known_window() {
        // 0..=100 in ascending order: median 50, p99 99, max 100.
        let mut ring = LatencyRing::new(101);
        for v in 0..=100 {
            ring.push(v as f64);
        }
        let p = ring.summary().unwrap();
        assert_eq!(p.median, 50.0);
        assert_eq!(p.p99, 99.0);
        assert_eq!(p.max, 100.0);
    }

    #[test]
    fn ring_evicts_oldest_when_full() {
        // Capacity 3, push 5 values: the window holds the last three (2, 3, 4).
        let mut ring = LatencyRing::new(3);
        for v in 0..5 {
            ring.push(v as f64);
        }
        let p = ring.summary().unwrap();
        assert_eq!(p.max, 4.0);
        assert_eq!(p.median, 3.0);
    }
}
