//! Streaming pairwise (cascade) summation for the long running sums.

use crate::scalar::Numeric;

// 64 levels cover any u64 term count (and any usize point count): the mask never
// carries past bit 63 for counts below 2^64, so `level` stays in 0..64.
const MAX_LEVELS: usize = 64;

/// Streaming pairwise (cascade) summation. Reorders the additions into a balanced
/// binary tree so the rounding error grows like O(log n · eps) instead of the naive
/// O(n · eps), at the same add count. Slot `k` holds a completed sum of `2^k` terms;
/// `occupied` tracks which slots are live.
pub(crate) struct PairwiseSum<T: Numeric> {
    blocks: [T; MAX_LEVELS],
    occupied: u64,
}

impl<T: Numeric> PairwiseSum<T> {
    pub(crate) fn new() -> Self {
        Self { blocks: [T::ZERO; MAX_LEVELS], occupied: 0 }
    }

    pub(crate) fn add(&mut self, mut value: T) {
        let mut level = 0;
        // Carry: while this level already holds a completed subtree, merge and move up.
        while self.occupied & (1u64 << level) != 0 {
            value = self.blocks[level] + value;
            self.occupied &= !(1u64 << level);
            level += 1;
        }
        self.blocks[level] = value;
        self.occupied |= 1u64 << level;
    }

    pub(crate) fn total(&self) -> T {
        // Combine the remaining partial subtrees, smallest first. Scan only up to the
        // highest occupied level (~log2 n), not the full bank, to keep the per-call
        // overhead O(log n) rather than a constant 64 for the short integration sums.
        let mut sum = T::ZERO;
        let top = (u64::BITS - self.occupied.leading_zeros()) as usize;
        #[allow(clippy::needless_range_loop)]
        for level in 0..top {
            if self.occupied & (1u64 << level) != 0 {
                sum += self.blocks[level];
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_zero() {
        let acc = PairwiseSum::<f64>::new();
        assert_eq!(acc.total(), 0.0);
    }

    #[test]
    fn matches_naive_on_short_input() {
        let values = [0.1, 0.2, 0.3, 0.4, 1.5, -0.7, 2.25, 8.0];

        let mut acc = PairwiseSum::new();
        let mut naive = 0.0;
        for &v in &values {
            acc.add(v);
            naive += v;
        }

        // Reordered additions, so assert closeness rather than bit-equality.
        assert!((acc.total() - naive).abs() < 1e-12);
    }

    #[test]
    fn f32_matches_naive_on_short_input() {
        let values = [0.1f32, 0.2, 0.3, 0.4, 1.5, -0.7, 2.25, 8.0];

        let mut acc = PairwiseSum::new();
        let mut naive = 0.0f32;
        for &v in &values {
            acc.add(v);
            naive += v;
        }

        assert!((acc.total() - naive).abs() < 1e-5);
    }

    #[test]
    fn beats_naive_on_long_sum() {
        // 1.0 then N tiny terms, each exactly half an ulp at 1.0. A naive `+=` rounds
        // every tiny term away and stays at 1.0; pairwise groups them into subtrees that
        // grow large enough to survive the merge, recovering the full 2^-29 they sum to.
        let tiny = 2f64.powi(-53);
        let n: u64 = 1 << 24;
        let analytic = 1.0 + (n as f64) * tiny; // 1.0 + 2^-29, exactly representable

        let mut acc = PairwiseSum::new();
        acc.add(1.0);
        let mut naive = 1.0f64;
        for _ in 0..n {
            acc.add(tiny);
            naive += tiny;
        }
        let pairwise = acc.total();

        assert_eq!(naive, 1.0, "naive should lose every tiny term");

        let pairwise_err = (pairwise - analytic).abs();
        let naive_err = (naive - analytic).abs();
        assert!(pairwise_err < 1e-15, "pairwise error {pairwise_err:e} too large");
        assert!(
            pairwise_err < naive_err,
            "pairwise ({pairwise_err:e}) should be strictly closer than naive ({naive_err:e})"
        );
    }
}
