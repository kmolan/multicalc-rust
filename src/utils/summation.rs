//! Blocked pairwise (cascade) summation for the long running sums.

use crate::scalar::Numeric;

// Terms are first summed naively into a fixed block, and each completed block sum is fed
// into a binary-carry tree of level slots (slot `k` holds a sum of `2^k` block sums). This
// keeps short sums (the default 120-interval integration fits in one block) at naive speed,
// while long sums still accumulate with O(log n · eps) error instead of the naive O(n · eps).
//
// BLOCK exceeds the default term count so those sums never touch the tree. LEVELS covers up
// to `BLOCK << LEVELS` terms with the full tree structure; past that the top slot absorbs the
// carry naively, which only happens at term counts far beyond anything computed here.
const BLOCK: usize = 128;
const LEVELS: usize = 32;

/// Blocked pairwise (cascade) summation. Sums terms naively within a block of `BLOCK`, then
/// combines completed block sums in a balanced binary tree so the rounding error grows like
/// O(log n · eps) rather than the naive O(n · eps). `occupied` tracks which tree slots are live.
pub(crate) struct PairwiseSum<T: Numeric> {
    tree: [T; LEVELS],
    occupied: u64,
    block_sum: T,
    block_count: usize,
}

impl<T: Numeric> PairwiseSum<T> {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            tree: [T::ZERO; LEVELS],
            occupied: 0,
            block_sum: T::ZERO,
            block_count: 0,
        }
    }

    #[inline]
    pub(crate) fn add(&mut self, value: T) {
        self.block_sum += value;
        self.block_count += 1;
        // Rarely taken, so `add` stays as cheap as a naive `+=` for the short sums.
        if self.block_count == BLOCK {
            let block = self.block_sum;
            self.push(block);
            self.block_sum = T::ZERO;
            self.block_count = 0;
        }
    }

    /// Merges a completed block sum into the tree, carrying like a binary counter.
    fn push(&mut self, mut value: T) {
        let mut level = 0;
        // The mask lets the compiler drop the bounds checks; `level` stays below LEVELS.
        while level < LEVELS && self.occupied & (1u64 << level) != 0 {
            value = self.tree[level & (LEVELS - 1)] + value;
            self.occupied &= !(1u64 << level);
            level += 1;
        }
        // Overflow guard: past `BLOCK << LEVELS` terms fold the carry into the top slot.
        if level == LEVELS {
            level = LEVELS - 1;
        }
        self.tree[level & (LEVELS - 1)] = value;
        self.occupied |= 1u64 << level;
    }

    #[inline]
    pub(crate) fn total(&self) -> T {
        // Combine the live tree slots (smallest first) with the partial block. Scan only up
        // to the highest occupied level (~log2 n): an unfilled block leaves the tree empty,
        // so a short sum returns its block directly.
        let mut sum = self.block_sum;
        let top = (u64::BITS - self.occupied.leading_zeros()) as usize;
        #[allow(clippy::needless_range_loop)]
        for level in 0..top {
            if self.occupied & (1u64 << level) != 0 {
                sum += self.tree[level & (LEVELS - 1)];
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
        // 1.0 then N tiny terms, each exactly half an ulp at 1.0. A naive `+=` rounds every
        // tiny term away and stays at 1.0; the blocked pairwise sum groups them into block
        // and tree sums that grow large enough to survive the merge, recovering the 2^-29 they
        // total to within a few ulp (the first partial block loses at most BLOCK tiny terms).
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
        assert!(
            pairwise_err < 1e-12,
            "pairwise error {pairwise_err:e} too large"
        );
        assert!(
            pairwise_err < naive_err,
            "pairwise ({pairwise_err:e}) should be strictly closer than naive ({naive_err:e})"
        );
    }
}
