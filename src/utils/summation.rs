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
