# utils

Internal numeric helpers shared across the crate.

- `summation::PairwiseSum` — blocked pairwise (cascade) summation for the long running sums. Terms
  are summed naively within a block, and completed block sums combine in a balanced binary tree, so
  the rounding error grows like O(log n · eps) instead of the naive O(n · eps) while short sums stay
  at naive speed.

This module is crate-internal (`pub(crate)`) and not part of the public API.
