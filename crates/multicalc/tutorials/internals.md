# Internals

`utils` holds crate-internal numeric helpers — `pub(crate)`, not part of the public API. The main
one is the blocked pairwise summation used for long running sums, where rounding error grows like
`O(log n · eps)` instead of the naive `O(n · eps)`.

---

[Back to the tutorial index](README.md)
