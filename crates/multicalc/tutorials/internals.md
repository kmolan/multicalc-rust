# Internals

`utils` holds crate-internal numeric helpers. They are `pub(crate)` and not part of the public API.
The main helper is blocked pairwise summation for long sums: the rounding error grows as
`O(log n · ε)`, against `O(n · ε)` for naive summation.

---

[Back to the tutorial index](README.md)
