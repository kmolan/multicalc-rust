# utils

Shared error handling for the crate.

- [`error_codes::CalcError`](error_codes.rs) — the typed error enum every fallible call returns. Each
  variant names one failure (bad input, non-finite value, out-of-range order, ...), so callers match
  on the cause instead of decoding a string.

Where a sensible default exists a "safe" wrapper (such as `get_single`) returns the answer directly;
otherwise the call returns `Result<T, CalcError>` and you decide how to handle bad input.

```rust
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::iterative_integration::IterativeSingle;
use multicalc::utils::error_codes::CalcError;

let integrator = IterativeSingle::default();
match integrator.get_single(&|x: f64| 2.0 * x, &[0.0, 2.0]) {
    Ok(area) => { /* 4.0 */ }
    Err(CalcError::IterationsZero) => { /* recover */ }
    Err(_) => { /* other causes */ }
}
```
