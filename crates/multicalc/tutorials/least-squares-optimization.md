# Least-squares optimization

Nonlinear least-squares solvers. They minimize the sum of squared residuals of a
`scalar_fn_vec!` function, differentiating it under autodiff by default.

- `LevenbergMarquardt`: the robust, damped default.
- `GaussNewton`: the faster undamped variant for well-conditioned problems.
- `minimize` returns a `MinimizationReport` whose `TerminationReason` says which convergence
  test stopped the solver.

Write the residuals `model - data` with `scalar_fn_vec!` and the solver differentiates them
under autodiff:

```rust
use multicalc::LevenbergMarquardt;
use multicalc::AutoDiffMulti;
use multicalc::c;
use multicalc::scalar_fn_vec;

// Fit a*e^(b*t) to (0, 100), (1, 50), (2, 25): the minimum is a = 100, b = -ln 2.
let residuals = scalar_fn_vec!(|v: &[f64; 2]| [
    c(-100.0) + v[0],
    c(-50.0) + v[0] * v[1].exp(),
    c(-25.0) + v[0] * (c(2.0) * v[1]).exp(),
]);
let report = LevenbergMarquardt::<AutoDiffMulti>::default()
    .minimize(&residuals, &[80.0, -0.3])
    .unwrap();
// report.solution ~ [100.0, -0.693]; report.termination says which test converged
```

`GaussNewton` has the same API and suits well-conditioned problems where damping is unnecessary.
For a plain linear least-squares fit, use the QR factorization from
[Linear algebra](linear-algebra.md) instead.

Errors: the solvers return [`SolveError`](error-handling.md): `DidNotConverge { iters, residual }`,
`NonFinite`, or a wrapped `Linalg` / `Diff` error from a failed inner step.

Credits: the Levenberg-Marquardt driver ports MINPACK's `lmder`/`lmpar` (Moré, Garbow,
Hillstrom; public domain, netlib), following Moré (1978), "The Levenberg-Marquardt algorithm:
Implementation and theory", and Nocedal & Wright, *Numerical Optimization*, chapters 4 and 10.
Full demos:
[curve_fit.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/curve_fit.rs)
and
[optimization_solvers.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/optimization_solvers.rs).


---

[Back to the tutorial index](README.md)
