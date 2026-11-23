# Least-squares optimization

Nonlinear least-squares solvers. They minimize `‖r(x)‖²` for the residuals `r` of a
`scalar_fn_vec!` function, with the Jacobian from the derivator (`AutoDiffMulti` by default).

- `LevenbergMarquardt`: the damped trust-region method, and the robust default. It ports MINPACK
  `lmder`/`lmpar`.
- `GaussNewton`: the undamped method, faster on well-conditioned problems. It has an optional
  backtracking line search (`with_backtracking`).
- Both take `with_ftol`, `with_xtol`, `with_gtol` and `with_patience`. `LevenbergMarquardt` also
  takes `with_stepbound` and `with_scale_diag`.
- `minimize` returns a `MinimizationReport` with `solution`, `objective_function`, `evaluations`
  and `termination`. The `TerminationReason` (`Ftol`, `Xtol` or `Gtol`) gives the convergence test
  that stopped the solver.

Write the residuals as `model - data`:

```rust
use multicalc::LevenbergMarquardt;
use multicalc::AutoDiffMulti;
use multicalc::constant;
use multicalc::scalar_fn_vec;

// Fit a*e^(b*t) to (0, 100), (1, 50), (2, 25): the minimum is a = 100, b = -ln 2.
let residuals = scalar_fn_vec!(|v: &[f64; 2]| [
    constant(-100.0) + v[0],
    constant(-50.0) + v[0] * v[1].exp(),
    constant(-25.0) + v[0] * (constant(2.0) * v[1]).exp(),
]);
let report = LevenbergMarquardt::<AutoDiffMulti>::default()
    .minimize(&residuals, &[80.0, -0.3])
    .unwrap();
// report.solution ~ [100.0, -0.693]; report.termination says which test converged
```

`GaussNewton` has the same API. For a linear least-squares fit, use the QR factorization from
[Linear algebra](linear-algebra.md).

Errors: the solvers return [`SolveError`](error-handling.md): `DidNotConverge { iters }`,
`NonFinite`, or a wrapped `Linalg` / `Diff` error from a failed inner step.

Credits: MINPACK `lmder`/`lmpar` (Moré, Garbow, Hillstrom; public domain, netlib); Moré (1978),
"The Levenberg-Marquardt algorithm: Implementation and theory"; Nocedal & Wright, *Numerical
Optimization*, chapters 4 and 10. Full demos:
[curve_fit.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/curve_fit.rs)
and
[optimization_solvers.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/optimization_solvers.rs).


---

[Back to the tutorial index](README.md)
