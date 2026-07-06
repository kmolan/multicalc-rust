# optimization

Nonlinear least-squares solvers: they minimize the sum of squared residuals of a `scalar_fn_vec!`
function, differentiating it under autodiff by default.

- [`LevenbergMarquardt`](levenberg_marquardt.rs) — the robust, damped default.
- [`GaussNewton`](gauss_newton.rs) — the faster undamped variant for well-conditioned problems.
- `minimize` returns a `MinimizationReport` whose `TerminationReason` says which convergence test
  stopped the solver ([`mod.rs`](mod.rs)).

```rust
use multicalc::optimization::LevenbergMarquardt;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::c;
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

Credits: the Levenberg-Marquardt driver ports MINPACK's `lmder`/`lmpar` (Moré, Garbow, Hillstrom;
public domain, netlib), following Moré (1978), "The Levenberg-Marquardt algorithm: Implementation and
theory", and Nocedal & Wright, *Numerical Optimization*, chapters 4 and 10.
