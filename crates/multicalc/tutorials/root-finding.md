# Root finding

Root finders for scalar equations and square systems `F(x) = 0`. Each solver takes an iteration
budget and reports why it stopped as a `RootTermination`.

- `Bisection`: halves a bracket `[a, b]` with a sign change. It converges within its budget.
- `Brent`: Brent's method (bisection, secant and inverse quadratic interpolation). It keeps the
  bracket and converges superlinearly on smooth functions, at one evaluation for each iteration.
- `Newton`: `x ← x − f(x)/f'(x)`, with `f'` from any `DerivatorSingleVariable` (`AutoDiffSingle`
  by default, or finite differences). `with_backtracking(true)` adds a backtracking line search
  for far starts.
- `NewtonSystem`: Newton for square systems `F: Rⁿ → Rⁿ`, with the Jacobian from any
  `DerivatorMultiVariable` (`AutoDiffMulti` by default) and an optional backtracking line search
  on `‖F‖`.
- All take `with_xtol`, `with_ftol` and `with_max_iterations`.
- The scalar solvers return a `RootReport` (`root`, `residual`, `iterations`, `termination`). The
  system solver returns a `RootReportN` (`root`, `residual_norm`, `iterations`, `termination`).
- `RootTermination` is `ResidualTolerance`, `StepTolerance` or `BracketWidth`.

```rust
use multicalc::{Bisection, Brent, Newton, NewtonSystem};
use multicalc::{AutoDiffMulti, AutoDiffSingle};
use multicalc::constant;
use multicalc::{scalar_fn, scalar_fn_vec};

// Bracket a scalar root: f(x) = x^2 - 2 on [0, 2].
let f = scalar_fn!(|x| constant(-2.0) + x * x);
let bracketed = Bisection::default().solve(&f, 0.0, 2.0).unwrap();   // ~ sqrt(2)
let brent = Brent::default().solve(&f, 0.0, 2.0).unwrap();           // ~ sqrt(2)

// Newton with exact derivatives, without and with the backtracking line search.
let quadratic = Newton::<AutoDiffSingle>::default().solve(&f, 2.0).unwrap();   // ~ 1.41421356
let damped = Newton::<AutoDiffSingle>::default()
    .with_backtracking(true)
    .solve(&f, 2.0)
    .unwrap();

// Square system: x^2 + y^2 = 4 and x*y = 1.
let system = scalar_fn_vec!(|v: &[f64; 2]| [constant(-4.0) + v[0] * v[0] + v[1] * v[1], constant(-1.0) + v[0] * v[1]]);
let solved = NewtonSystem::<AutoDiffMulti>::default().solve(&system, &[1.5, 0.8]).unwrap();
// solved.root ~ [1.9319, 0.5176]; solved.termination says which test converged
```

Errors: root finders return [`SolveError`](error-handling.md): `DidNotConverge`, `InvalidBracket`
(bisection endpoints that do not enclose a sign change), `NonFinite`, or a wrapped `Linalg` /
`Diff` error.

`Newton` returns `Linalg(Singular)` when `|f'(x)|·(1 + |x|) ≤ ε·|f(x)|`, because that step
passes `(1 + |x|)/ε`.

The system step uses the LU solve and the overflow-safe `enorm` from
[Linear algebra](linear-algebra.md). Full demo:
[root_finding.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/root_finding.rs).


---

[Back to the tutorial index](README.md)
