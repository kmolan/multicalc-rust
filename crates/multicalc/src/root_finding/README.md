# root_finding

Root finders for scalar equations and square systems `F(x) = 0`. Each solver takes an iteration
budget and reports why it stopped as a `RootTermination` ([`mod.rs`](mod.rs)).

- [`Bisection`](bisection.rs) — brackets a scalar root and halves the interval; guaranteed to
  converge within its budget.
- [`Newton`](newton.rs) — Newton steps with a derivative from any `Derivator` (exact autodiff by
  default, finite differences on request); `with_backtracking(true)` adds a damped line search that
  rescues far starts.
- [`NewtonSystem`](newton_system.rs) — Newton for square systems `F: Rⁿ → Rⁿ` with the exact
  Jacobian and an optional backtracking line search on `‖F‖`.
- The scalar solvers return a `RootReport`; the system solver returns a `RootReportN`.

```rust
use multicalc::root_finding::{Bisection, Newton, NewtonSystem};
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::scalar::c;
use multicalc::{scalar_fn, scalar_fn_vec};

// Bracket a scalar root: f(x) = x^2 - 2 on [0, 2].
let f = scalar_fn!(|x| c(-2.0) + x * x);
let bracketed = Bisection::default().solve(&f, 0.0, 2.0).unwrap();   // ~ sqrt(2)

// Newton with exact derivatives; damped Newton adds a backtracking line search.
let quadratic = Newton::<AutoDiffSingle>::default().solve(&f, 2.0).unwrap();
let damped = Newton::<AutoDiffSingle>::default()
    .with_backtracking(true)
    .solve(&f, 2.0)
    .unwrap();

// Square system: x^2 + y^2 = 4 and x*y = 1.
let system = scalar_fn_vec!(|v: &[f64; 2]| [c(-4.0) + v[0] * v[0] + v[1] * v[1], c(-1.0) + v[0] * v[1]]);
let solved = NewtonSystem::<AutoDiffMulti>::default().solve(&system, &[1.5, 0.8]).unwrap();
// solved.root ~ [1.9319, 0.5176]; solved.termination says which test converged
```

Credits: textbook bisection and Newton–Raphson iteration; the system step reuses the crate's LU
solve and overflow-safe `enorm` from [`linear_algebra`](../linear_algebra).
