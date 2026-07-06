# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Scientific computing for single- and multi-variable calculus in pure, safe Rust — from
derivatives and integrals to curve fitting and linear algebra, running everywhere from your
laptop to a bare-metal microcontroller.**

`multicalc` gives you exact derivatives via forward-mode autodiff, numerical integration over
finite and infinite domains, Jacobians and Hessians, vector-field operators, nonlinear
least-squares solvers, root finding, and a small dense linear-algebra core (LU, QR, Cholesky,
SVD) — all `no_std`, allocation-free, and panic-free by default.

## Highlights

- **Pure, safe Rust** — `#![forbid(unsafe_code)]`, no C dependencies.
- **Runs anywhere** — `no_std`, no heap, and no panics, so the same code works on a host or on
  an embedded/bare-metal target. Transcendental functions come from [`libm`](https://crates.io/crates/libm).
- **Exact by default** — differentiation, Jacobians, Hessians, and Newton steps use forward-mode
  automatic differentiation, not finite-difference approximations (which remain available for
  black-box functions).
- **Generic over the scalar** — use `f32` or `f64` (defaults to `f64`).
- **Honest error handling** — every fallible call returns a typed `CalcError`; convenience
  wrappers fill in sensible defaults.
- **Batteries included** — a runnable example for every module and a test suite covering each
  error path.

## What it does

| Area | Capabilities |
| --- | --- |
| **Differentiation** | Any order, total and partial — exact via autodiff, or finite differences for black boxes |
| **Integration** | Iterative rules (Boole, Simpson, Trapezoidal) over finite/semi-infinite/infinite limits, plus Gauss-Legendre/Hermite/Laguerre quadrature |
| **Multivariable** | Jacobian and Hessian matrices |
| **Vector calculus** | Line and flux integrals, curl, divergence |
| **Approximation** | Linear and quadratic (Taylor) models with goodness-of-fit metrics |
| **Optimization** | Levenberg-Marquardt and Gauss-Newton nonlinear least-squares |
| **Root finding** | Bracketed bisection, Newton, and Newton systems, with optional damped line search |
| **Linear algebra** | Dense LU, QR, Cholesky, and SVD — solves, inverses, pseudo-inverse, rank, condition number |

## Install

```sh
cargo add multicalc
```

## Quick look

Exact derivatives of any order — `scalar_fn!` builds a function autodiff can differentiate:

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffSingle;
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::scalar_fn;

let f = scalar_fn!(|x| x * x * x);           // f(x) = x^3
let d = AutoDiffSingle::default();           // forward-mode autodiff, exact

let first = d.get(1, &f, 2.0).unwrap();      // 12.0
let third = d.get(3, &f, 2.0).unwrap();      //  6.0
```

Fit `a·e^(b·t)` to data with Levenberg-Marquardt — author the residuals and the solver
differentiates them for you:

```rust
use multicalc::optimization::LevenbergMarquardt;
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::c;
use multicalc::scalar_fn_vec;

// Fit through (0, 100), (1, 50), (2, 25): the minimum is a = 100, b = -ln 2.
let residuals = scalar_fn_vec!(|v: &[f64; 2]| [
    c(-100.0) + v[0],
    c(-50.0)  + v[0] * v[1].exp(),
    c(-25.0)  + v[0] * (c(2.0) * v[1]).exp(),
]);
let report = LevenbergMarquardt::<AutoDiffMulti>::default()
    .minimize(&residuals, &[80.0, -0.3])
    .unwrap();
// report.solution ~ [100.0, -0.693]
```

There's a walk-through like this for **every** module — integration, quadrature, Jacobians and
Hessians, vector calculus, approximation, root finding, and linear solves — in the full guide
below.

## Documentation

- **[Full guide](crates/multicalc/README.md)** — every feature with a runnable snippet, plus
  notes on `no_std`, error handling, and heap allocation.
- **[API docs](https://docs.rs/multicalc)** on docs.rs.
- **[Examples](crates/multicalc/examples)** — self-contained programs for each module. Run one
  with `cargo run --example <name>`.
- **[Benchmarks](crates/multicalc/benches)** — accuracy figures and measured latency.

## Repository layout

The published library crate lives in [`crates/multicalc`](crates/multicalc); the repository
root is a Cargo workspace.

## Contributing

Contributions are welcome — see [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

## Acknowledgements

The least-squares solvers and QR factorization port the public-domain MINPACK routines
(Moré, Garbow, Hillstrom; netlib), following Moré (1978) and Nocedal & Wright, *Numerical
Optimization*.

## License

Licensed under the [MIT License](LICENSE).

## Contact

anmolkathail@gmail.com
