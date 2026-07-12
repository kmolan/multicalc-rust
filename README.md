# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Scientific computing for single- and multi-variable calculus in pure, safe Rust. Numerical
derivatives, integrals, curve fitting and linear algebra; built and tested on five
hardware targets. Exercise the same code from a 64-bit server CPU down to a bare-metal microcontroller.**

<p align="center">
  <img src="https://github.com/kmolan/multicalc-rust/blob/main/showcase/viz/examples/support/ik_servo_showcase.gif" width="75%" alt="1 kHz 3-link arm running a full Levenberg-Marquardt solve every millisecond">
  <img src="https://github.com/kmolan/multicalc-rust/blob/main/showcase/viz/examples/support/newton_fractal_showcase.gif" width="75%" alt="Morphing Newton fractal, every pixel a full Newton solve">
</p>

*Two of four live [showcase demos](showcase/viz#showcases): a 1 kHz 3-link arm running a complete
Levenberg-Marquardt solve every millisecond, and a Newton fractal at ~4 million solves/sec on one
core — every number measured live.*

## Highlights

- **Pure, safe Rust**: `#![forbid(unsafe_code)]`, no C dependencies.
- **Tested against multiple platforms**: Every commit is built and tested across **five targets**: the
  `x86_64` and `aarch64` Linux hosts and three ARM Cortex-M bare-metal ABIs (`thumbv7em`
  soft-float, `thumbv7em` hardware-FPU, and `thumbv6m`). `no_std`, no heap, and no panics rules apply to every platform build, and the transcendental functions come from
  [`libm`](https://crates.io/crates/libm).
- **Fast, and measured**: a derivative in **~1 ns**, a full Levenberg-Marquardt curve fit in
  **microseconds**, and solvers that land on the answer to the **last few bits** (objectives near
  `1e-30`, errors within ~1 ulp). Comprehensive benchmarks enforced for every commit across each supported platform, see the [benchmarks](crates/multicalc/benches).
- **Exact by default**: differentiation, Jacobians, Hessians, and Newton steps use forward-mode
  automatic differentiation, not finite-difference approximations (which remain available for
  black-box functions).
- **Generic over the scalar**: use `f32` or `f64` (defaults to `f64`).
- **Batteries included**: a runnable example for every module and a test suite covering each
  error path.

## What it does

| Area | Capabilities |
| --- | --- |
| **Differentiation** | Any order, total and partial: exact via autodiff, or finite differences for black boxes |
| **Integration** | Iterative rules (Boole, Simpson, Trapezoidal) over finite/semi-infinite/infinite limits, plus Gauss-Legendre/Hermite/Laguerre quadrature |
| **Multivariable** | Jacobian and Hessian matrices |
| **Vector calculus** | Line and flux integrals, curl, divergence |
| **Approximation** | Linear and quadratic (Taylor) models with goodness-of-fit metrics |
| **Optimization** | Levenberg-Marquardt and Gauss-Newton nonlinear least-squares |
| **Root finding** | Bracketed bisection, Newton, and Newton systems, with optional damped line search |
| **Linear algebra** | Dense LU, QR, Cholesky, and SVD: solves, inverses, pseudo-inverse, rank, condition number |
| **Spatial** | Quaternions and SO(2)/SE(2)/SO(3)/SE(3) Lie groups: compose, act, exp/log, adjoint, geodesic interpolation |

## Install

```sh
cargo add multicalc
```

## Quick look

Exact derivatives of any order. `scalar_fn!` builds a function autodiff can differentiate:

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffSingle;
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::scalar_fn;

let f = scalar_fn!(|x| x * x * x);           // f(x) = x^3
let d = AutoDiffSingle::default();           // forward-mode autodiff, exact

let first = d.get(1, &f, 2.0).unwrap();      // 12.0
let third = d.get(3, &f, 2.0).unwrap();      //  6.0
```

Fit `a·e^(b·t)` to data with Levenberg-Marquardt. Author the residuals and the solver
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

There's a walk-through like this for **every** module in the full guide
below.

## Documentation

- **[Full guide](crates/multicalc/README.md)**: Every feature with a runnable snippet, plus
  notes on `no_std`, error handling, and heap allocation.
- **[API docs](https://docs.rs/multicalc)** on docs.rs.
- **[Examples](crates/multicalc/examples)**: Self-contained programs for each module. Run one
  with `cargo run --example <name>`.
- **[Live showcases](showcase/viz#showcases)**: Four animated Rerun demos — a 1 kHz IK on a 3-link arm, a
  Newton fractal, Fourier epicycles drawing Ferris, and gradient-driven marbles — each streaming
  live-measured speed and accuracy.
- **[Benchmarks](crates/multicalc/benches)**: Accuracy figures and measured latency.

## Repository layout

The published library crate lives in [`crates/multicalc`](crates/multicalc); the repository
root is a Cargo workspace. A second, dev-only crate,
[`tools/embedded-smoke`](tools/embedded-smoke), runs `multicalc` on the three bare-metal
Cortex-M targets under QEMU every PR.

## Contributing

Contributions are welcome. See [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

## Acknowledgements

The least-squares solvers and QR factorization port the public-domain MINPACK routines
(Moré, Garbow, Hillstrom; netlib), following Moré (1978) and Nocedal & Wright, *Numerical
Optimization*.

## License

Licensed under the [MIT License](LICENSE).

## Contact

anmolkathail@gmail.com
