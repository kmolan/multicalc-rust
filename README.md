# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Single- and multi-variable calculus, automatic differentiation, nonlinear least squares, and
Lie groups in pure safe `no_std` Rust: the same code from a 64-bit server down to a bare-metal
microcontroller.**

https://github.com/user-attachments/assets/800e887d-b78b-488d-90bb-ff2dbdbb2960

*A reel of the live [showcase demos](demos#live-showcases): the 3D and 2D arm IK solvers, then the
Newton fractal and the gradient-driven marbles; every number on screen is measured live.*

## Highlights

- **Runs the same math from a server to a microcontroller.** Every commit is built and tested on **six targets**:
  the `x86_64` and `aarch64` Linux hosts and on four bare-metal ABIs (`thumbv7em` soft-float,
  `thumbv7em` hardware-FPU, `thumbv6m`, and `riscv32imc`), running the real math under QEMU.
  `no_std`, no-alloc, and no-panic rules hold on every one.
- **Every module checked against reference libraries.** Each module's results are verified against established libraries like `numpy` and `scipy` fixtures within ~1 ulp, thus validating the rust
  implementation.
- **Fast, and measured** (i7-12650H): a third derivative in 26.7 ns, a 10×10 LU solve in 239 ns,
  a full Levenberg-Marquardt fit in 2 µs. In the live demos: ~4 million Newton solves/sec on one
  core, and a 1 kHz arm IK at ~6 µs/solve. See the [benchmarks](benchmarks).
- **Exact derivatives, not estimates.** Differentiation, Jacobians, Hessians, Newton steps, and
  Levenberg-Marquardt fits use forward-mode automatic differentiation, so derivatives are exact
  to machine precision; finite differences remain available for black-box functions.
- **Pure safe and panic-free.** `#![forbid(unsafe_code)]`, no C dependencies, and `unwrap`/
  `panic` denied on library paths; every fallible call returns a typed error.
- **One dependency.** `no_std`, no heap by default, with transcendentals from
  [`libm`](https://crates.io/crates/libm).


## What it does

- [Automatic differentiation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#scalars-and-automatic-differentiation): `Dual`, `HyperDual`, and `Jet` scalars for exact first, second, and nth-order derivatives.
- [Differentiation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#derivatives-jacobians-and-hessians): derivatives of any order (total and partial), plus Jacobian and Hessian matrices; autodiff by default, finite differences for black-box functions.
- [Integration](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#integration): iterative Newton-Cotes rules (Boole, Simpson, Trapezoidal) and Gaussian quadrature (Legendre, Hermite, Laguerre) over finite, semi-infinite, and infinite limits.
- [Linear algebra](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#linear-algebra): fixed-size, stack-allocated `Matrix` and `Vector` with LU, Cholesky, column-pivoted QR, and SVD: solves, determinant, inverse, pseudo-inverse, and condition number.
- [Least-squares optimization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#least-squares-optimization): `LevenbergMarquardt` and `GaussNewton` solvers for nonlinear curve fitting.
- [Root finding](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#root-finding): bracketed bisection and Newton solvers for scalar equations and square systems, with an optional damped line search.
- [Vector calculus](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#vector-calculus): curl, divergence, and line and flux integrals.
- [Approximation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#taylor-approximation): linear and quadratic Taylor models with goodness-of-fit metrics.
- [ODE integrators](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#ode-integrators): fixed-step `Rk4` and adaptive `Rk45` (Dormand-Prince 5(4)) with PI step control and dense output.
- [Discretization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#discretization): zero-order hold, Van Loan, and discrete white-noise models for continuous-time linear systems.
- [Spatial math](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#spatial-quaternions-and-lie-groups): `Quaternion` and the `SO2`/`SE2`/`SO3`/`SE3` Lie groups for 2D and 3D rotations and rigid-body transforms.
- [Kinematics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#kinematics): differential-drive and unicycle maps between wheel and body motion, with exact SE(2) odometry.
- [Estimation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#estimation): a linear `KalmanFilter` with Joseph-form covariance updates, optional control input, and innovation access for measurement gating.

## Install

```sh
cargo add multicalc
```

## Tutorial

The [guide](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md) is a comprehesive tutorial for each module. It shows the full imports,
expected outputs in comments, error-path notes, and pointers to runnable demos. Start there when you need the complete picture of a feature.

## Documentation

- **[Guide](crates/multicalc/GUIDE.md)**: A worked section for every module: imports, a runnable
  snippet, error paths, and a demo pointer.
- **[Crate README](crates/multicalc/README.md)** / **[API docs](https://docs.rs/multicalc)**: the
  crates.io page and full API reference, with notes on `no_std`, error handling, and heap allocation.
- **[Examples](demos#start-here)**: Self-contained, self-checking programs for each module in the
  `demos/` crate. Run one with `cargo run -p multicalc-demos --example <name>`.
- **[Benchmarks](benchmarks)**: Per-module accuracy tables and latency measurements, generated
  from the QA fixtures and checked in CI.
- **[Live showcases](demos#live-showcases)**: Five animated Rerun demos: a 1 kHz IK on a 3-link arm,
  an 8-link SE(3) arm tracking a moving 3D pose, a Newton fractal, Fourier epicycles drawing Ferris,
  and gradient-driven marbles, each streaming live-measured speed and accuracy.
- **[QA crate](tools/qa)**: `multicalc-qa` holds the CI-enforced accuracy fixtures and generates the [benchmarks](benchmarks) tables from them.

## Repository layout

The published library crate lives in [`crates/multicalc`](crates/multicalc); the repository
root is a Cargo workspace. Runnable demos live in the dev-only [`demos/`](demos) crate (basics and
live Rerun showcases), and [`tools/embedded-smoke`](tools/embedded-smoke) runs `multicalc` on the
four bare-metal targets (three Cortex-M targets + `riscv32imc`) under QEMU every PR.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Acknowledgements

The least-squares solvers and QR factorization port the public-domain MINPACK routines (Moré,
Garbow, Hillstrom; netlib); the full citation is in the
[crate README](crates/multicalc/README.md#acknowledgements).

## License

Licensed under the [MIT License](LICENSE).

## Contact

anmolkathail@gmail.com
