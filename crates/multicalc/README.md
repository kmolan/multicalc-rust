# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

`multicalc` is a pure `no_std` Rust library for numerical calculus and the linear algebra
around it: exact derivatives via automatic differentiation, integration, Jacobians and Hessians,
nonlinear least-squares curve fitting, root finding, ODE integration, and 2D/3D rigid-body
math.

## Why use it

- **Runs the same math from a server to a microcontroller.** Every commit is built and tested
  on `x86_64` and `aarch64` Linux and on four bare-metal ABIs (`thumbv7em` soft-float,
  `thumbv7em` hardware-FPU, `thumbv6m`, and `riscv32imc-unknown-none-elf`). The bare-metal
  builds run the real math under QEMU and check the answers, so `no_std`, no-heap, and no-panic
  rules hold from a 64-bit CPU down to a microcontroller with no operating system.
- **Every module checked against the reference libraries.** Each module's results are verified
  against `mpmath`, `numpy`, and `scipy` fixtures, thus validating the rust implementation.
- **Fast, and measured.** On an i7-12650H: a third derivative in 26.7 ns, a small Jacobian in
  9.3 ns, a 10×10 LU solve in 239 ns, and a full Levenberg-Marquardt curve fit in 2.0 µs. See
  the [benchmarks](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks).
- **Exact derivatives, not estimates.** Differentiation, Jacobians, Hessians, Newton steps, and
  least-squares fits use forward-mode automatic differentiation (`Dual`, `HyperDual`, `Jet`), so
  derivatives are exact to machine precision rather than finite-difference approximations.
- **Pure safe and panic-free.** `forbid(unsafe_code)` across the workspace; `unwrap`/`expect`/
  `panic` are denied on library paths. Every fallible call returns a typed error.
- **One dependency.** Transcendental functions come from [`libm`](https://crates.io/crates/libm)
  (re-exported), so the math works without `std`.

## What it does

Each capability links its chapter in the [guide](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md).

- [Automatic differentiation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#scalars-and-automatic-differentiation): `Dual`, `HyperDual`, and `Jet` scalars for exact first, second, and nth-order derivatives.
- [Differentiation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#derivatives-jacobians-and-hessians): derivatives of any order (total and partial), plus Jacobian and Hessian matrices — autodiff by default, finite differences for black-box functions.
- [Integration](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#integration): iterative Newton-Cotes rules (Boole, Simpson, Trapezoidal) and Gaussian quadrature (Legendre, Hermite, Laguerre) over finite, semi-infinite, and infinite limits.
- [Linear algebra](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#linear-algebra): fixed-size, stack-allocated `Matrix` and `Vector` with LU, Cholesky, column-pivoted QR, and SVD — solves, determinant, inverse, pseudo-inverse, and condition number.
- [Least-squares optimization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#least-squares-optimization): `LevenbergMarquardt` and `GaussNewton` solvers for nonlinear curve fitting.
- [Root finding](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#root-finding): bracketed bisection and Newton solvers for scalar equations and square systems, with an optional damped line search.
- [Vector calculus](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#vector-calculus): curl, divergence, and line and flux integrals.
- [Approximation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#taylor-approximation): linear and quadratic Taylor models with goodness-of-fit metrics.
- [ODE integrators](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#ode-integrators): fixed-step `Rk4` and adaptive `Rk45` (Dormand-Prince 5(4)) with PI step control and dense output.
- [Discretization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#discretization): zero-order hold, Van Loan, and discrete white-noise models for continuous-time linear systems.
- [Spatial math](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#spatial-quaternions-and-lie-groups): `Quaternion` and the `SO2`/`SE2`/`SO3`/`SE3` Lie groups for 2D and 3D rotations and rigid-body transforms.

## Install

```sh
cargo add multicalc
```

### Feature flags

- `alloc` (off by default) — enables the heap-based methods for inputs too large for the stack.
  See [Heap allocation](#heap-allocation).

### MSRV and edition

Edition 2024, minimum supported Rust version **1.85**.

## Examples

### Exact derivatives

One formula, differentiated to any order by forward-mode autodiff:

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffSingle;
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::scalar_fn;

let f = scalar_fn!(|x| x * x * x);           // f(x) = x^3
let d = AutoDiffSingle::default();           // forward-mode autodiff, exact

let first = d.get(1, &f, 2.0).unwrap();      // 12.0
let third = d.get(3, &f, 2.0).unwrap();      //  6.0
```

### Nonlinear curve fitting

Author the residuals `model - data` with `scalar_fn_vec!`; `LevenbergMarquardt` differentiates
them under autodiff and drives the fit:

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

### Linear solves

Fixed-size `Matrix` and `Vector` with dimensions as const generics (shape mismatches are
compile errors):

```rust
use multicalc::linear_algebra::{Matrix, Vector};

// Solve A·x = b.
let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
let b = Vector::new([7.0, 19.0, 49.0]);
let x = a.solve(b).unwrap();                        // [1, 2, 3]

// A symmetric positive-definite matrix has a faster Cholesky path.
let s = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
let s_inv = s.cholesky().unwrap().inverse();
```

### Rotations and rigid-body transforms

`Quaternion` plus the `SO2`/`SE2`/`SO3`/`SE3` Lie groups, generic over the scalar so autodiff
flows straight through:

```rust
use multicalc::spatial::{SE3, SO3};
use multicalc::linear_algebra::Vector;

let r = SO3::<f64>::exp(Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2])); // 90° about z
let g = SE3::from_parts(r, Vector::new([1.0, 2.0, 3.0]));
let p = g.act(Vector::new([1.0, 0.0, 0.0]));   // rotate then translate → (1, 3, 3)
let xi = g.log();                              // 6-vector twist [v; ω]
```

The [guide](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md) has a
worked section for every module.

## Gaussian quadrature

Each Gaussian rule integrates over a fixed domain. Pass the **bare** integrand `f(x)`; the
weights already carry the weighting factor.

| Rule           | Computes                                              |
| -------------- | ---------------------------------------------------- |
| Gauss-Legendre | $\int_a^b f(x)\, \mathrm{d}x$                         |
| Gauss-Laguerre | $\int_0^\infty f(x)\, e^{-x}\, \mathrm{d}x$           |
| Gauss-Hermite  | $\int_{-\infty}^\infty f(x)\, e^{-x^2}\, \mathrm{d}x$ |

## Error handling

Where a sensible default exists, a "safe" wrapper (such as `get_single` or `get_double`) returns
the answer directly. Otherwise the call returns a `Result` whose error is the module family's own
enum — `LinalgError`, `DiffError`, `IntegrateError`, or `SolveError` — each convertible into the
`CalcError` umbrella. All variants are listed in
[error.rs](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/src/error.rs).

## Accuracy

Accuracy is verified against external-library fixtures (`mpmath`, `numpy`, `scipy`) in the
`multicalc-qa` crate, with per-module tables generated from those fixtures. See
[benchmarks/README.md](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/README.md)
for the index, or go straight to
[calculus](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/calculus.md),
[linear_algebra](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/linear_algebra.md),
[optimization](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/optimization.md),
or [root_finding](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/root_finding.md).

## Runnable demos

Runnable, self-contained programs for each module live in the
[`demos/`](https://github.com/kmolan/multicalc-rust/tree/main/demos) crate. See
[demos/README.md](https://github.com/kmolan/multicalc-rust/blob/main/demos/README.md). Run one
with:

```sh
cargo run -p multicalc-demos --example <name>
```

## Contributing

See [CONTRIBUTING.md](https://github.com/kmolan/multicalc-rust/blob/main/CONTRIBUTING.md).

## Acknowledgements

The least-squares solvers and QR factorization port the public-domain MINPACK routines `lmder`,
`lmpar`, `qrfac`, and `qrsolv` (Moré, Garbow, Hillstrom; netlib), following Moré (1978), "The
Levenberg-Marquardt algorithm: Implementation and theory", and Nocedal & Wright, *Numerical
Optimization* (chapters 4 and 10).

## License

multicalc is licensed under the MIT license.

## Contact

anmolkathail@gmail.com
