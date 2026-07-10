# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

`multicalc` is a Rust library for single- and multi-variable calculus: derivatives, integrals,
Jacobians and Hessians, vector-field operators, and Taylor approximation in a `no-std` environment.

## Why use it

- **Tested across five hardware targets**: Every commit is built and tested on the `x86_64` and
  `aarch64` Linux hosts and on three ARM Cortex-M bare-metal ABIs (`thumbv7em` soft-float,
  `thumbv7em` hardware-FPU, and `thumbv6m`). The Cortex-M builds run the real math under QEMU on
  each ABI and check the answers against known values, so the same code is exercised from a 64-bit
  server CPU down to a microcontroller with no operating system. `no_std`, no-heap, and
  no-panic rules hold on every one.
- **Accurate to the last few bits**: differentiation, Jacobians, Hessians, and Newton steps use
  forward-mode automatic differentiation, so derivatives are exact rather than finite-difference
  estimates. The least-squares and root-finding solvers drive objectives down near `1e-30` and land
  within about one unit in the last place (ulp) of the true answer.
- **Fast, and measured**: a single derivative takes about **1 ns** and a full Levenberg-Marquardt
  curve fit finishes in **microseconds**. Latency and accuracy are tracked by benchmarks that run
  on every commit across each supported platform; see the [benchmarks](./benches).
- Generic over the scalar type: use `f32` or `f64`, defaulting to `f64`.
- Transcendental functions come from [`libm`](https://crates.io/crates/libm), so the math works
  without `std`.
- Every fallible call returns a typed [`CalcError`](./src/utils/error_codes.rs); convenience
  wrappers fill in sensible defaults.
- A runnable example for every module, and a test suite covering each error path.

<p align="center">
  <img src="https://github.com/kmolan/multicalc-rust/blob/main/showcase/viz/examples/support/ik_servo_showcase.gif" width="75%" alt="1 kHz 3-link arm running a full Levenberg-Marquardt solve every millisecond">
  <img src="https://github.com/kmolan/multicalc-rust/blob/main/showcase/viz/examples/support/newton_fractal_showcase.gif" width="75%" alt="Morphing Newton fractal, every pixel a full Newton solve">
</p>

*Two of four live [showcase demos](showcase/viz#showcases): a 1 kHz 3-link arm running a complete
Levenberg-Marquardt solve every millisecond, and a Newton fractal at ~4 million solves/sec on one
core — every number measured live.*

## What it does

- **Differentiation** of any order, total and partial: exact via forward-mode autodiff by default,
  with finite differences available for black-box functions.
- **Integration** of any order:
  - Iterative rules (Boole, Simpson, Trapezoidal) over finite, semi-infinite, and infinite limits.
  - Gaussian quadrature: Gauss-Legendre, Gauss-Hermite, Gauss-Laguerre.
- **Jacobian** and **Hessian** matrices.
- **Vector calculus**: line and flux integrals, curl, and divergence.
- **Approximation**: linear and quadratic (Taylor) models, with goodness-of-fit metrics.
- **Least-squares optimization**: Levenberg-Marquardt and Gauss-Newton solvers for nonlinear
  curve fitting.
- **Root finding**: bracketed bisection and Newton solvers for scalar equations and square
  systems, with exact derivatives by default and an optional damped (backtracking) line search.

## Install

```sh
cargo add multicalc
```

Enable the `alloc` feature if you want the heap-based methods (see [Heap allocation](#heap-allocation)).

### A note on `no_std`

Methods like `f64::sin` are only available with `std`. In a `no_std` crate, call the `libm` versions
instead, for example `libm::sin(x)` in place of `x.sin()`. `multicalc` re-exports `libm`, so you can
reach it as `multicalc::libm`. The examples below use `x.sin()` because they assume `std`.

## At a glance

### Derivatives

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffSingle;
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::scalar_fn;

let f = scalar_fn!(|x| x * x * x);           // f(x) = x^3
let d = AutoDiffSingle::default();           // forward-mode autodiff, exact

let first = d.get(1, &f, 2.0).unwrap();      // 12.0
let third = d.get(3, &f, 2.0).unwrap();      //  6.0
// get_single / get_double are wrappers for orders 1 and 2
```

For several variables, the derivative order is just the number of indices you pass:

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::numerical_derivative::derivator::DerivatorMultiVariable;
use multicalc::scalar_fn;

// g(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z
let g = scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin() + v[0] * v[1].cos() + v[0] * v[1] * v[2].exp());
let d = AutoDiffMulti::default();
let point = [1.0, 2.0, 3.0];

let dx    = d.get_single_partial(&g, 0, &point).unwrap();  // dg/dx
let mixed = d.get(&g, &[0, 1], &point).unwrap();           // d(dg/dx)/dy
let third = d.get(&g, &[0, 0, 1], &point).unwrap();        // d^3 g / dx^2 dy
```

Pass a finite-difference derivator (`FiniteDifferenceSingle` / `FiniteDifferenceMulti`) instead when
the function is a black box you cannot author with `scalar_fn!`.

### Integration

```rust
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::iterative_integration::IterativeSingle;

let f = |x: f64| 2.0 * x;
let integrator = IterativeSingle::default();   // Boole's rule, 120 intervals

let area = integrator.get_single(&f, &[0.0, 2.0]).unwrap();   // 4.0

// infinite and semi-infinite limits are supported for decaying integrands
let bell = integrator
    .get_single(&|x| (-x * x).exp(), &[f64::NEG_INFINITY, f64::INFINITY])
    .unwrap();                                                // sqrt(pi)
```

Choose the rule and interval count with `from_parameters`:

```rust
use multicalc::numerical_integration::mode::IterativeMethod;
let integrator = IterativeSingle::from_parameters(120, IterativeMethod::Simpsons);
```

### Gaussian quadrature

Each Gaussian rule integrates over a fixed domain. Pass the **bare** integrand `f(x)`; the weights
already carry the weighting factor.

```rust
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::gaussian_integration::GaussianSingle;
use multicalc::numerical_integration::mode::GaussianQuadratureMethod;

// Gauss-Hermite integrates f(x) * e^(-x^2) over the whole real line.
let hermite = GaussianSingle::from_parameters(5, GaussianQuadratureMethod::GaussHermite);
let val = hermite
    .get_single(&|x| x * x, &[f64::NEG_INFINITY, f64::INFINITY])
    .unwrap();                                                // sqrt(pi)/2
```

| Rule           | Computes                                            |
| -------------- | --------------------------------------------------- |
| Gauss-Legendre | $\int_a^b f(x)\, \mathrm{d}x$                        |
| Gauss-Laguerre | $\int_0^\infty f(x)\, e^{-x}\, \mathrm{d}x$          |
| Gauss-Hermite  | $\int_{-\infty}^\infty f(x)\, e^{-x^2}\, \mathrm{d}x$ |

### Jacobian and Hessian

A vector-valued function is authored with `scalar_fn_vec!`, so its rows differentiate under autodiff:

```rust
use multicalc::numerical_derivative::jacobian::Jacobian;
use multicalc::numerical_derivative::hessian::Hessian;
use multicalc::scalar::c;
use multicalc::{scalar_fn, scalar_fn_vec};

// the vector function (x*y*z, x^2 + y^2)
let f = scalar_fn_vec!(|v: &[f64; 3]| [v[0] * v[1] * v[2], v[0] * v[0] + v[1] * v[1]]);
let jacobian: Jacobian = Jacobian::default();
let j = jacobian.get(&f, &[1.0, 2.0, 3.0]).unwrap();   // [[6, 3, 2], [2, 4, 0]]

// g(x, y) = y*sin(x) + 2*x*e^y
let g = scalar_fn!(|v: &[f64; 2]| v[1] * v[0].sin() + c(2.0) * v[0] * v[1].exp());
let hessian: Hessian = Hessian::default();
let h = hessian.get(&g, &[1.0, 2.0]).unwrap();
```

### Vector calculus

Curl and divergence take an explicit derivator; pass `AutoDiffMulti::default()` for exact results.
Line and flux integrals sample the field, so they take plain closures.

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::c;
use multicalc::scalar_fn_vec;
use multicalc::vector_field::{curl, divergence, line_integral, flux_integral};

// field (2xy, 3cos y)
let field = scalar_fn_vec!(|v: &[f64; 2]| [c(2.0) * v[0] * v[1], c(3.0) * v[1].cos()]);
let curl_2d = curl::get_2d(AutoDiffMulti::default(), &field, &[1.0, 3.14]).unwrap();
let div_2d = divergence::get_2d(AutoDiffMulti::default(), &field, &[1.0, 3.14]).unwrap();

// field (y, -x) along the unit circle (cos t, sin t)
let g: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&(|v: &[f64; 2]| v[1]), &(|v: &[f64; 2]| -v[0])];
let curve: [&dyn Fn(f64) -> f64; 2] = [&(|t: f64| t.cos()), &(|t: f64| t.sin())];
let limit = [0.0, 2.0 * std::f64::consts::PI];
let line = line_integral::get_2d(&g, &curve, &limit).unwrap();   // -2*pi
let flux = flux_integral::get_2d(&g, &curve, &limit).unwrap();   //  0
```

### Approximation

```rust
use multicalc::approximation::linear_approximation::LinearApproximator;
use multicalc::scalar_fn;

let f = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] * v[1] + v[2] * v[2] * v[2]);
let linear: LinearApproximator = LinearApproximator::default();
let model = linear.get(&f, &[1.0, 2.0, 3.0]).unwrap();

let y = model.predict(&[1.1, 2.1, 3.1]);
// model.get_prediction_metrics(&samples, &f) returns RMSE, R^2, and more
```

`QuadraticApproximator` works the same way and captures curvature as well.

### Least-squares fitting

Author the residuals `model - data` with `scalar_fn_vec!`; the solver differentiates them under
autodiff. `LevenbergMarquardt` is the robust default; `GaussNewton` is the faster undamped variant
for well-conditioned problems.

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

For a plain linear least-squares or linear solve, use the QR factorization directly:

```rust
use multicalc::linear_algebra::{Matrix, PivotedQr, Vector};

// Least-squares fit of y = a + b*t through (0, 1), (1, 3), (2, 5): a = 1, b = 2.
let a = Matrix::<3, 2>::new([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]);
let b = Vector::new([1.0, 3.0, 5.0]);
let x = PivotedQr::decompose(a).unwrap().solve_least_squares(b).unwrap();
```

### Root finding

`Bisection` brackets a scalar root and is guaranteed to converge. `Newton` takes Newton steps with
exact derivatives by default (finite differences on request); `with_backtracking(true)` adds a damped
line search that rescues far starts. `NewtonSystem` solves square systems `F(x) = 0` with the exact
Jacobian. Every solver takes an iteration budget and reports why it stopped.

```rust
use multicalc::root_finding::{Bisection, Newton, NewtonSystem};
use multicalc::numerical_derivative::autodiff::{AutoDiffMulti, AutoDiffSingle};
use multicalc::scalar::c;
use multicalc::{scalar_fn, scalar_fn_vec};

// Bracket a scalar root: f(x) = x^2 - 2 on [0, 2].
let f = scalar_fn!(|x| c(-2.0) + x * x);
let bracketed = Bisection::default().solve(&f, 0.0, 2.0).unwrap();   // ~ sqrt(2)

// Newton with exact derivatives; damped Newton adds a backtracking line search.
let quadratic = Newton::<AutoDiffSingle>::default().solve(&f, 2.0).unwrap();   // ~ 1.41421356
let damped = Newton::<AutoDiffSingle>::default()
    .with_backtracking(true)
    .solve(&f, 2.0)
    .unwrap();

// Square system: x^2 + y^2 = 4 and x*y = 1.
let system = scalar_fn_vec!(|v: &[f64; 2]| [c(-4.0) + v[0] * v[0] + v[1] * v[1], c(-1.0) + v[0] * v[1]]);
let solved = NewtonSystem::<AutoDiffMulti>::default().solve(&system, &[1.5, 0.8]).unwrap();
// solved.root ~ [1.9319, 0.5176]; solved.termination says which test converged
```

### Linear solves

```rust
use multicalc::linear_algebra::{Matrix, Vector};

// Solve A·x = b.
let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
let b = Vector::new([7.0, 19.0, 49.0]);
let x = a.solve(b).unwrap();                       // [1, 2, 3]

let lu = a.lu().unwrap();
let det = lu.determinant();
let inv = lu.inverse();

// A symmetric positive-definite matrix has a faster Cholesky path.
let s = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
let s_inv = s.cholesky().unwrap().inverse();
```

### Singular values and pseudo-inverse

The singular value decomposition (one-sided Jacobi) gives the pseudo-inverse, minimum-norm
least-squares solve, rank, and condition number for any shape.

```rust
use multicalc::linear_algebra::{Matrix, Vector};

// Thin SVD of a tall matrix: A = U · diag(σ) · Vᵀ.
let a = Matrix::<3, 2>::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
let svd = a.svd().unwrap();
let sigma = svd.singular_values();          // descending, non-negative
let cond = svd.condition_number();          // σ_max / σ_min

// Moore-Penrose pseudo-inverse: tall, square, or wide (M < N) inputs.
let a_pinv = a.pseudo_inverse().unwrap();

// Minimum-norm least-squares solve of A·x = b, without forming A⁺.
let x = svd.solve(Vector::new([1.0, 2.0, 3.0]));
```

## Error handling

Where a sensible default exists, a "safe" wrapper (such as `get_single` or `get_double`) returns the
answer directly. Otherwise the call returns `Result<T, CalcError>` for the working scalar `T` (`f64`
by default), so you decide how to handle bad input. All variants are listed in
[error_codes.rs](./src/utils/error_codes.rs).

## Heap allocation

By default everything uses fixed-size stack arrays. Enable the `alloc` feature for `Vec`-based methods
that handle inputs too large for the stack. This currently covers the Jacobian's `get_on_heap`, which
returns a `Vec<Vec<T>>` of the scalar (`Vec<Vec<f64>>` by default).

## Examples

Runnable, self-contained programs for each module live in [`examples/`](./examples). See
[examples/README.md](./examples/README.md). Run one with:

```sh
cargo run --example <name>
```

## Benchmarks

See [benches/README.md](./benches/README.md) for the index, or go straight to
[calculus](./benches/calculus.md), [linear_algebra](./benches/linear_algebra.md),
[optimization](./benches/optimization.md), or [root_finding](./benches/root_finding.md) for
accuracy figures and measured latency.

## Contributing

See [CONTRIBUTIONS.md](../../CONTRIBUTING.md).

## Acknowledgements

The least-squares solvers and QR factorization port the public-domain MINPACK routines `lmder`,
`lmpar`, `qrfac`, and `qrsolv` (Moré, Garbow, Hillstrom; netlib), following Moré (1978), "The
Levenberg-Marquardt algorithm: Implementation and theory", and Nocedal & Wright, *Numerical
Optimization* (chapters 4 and 10).

## License

multicalc is licensed under the MIT license.

## Contact

anmolkathail@gmail.com
