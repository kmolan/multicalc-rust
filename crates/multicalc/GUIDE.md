# multicalc guide

A working tour of every public module: what it offers, the entry points, a runnable snippet,
the errors it can return, and a pointer to a full demo. It is meant to be read start to finish
by someone new to the crate, or dipped into per module once you know your way around.

Operations are generic over the [`Numeric`](#scalars-and-automatic-differentiation) scalar
trait (implemented for `f32` and `f64`, defaulting to `f64`) with transcendental functions
from `libm` so everything works without `std`. Methods like `f64::sin` need `std`; in a
`no_std` crate call the `libm` versions (`libm::sin(x)` in place of `x.sin()`). `multicalc`
re-exports `libm`, reachable as `multicalc::libm`.

Where a sensible default exists, a "safe" wrapper (such as `get_single`) returns the answer
directly. Otherwise a call returns a `Result` whose error is the module family's own enum;
see [Error handling](#error-handling).

## Contents

- [Scalars and automatic differentiation](#scalars-and-automatic-differentiation)
- [Derivatives, Jacobians, and Hessians](#derivatives-jacobians-and-hessians)
- [Integration](#integration)
- [Gaussian quadrature tables](#gaussian-quadrature-tables)
- [Taylor approximation](#taylor-approximation)
- [Linear algebra](#linear-algebra)
- [Least-squares optimization](#least-squares-optimization)
- [Root finding](#root-finding)
- [Vector calculus](#vector-calculus)
- [ODE integrators](#ode-integrators)
- [Discretization](#discretization)
- [Spatial: quaternions and Lie groups](#spatial-quaternions-and-lie-groups)
- [Kinematics](#kinematics)
- [Control](#control)
- [Estimation](#estimation)
- [Error handling](#error-handling)
- [Internals](#internals)

## Scalars and automatic differentiation

The scalar number system every calculus module is generic over: the `Numeric` trait plus the
forward-mode automatic differentiation numbers that also implement it.

- `Numeric`: the scalar trait, implemented for `f32` and `f64`.
- `Dual`, `HyperDual`, `Jet<T, N>`: autodiff scalars (dual numbers) carrying exact first,
  second, and arbitrary nth-order derivatives (`Dual` is `Jet<T, 2>`).
- `ScalarFn` / `ScalarFnN` / `VectorFn`: function traits whose `eval` is generic over the
  scalar, so one formula runs at `f64` or at any autodiff type.
- The `scalar_fn!` / `scalar_fn_vec!` macros build those traits from closure syntax, and `c()`
  marks numeric constants inside the body (a bare `2.0 * x` cannot typecheck in a generic body).

One formula, differentiated exactly to any order:

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffSingle;
use multicalc::numerical_derivative::derivator::DerivatorSingleVariable;
use multicalc::scalar_fn;

let f = scalar_fn!(|x| x * x * x);           // f(x) = x^3, evaluable at any Numeric
let d = AutoDiffSingle::default();           // forward-mode autodiff, exact

let first = d.get(1, &f, 2.0).unwrap();      // 12.0
let third = d.get(3, &f, 2.0).unwrap();      //  6.0
```

Errors: differentiation calls return [`DiffError`](#error-handling) (for example `OrderZero`).

Credits: standard forward-mode dual numbers. Full demo:
[autodiff_scalars.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/autodiff_scalars.rs).

## Derivatives, Jacobians, and Hessians

Derivatives of any order, total and partial (exact via forward-mode autodiff, or by finite
differences for black-box functions), plus Jacobian and Hessian matrices.

- `autodiff::{AutoDiffSingle, AutoDiffMulti}`: exact derivatives.
- `finite_difference::{FiniteDifferenceSingle, FiniteDifferenceMulti}`: for functions you
  cannot author with `scalar_fn!`.
- Both implement the `derivator::DerivatorSingleVariable` / `DerivatorMultiVariable` traits
  (`get`, `get_single`, `get_double`, `get_single_partial`).
- `jacobian::Jacobian` and `hessian::Hessian` build the matrices.

For several variables, the derivative order is just the number of indices you pass:

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::numerical_derivative::derivator::DerivatorMultiVariable;
use multicalc::scalar_fn;

// g(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z; order = number of indices passed
let g = scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin() + v[0] * v[1].cos() + v[0] * v[1] * v[2].exp());
let d = AutoDiffMulti::default();
let point = [1.0, 2.0, 3.0];

let dx    = d.get_single_partial(&g, 0, &point).unwrap();  // dg/dx
let mixed = d.get(&g, &[0, 1], &point).unwrap();           // d(dg/dx)/dy
let third = d.get(&g, &[0, 0, 1], &point).unwrap();        // d^3 g / dx^2 dy
```

Pass a finite-difference derivator (`FiniteDifferenceSingle` / `FiniteDifferenceMulti`) instead
when the function is a black box you cannot author with `scalar_fn!`.

A vector-valued function is authored with `scalar_fn_vec!`, so its rows differentiate under
autodiff to give the Jacobian; a scalar field gives the Hessian:

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

With the `alloc` feature, `Jacobian::get_on_heap` returns a `Vec<Vec<T>>` for inputs too large
for the stack.

Errors: these calls return [`DiffError`](#error-handling): `OrderZero`, `OrderUnsupported`,
`StepSizeZero` (finite differences), or `IndexOutOfRange`. Full demos:
[differentiation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/differentiation.rs)
and
[jacobian_hessian.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/jacobian_hessian.rs).

## Integration

Definite integration of any order: iterative Newton-Cotes rules and Gaussian quadrature, over
finite, semi-infinite, and infinite limits.

- `iterative_integration::IterativeSingle`: Boole (default), Simpson, and Trapezoidal rules;
  pick the rule and interval count with `from_parameters`.
- Pairwise summation is the default; chain `.with_kahan_summation()` to opt into Kahan.
- `gaussian_integration::GaussianSingle`: Gauss-Legendre, Gauss-Hermite, and Gauss-Laguerre.
  Pass the **bare** integrand; the weights already carry the weighting factor.
- Both implement the `integrator::IntegratorSingleVariable` / `MultiVariable` traits
  (`get_single`, `get_double`, …); the rules live in `mode`.

Iterative rules over finite and infinite limits:

```rust
use multicalc::numerical_integration::integrator::IntegratorSingleVariable;
use multicalc::numerical_integration::iterative_integration::IterativeSingle;

let integrator = IterativeSingle::default();                 // Boole's rule, 120 intervals
let area = integrator.get_single(&|x: f64| 2.0 * x, &[0.0, 2.0]).unwrap();   // 4.0

// infinite / semi-infinite limits are supported for decaying integrands
let bell = integrator
    .get_single(&|x| (-x * x).exp(), &[f64::NEG_INFINITY, f64::INFINITY])
    .unwrap();                                               // sqrt(pi)
```

Choose the rule and interval count with `from_parameters`:

```rust
use multicalc::numerical_integration::iterative_integration::IterativeSingle;
use multicalc::numerical_integration::mode::IterativeMethod;
let integrator = IterativeSingle::from_parameters(120, IterativeMethod::Simpsons);
```

Each Gaussian rule integrates over a fixed domain. Pass the bare integrand `f(x)`; the weights
already carry the weighting factor:

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

| Rule           | Computes                                              |
| -------------- | ---------------------------------------------------- |
| Gauss-Legendre | $\int_a^b f(x)\, \mathrm{d}x$                         |
| Gauss-Laguerre | $\int_0^\infty f(x)\, e^{-x}\, \mathrm{d}x$           |
| Gauss-Hermite  | $\int_{-\infty}^\infty f(x)\, e^{-x^2}\, \mathrm{d}x$ |

Gaussian nodes and weights come from the [quadrature tables](#gaussian-quadrature-tables).

Errors: integration calls return [`IntegrateError`](#error-handling): `IterationsZero`,
`LimitsIllDefined`, `QuadratureOrderOutOfRange`, or `NonFinite`. Full demos:
[iterative_integration.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/iterative_integration.rs)
and
[gaussian_integration.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/gaussian_integration.rs).

## Gaussian quadrature tables

Precomputed Gauss-Legendre, Gauss-Hermite, and Gauss-Laguerre quadrature nodes and weights, up
to order `MAX_ORDER` (30). These back the Gaussian rules in [Integration](#integration); most
users reach them through `GaussianSingle` rather than directly.

- `nodes(method, order)` returns the `(weight, abscissa)` pairs as `&'static [(f64, f64)]`, or
  `IntegrateError::QuadratureOrderOutOfRange` if the order is unavailable.
- Per-family data lives in the `legendre`, `hermite`, and `laguerre` submodules.

```rust
use multicalc::gaussian_tables;
use multicalc::numerical_integration::mode::GaussianQuadratureMethod;

let pairs = gaussian_tables::nodes(GaussianQuadratureMethod::GaussHermite, 5).unwrap();
for (weight, abscissa) in pairs {
    // weight * f(abscissa) is one term of the 5-point Gauss-Hermite sum
}
```

Errors: an out-of-range order returns `IntegrateError::QuadratureOrderOutOfRange` (see
[Error handling](#error-handling)). Credits: generated by
`scripts/build_gaussian_integration_tables.py`.

## Taylor approximation

Local Taylor models of a function around a point (linear and quadratic) with goodness-of-fit
metrics.

- `linear_approximation::LinearApproximator`: first-order model.
- `quadratic_approximation::QuadraticApproximator`: same API, also captures curvature.
- `get` builds the model; `predict` evaluates it; `get_prediction_metrics` returns MAE, MSE,
  RMSE, R², and adjusted R² against sample points.
- Metrics use pairwise summation by default; chain `.with_kahan_summation()` to opt into Kahan.

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

Errors: the underlying derivatives return [`DiffError`](#error-handling). Full demo:
[approximation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/approximation.rs).

## Linear algebra

Fixed-size, stack-allocated `Matrix` and `Vector`: dimensions are const generics (shape
mismatches are compile errors), nothing is heap-allocated, and the math never panics.

- `Matrix::lu` → `Lu`: partial-pivoting Doolittle LU; `solve`, `determinant`, `inverse`.
- `Matrix::cholesky` → `Cholesky`: faster path for symmetric positive-definite matrices.
- `PivotedQr`: column-pivoted Householder QR; `solve_least_squares`.
- `Matrix::svd` → `Svd`: one-sided Jacobi SVD; `singular_values`, `condition_number`,
  `pseudo_inverse`, minimum-norm `solve`.

Direct linear solves via LU and Cholesky:

```rust
use multicalc::linear_algebra::{Matrix, Vector};

// Solve A·x = b.
let a = Matrix::<3, 3>::new([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]]);
let b = Vector::new([7.0, 19.0, 49.0]);
let x = a.solve(b).unwrap();                        // [1, 2, 3]

let lu = a.lu().unwrap();
let det = lu.determinant();
let inv = lu.inverse();

// A symmetric positive-definite matrix has a faster Cholesky path.
let s = Matrix::<2, 2>::new([[4.0, 2.0], [2.0, 3.0]]);
let s_inv = s.cholesky().unwrap().inverse();
```

The singular value decomposition (one-sided Jacobi) gives the pseudo-inverse, minimum-norm
least-squares solve, rank, and condition number for any shape:

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

For an overdetermined linear least-squares fit, use the column-pivoted QR directly:

```rust
use multicalc::linear_algebra::{Matrix, PivotedQr, Vector};

// Least-squares fit of y = a + b*t through (0, 1), (1, 3), (2, 5): a = 1, b = 2.
let a = Matrix::<3, 2>::new([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]);
let b = Vector::new([1.0, 3.0, 5.0]);
let x = PivotedQr::decompose(a).unwrap().solve_least_squares(b).unwrap();
```

Errors: factorizations and solves return [`LinalgError`](#error-handling): `Singular`,
`NotPositiveDefinite`, `Underdetermined` (a least-squares system with `M < N`), or `NonFinite`.

Credits: the QR factorization, damped solve, and overflow-safe norm port MINPACK's `qrfac`,
`qrsolv`, and `enorm` (Moré, Garbow, Hillstrom; public domain, netlib). LU and Cholesky follow
the standard Doolittle and Cholesky–Banachiewicz algorithms; the SVD follows Golub & Van Loan,
*Matrix Computations*, and Demmel & Veselić for high relative accuracy. Full demos:
[linear_algebra.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/linear_algebra.rs)
and
[svd.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/svd.rs).

## Least-squares optimization

Nonlinear least-squares solvers: they minimize the sum of squared residuals of a
`scalar_fn_vec!` function, differentiating it under autodiff by default.

- `LevenbergMarquardt`: the robust, damped default.
- `GaussNewton`: the faster undamped variant for well-conditioned problems.
- `minimize` returns a `MinimizationReport` whose `TerminationReason` says which convergence
  test stopped the solver.

Author the residuals `model - data` with `scalar_fn_vec!`; the solver differentiates them under
autodiff:

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

`GaussNewton` has the same API and suits well-conditioned problems where damping is unnecessary.
For a plain linear least-squares fit, use the QR factorization from
[Linear algebra](#linear-algebra) instead.

Errors: the solvers return [`SolveError`](#error-handling): `DidNotConverge { iters, residual }`,
`NonFinite`, or a wrapped `Linalg` / `Diff` error from a failed inner step.

Credits: the Levenberg-Marquardt driver ports MINPACK's `lmder`/`lmpar` (Moré, Garbow,
Hillstrom; public domain, netlib), following Moré (1978), "The Levenberg-Marquardt algorithm:
Implementation and theory", and Nocedal & Wright, *Numerical Optimization*, chapters 4 and 10.
Full demos:
[curve_fit.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/curve_fit.rs)
and
[optimization_solvers.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/optimization_solvers.rs).

## Root finding

Root finders for scalar equations and square systems `F(x) = 0`. Each solver takes an iteration
budget and reports why it stopped as a `RootTermination`.

- `Bisection`: brackets a scalar root and halves the interval; guaranteed to converge within
  its budget.
- `Newton`: Newton's method with a derivative from any `Derivator` (exact autodiff by default,
  finite differences on request); `with_backtracking(true)` adds a damped line search that
  rescues far starts.
- `NewtonSystem`: Newton for square systems `F: Rⁿ → Rⁿ` with the exact Jacobian and an
  optional backtracking line search on `‖F‖`.
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

Errors: root finders return [`SolveError`](#error-handling): `DidNotConverge`, `InvalidBracket`
(bisection endpoints that do not enclose a sign change), `NonFinite`, or a wrapped `Linalg` /
`Diff` error.

Credits: textbook bisection and Newton–Raphson iteration; the system step reuses the crate's LU
solve and overflow-safe `enorm` from [Linear algebra](#linear-algebra). Full demo:
[root_finding.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/root_finding.rs).

## Vector calculus

Curl and divergence via autodiff, plus line and flux integrals sampled along a curve.

- `curl::{get_2d, get_3d}` and `divergence::{get_2d, get_3d}` take an explicit derivator (pass
  `AutoDiffMulti::default()` for exact results) and a `scalar_fn_vec!` field.
- `line_integral` and `flux_integral` sample the field, so they take plain closures for the
  field and the parametric curve.

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

The 3D curl is `(dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy)`.

Errors: the operators return [`DiffError`](#error-handling) from differentiation, and the
integrals return [`IntegrateError`](#error-handling). Full demo:
[vector_field.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/vector_field.rs).

## ODE integrators

Initial-value solvers for `y' = f(t, y)` systems, generic over the state dimension.

- `Rk4`: fixed-step classical Runge–Kutta. `Rk4::step` advances one step; `Rk4::integrate`
  runs a fixed number of steps with a per-step callback.
- `Rk45`: adaptive Dormand–Prince 5(4) with PI step control and cubic-Hermite dense output.
  `solve` integrates to a target time, `solve_on_grid` fills requested sample times via dense
  output, and `for_each_step` exposes each accepted step. Tolerances are set with `with_rtol`
  and `with_atol`.

```rust
use multicalc::ode::{Rk4, Rk45};
use multicalc::linear_algebra::Vector;

// Harmonic oscillator y'' = -y as the first-order system [position, velocity].
let f = |_t: f64, y: &Vector<2, f64>| Vector::new([y[1], -y[0]]);
let y0 = Vector::new([1.0, 0.0]);

let y1 = Rk4::step(&f, 0.0, &y0, 0.1);                                  // one fixed step of size 0.1

// Adaptive solve over one full period returns to the start [1, 0].
let yf = Rk45::default().solve(&f, 0.0, &y0, core::f64::consts::TAU).unwrap();
assert!((yf[0] - 1.0).abs() < 1e-6 && yf[1].abs() < 1e-6);
```

Dense output samples a whole grid in one pass, and `for_each_step` lets you track a conserved
quantity as the solver runs:

```rust
use multicalc::ode::Rk45;
use multicalc::linear_algebra::Vector;

let f = |_t: f64, y: &Vector<2, f64>| Vector::new([y[1], -y[0]]);
let y0 = Vector::new([1.0, 0.0]);
let solver = Rk45::default().with_rtol(1e-9).with_atol(1e-12);

let times = [0.5, 1.0, 2.0, 3.0];
let mut out = [Vector::<2, f64>::zeros(); 4];
solver.solve_on_grid(&f, 0.0, &y0, &times, &mut out).unwrap();
```

Errors: the adaptive solver returns [`IntegrateError`](#error-handling): `StepSizeTooSmall`,
`DidNotConverge { steps }`, or `NonFinite`. Full demo (harmonic oscillator plus an acrobot,
a tumbling quadrotor, and an N-body model):
[ode.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/ode.rs).

## Discretization

Turn a continuous-time linear system into its discrete-time equivalent over a step `dt`.

- `zoh(a, b, dt)`: zero-order-hold discretization of `(A, B)`, returning the discrete `(F, G)`.
- `van_loan(a, qc, dt)`: Van Loan discretization of continuous process noise, returning the
  discrete transition and process-noise covariance `(F, Q_d)`.
- `q_discrete_white_noise(dt, var)`: the filterpy-compatible discrete white-noise model.

Because the routines run through the matrix exponential, an autodiff scalar flows straight
through them, a single `Dual` recovers a derivative with respect to a parameter:

```rust
use multicalc::discretization::{q_discrete_white_noise, van_loan, zoh};
use multicalc::linear_algebra::Matrix;
use multicalc::scalar::Dual;

let dt = 0.1;

// Zero-order hold of the double integrator: F = [[1, dt], [0, 1]], G = [[dt^2/2], [dt]].
let a = Matrix::<2, 2>::new([[0.0, 1.0], [0.0, 0.0]]);
let b = Matrix::<2, 1>::new([[0.0], [1.0]]);
let (f, g) = zoh::<2, 1, 3, f64>(a, b, dt).unwrap();      // f[(0, 1)] = dt, g[(1, 0)] = dt

// Van Loan process-noise discretization of continuous white noise on velocity.
let qc = Matrix::<2, 2>::new([[0.0, 0.0], [0.0, 1.0]]);
let (_f, qd) = van_loan::<2, 4, f64>(a, qc, dt).unwrap(); // qd[(1, 1)] = dt, symmetric

// Discrete white-noise model.
let q = q_discrete_white_noise::<2, f64>(dt, 2.0);        // q[(1, 1)] = 2*dt^2

// d/dx expm(x·M) at x = 0 equals M, recovered by one Dual through expm.
let m = Matrix::<2, 2>::new([[0.2, 0.5], [-0.1, 0.3]]);
let ad = Matrix::<2, 2, Dual<f64>>::from_fn(|i, j| Dual::new(0.0, m[(i, j)]))
    .expm()
    .unwrap();
// ad[(0, 1)].deriv == m[(0, 1)]
```

Errors: the matrix-exponential step returns [`LinalgError`](#error-handling) on a non-finite
input. Full demo:
[discretization.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/discretization.rs).

## Spatial: quaternions and Lie groups

Rotations, Lie groups, and rigid-body transforms for 2D and 3D. Fixed-size, stack-allocated, no
panics, and generic over the `Numeric` scalar, so `f32`, `f64`, and the autodiff duals all work.

- `Quaternion`: Hamilton quaternion, stored scalar-first `[w, x, y, z]`: the raw algebra plus
  axis-angle / rotation-matrix / ZYX-Euler conversions, `slerp`, and `exp`/`ln`.
- `SO2` / `SE2`: 2D rotation and rigid-body transform.
- `SO3` / `SE3`: 3D rotation (wrapping a unit `Quaternion`, which carries the unit-rotation
  invariant) and rigid-body transform.
- `Twist` / `Wrench`: typed spatial velocity and force in the linear-first `[v; ω]` /
  `[force; torque]` ordering.

Every group provides `identity`, `compose` (also `*`), `inverse`, `act` on a point, `exp`/`log`,
`hat`/`vee`, `adjoint`, geodesic `interpolate`, and matrix conversions. Conventions: the tangent
ordering is `[v; ω]` (linear part first) for `SE2`/`SE3`; the retract is right-perturbation
`X · exp(ξ)`; angles are radians. `exp`/`log` Taylor-continue near θ = 0 so derivatives stay
finite at rest.

```rust
use multicalc::spatial::{SE3, SO3};
use multicalc::linear_algebra::Vector;

// A 90° rotation about z, applied to a point.
let r = SO3::<f64>::exp(Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]));
let p = r.act(Vector::new([1.0, 0.0, 0.0]));         // ≈ (0, 1, 0)

// A rigid transform: rotate, then translate.
let g = SE3::from_parts(r, Vector::new([1.0, 2.0, 3.0]));
let q = g.act(Vector::new([1.0, 0.0, 0.0]));         // ≈ (1, 3, 3)

// exp/log round trip on the tangent twist [v; ω].
let xi = g.log();
let g2 = SE3::exp(xi);
```

Because everything is generic over the scalar, a derivative with respect to a joint angle or
pose parameter flows through `act`, `compose`, and `exp`/`log` under autodiff, the basis for
the inverse-kinematics showcases. Full demo:
[lie_groups.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/lie_groups.rs);
worked applications:
[3d_arm_ik.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/3d_arm_ik.rs)
and
[2d_arm_ik.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/2d_arm_ik.rs).

## Kinematics

Maps between wheel motion and body motion for a differential drive, and pose integration on SE(2).
Fixed-size, no allocation, no panics, and generic over the `Numeric` scalar.

The body intermediate is deliberately 2-DOF, not 3: a differential drive has exactly two degrees of
freedom `(v, ω)` and exactly two wheels, so the map is a bijection and both round trips are exact
identities. There is no lateral field to silently drop.

- `DifferentialDrive`: the geometry, a wheel radius and a track width. Constructing it is the only fallible
  operation in the module; with the geometry checked once, every map below is total.
- `WheelVelocities` / `BodyTwist`: motion per second, related by `forward` and `inverse`. A
  `BodyTwist` is the se(2) twist a differential drive can realise, with the lateral term dropped.
- `WheelRotations` / `BodyArc`: motion over one tick, related by `forward_arc` and `inverse_arc`.
  `WheelRotations` is what an encoder reports; a `BodyArc` is arc length and heading change, the
  exponential coordinates of the relative pose.
- `integrate`: advances an `SE2` pose along the exact constant-twist arc.
- `Unicycle`: the same plant as an ODE right-hand side, for `Rk4`/`Rk45`.
- `OdometryStep`: the process model as a `VectorFn`, for autodiff Jacobians.

```rust
use multicalc::kinematics::{BodyTwist, DifferentialDrive, WheelVelocities, integrate};
use multicalc::scalar::Dual;
use multicalc::spatial::SE2;

// Geometry: a 36 mm wheel radius and a 235 mm track width.
let dd = DifferentialDrive::new(0.036_f64, 0.235).unwrap();

// Wheel velocities to a body twist, and back exactly.
let twist = dd.forward(WheelVelocities::new(10.0, 10.0));   // v = 0.36 m/s, ω = 0
let wheels = dd.inverse(BodyTwist::new(0.36, 0.0));         // back to (10, 10)

// The encoder path: distance travelled -> wheel rotation -> body arc -> pose.
let rotations = dd.wheel_rotations_from_travel(0.01, 0.012);
let pose = integrate(SE2::identity(), dd.forward_arc(rotations));

// Autodiff straight through an odometry step: d(pose)/d(arc length).
let step = integrate(
    SE2::<Dual<f64>>::identity(),
    BodyTwist::new(Dual::variable(0.4), Dual::constant(0.3)).integrate_over(Dual::constant(1.0)),
);
let dx_ds = step.translation()[0].deriv;
```

Because `integrate` is built on `SE2::exp`, a straight line (ω = 0) is handled by the same code path
as an arc, with no `1/ω` to blow up: the value and its derivative stay finite at exactly zero
curvature. The arc is exact for a constant twist at any step size, so the modelling error is the
zero-order hold on the wheel velocities rather than integration error.

Full demo:
[kinematics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/kinematics.rs).

## Control

Feedback controllers and steering laws for a mobile robot: a PID with anti-windup and a filtered
derivative, the pure-pursuit path-following law, and Follow-the-Gap reactive avoidance. Fixed-size,
no allocation, no panics, and generic over the `Numeric` scalar, so the same code runs at `f32` on a
microcontroller.

Angles are radians in the robot body frame, measured from the forward (+x) axis and positive
counter-clockwise. Every controller takes its configuration once, validated, and every subsequent
call is total.

- `Pid`: three gains and a fixed timestep. `with_output_limits` clamps the output and stops the
  integral winding up against the clamp; `with_derivative_filter` puts a one-pole low-pass on the
  derivative term, which is what makes a D gain usable on a noisy measurement.
- `OnePoleLowPass`: the filter on its own, by smoothing coefficient (`new`) or by cutoff frequency
  (`from_cutoff`).
- `pure_pursuit_curvature`: the exact `κ = 2·sin(α)/L_d` steering curvature toward a lookahead
  point, written in body-frame coordinates. `Curvature::to_body_twist` turns it into a command at a
  chosen speed.
- `FollowTheGap`: reactive avoidance over a forward range scan. Const-generic on the beam count,
  so the working buffer is stack-allocated and the beam geometry is fixed at compile time.

```rust
use multicalc::control::{FollowTheGap, Pid, pure_pursuit_curvature};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE2;

// A speed loop: PID on the forward speed, output limited, derivative filtered.
let mut speed_loop = Pid::new(2.0_f64, 1.0, 0.05, 0.01)
    .unwrap()
    .with_output_limits(-1.0, 1.0)
    .unwrap()
    .with_derivative_filter(0.2)
    .unwrap();
let command = speed_loop.update(0.4, 0.35); // setpoint 0.4 m/s, measured 0.35 m/s

// Steering toward a point 2 m ahead and 1 m to the left: a left turn, so positive curvature.
let curvature = pure_pursuit_curvature(SE2::identity(), Vector::new([2.0, 1.0]), 2.0).unwrap();
let twist = curvature.to_body_twist(0.4);

// Reactive avoidance: 31 beams over 120°, 4 m range, a 0.5 m chassis, a 0.5 m free-range threshold,
// 0.4 m/s cruise.
let follower: FollowTheGap<31, f64> =
    FollowTheGap::try_new(2.0 * core::f64::consts::PI / 3.0, 4.0, 0.5, 0.5, 0.4).unwrap();

// A clear scan drives straight ahead at cruise speed.
let output = follower.compute(&[4.0; 31], 0.0).unwrap();
assert!(output.heading().abs() < 1e-12);

// A wall all round stops, and says why.
let blocked = follower.compute(&[0.2; 31], 0.0).unwrap();
assert!(blocked.is_blocked());
assert_eq!(blocked.body_twist().linear(), 0.0);
```

`FollowTheGap` works in two passes over the scan. It sanitizes it — a beam that is non-finite or
non-positive is a dropped return and reads as free space at maximum range. Then it walks every
maximal run of beams above the free-range threshold, discards any run whose bounding returns are
closer together than the chassis width, and scores the rest by `span − goal_bias · |aim − goal_angle|`.
The `span` is the run's usable arc, held off each bounded edge by the angle the robot's half-width
subtends at that edge's range, and `aim` is the goal angle clamped into it — so the follower keeps
its shoulders out of the obstacles that form the gap. It steers at the winner with a yaw rate of
`steering_gain · heading` and a forward speed scaled linearly by frontal clearance.

Measuring the gap in metres rather than beams is what makes the width test meaningful: the same
angular gap is passable at 4 m and impassable at 0.4 m, and the law of cosines across the two
bounding returns settles it directly. A run that reaches either end of the field of view has no
bounding return on that side and counts as open, because the sensor saw nothing out there and
inventing a wall would stop the robot on no evidence.

It is a purely reactive method, and the doc comment says so plainly: with no map and no memory it
can dither in a three-sided concave pocket. When no run is both free and wide enough it returns a
stopped twist with `is_blocked()` set rather than inventing a heading — the recovery policy, such as
rotating in place until a gap opens, belongs to the caller.

Full demo:
[avoidance.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/avoidance.rs).

## Estimation

State estimation from noisy measurements. `KalmanFilter` is the linear filter: `predict` rolls the
state forward through a matrix model and grows the covariance by the process noise; `update` folds in
a measurement and shrinks it. Fixed-size, no allocation, and generic over the `Numeric` scalar, so a
`Dual` state differentiates the whole filter.

- `KalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>`: built from an initial estimate and the
  four model matrices — transition, measurement model, process noise, measurement noise.
- `predict` / `predict_with_control`: the time step, undriven or with a `control_model ·
  control_input` term. `CONTROL_DIMENSION` lives on the method, so undriven users never meet it.
- `update`: the measurement step. The only fallible operation in the module.
- `CovarianceUpdate`: `Joseph` (the default) or `Naive`.
- `innovation` / `innovation_covariance` / `normalized_innovation_squared`: for measurement gating.
- The setters (`set_state_transition`, `set_process_noise`, …) cover the time-varying case, where a
  changing timestep changes the model between steps.

```rust
use multicalc::estimation::KalmanFilter;
use multicalc::linear_algebra::{Matrix, Vector};

// Constant velocity: position integrates velocity over a 1 s step; position is measured.
let mut filter = KalmanFilter::new(
    Vector::new([0.0, 0.0]),                    // initial state [position, velocity]
    Matrix::new([[1.0, 0.0], [0.0, 1.0]]),      // initial covariance
    Matrix::new([[1.0, 1.0], [0.0, 1.0]]),      // state transition
    Matrix::new([[1.0, 0.0]]),                  // measurement model
    Matrix::new([[0.01, 0.0], [0.0, 0.01]]),    // process noise
    Matrix::new([[0.1]]),                       // measurement noise
);

filter.predict();
filter.update(Vector::new([1.0])).unwrap();
let position = filter.state()[0];

// Gate an outlier before folding it in.
filter.predict();
filter.update(Vector::new([1.9])).unwrap();
let gate = filter.normalized_innovation_squared().unwrap();
```

The covariance update is Joseph form by default — `(I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ` — which stays
symmetric and positive definite by construction, where the naive `(I − K·H)·P` loses symmetry as
rounding accumulates. Joseph is not a guarantee at every scale: across roughly 10⁷ single-precision
updates it too drifts, and symmetrize-and-clamp conditioning is the answer there.

`update` returns `EstimationError::NonFinite` for a non-finite measurement or innovation covariance,
and `EstimationError::NotPositiveDefinite` when the innovation covariance cannot be factorized — the
gain is undefined. `predict` is a cheap element-wise path and propagates non-finite values silently.

`ExtendedKalmanFilter<STATE_DIMENSION, MEASUREMENT_DIMENSION, T>` takes the process and measurement
models as functions rather than matrices — any `VectorFn` — and re-linearizes them at the current
estimate on every step. **The Jacobians are taken by automatic differentiation: write the model once
and its partial derivatives are exact, with no hand-derived Jacobians anywhere** — the classic source
of silent estimator bugs.

- `new` / `from_derivator`: the autodiff default, or an explicit differentiation backend (e.g.
  `FiniteDifferenceMulti`).
- `predict(&process_model)` / `update(&measurement_model, measurement)`: the models are passed per
  step, not stored, so the type stays `ExtendedKalmanFilter<3, 2>`. A control input or a changing
  timestep lives in the model as a field the caller sets between steps — there is no
  `predict_with_control`. Unlike the linear filter, `predict` here evaluates and differentiates a
  model, so it returns a `Result`.
- `update_with_residual(&measurement_model, residual)`: `update` with a caller-formed residual, for
  when a measurement component is an angle — plain subtraction is wrong across the ±π wrap, and only
  the caller knows which components are angular.
- `CovarianceUpdate`, the accessors, and `normalized_innovation_squared` are shared with the linear
  filter. `predict` and `update` also return `EstimationError::Diff` if a Jacobian step fails —
  reachable only with a finite-difference backend, as the autodiff default cannot.

```rust
use multicalc::estimation::ExtendedKalmanFilter;
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::{Numeric, VectorFn};

// Range to a landmark at (3, 4): nonlinear in the pose, so the linear filter cannot take it.
struct RangeToLandmark;
impl VectorFn<2, 1> for RangeToLandmark {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 1] {
        let to_landmark_x = S::from_f64(3.0) - state[0];
        let to_landmark_y = S::from_f64(4.0) - state[1];
        [(to_landmark_x * to_landmark_x + to_landmark_y * to_landmark_y).sqrt()]
    }
}

// A stationary target: the pose carries over unchanged.
struct Stationary;
impl VectorFn<2, 2> for Stationary {
    fn eval<S: Numeric>(&self, state: &[S; 2]) -> [S; 2] {
        [state[0], state[1]]
    }
}

let mut filter = ExtendedKalmanFilter::<2, 1>::new(
    Vector::new([0.0, 0.0]),                  // initial pose, 5.0 from the landmark
    Matrix::new([[1.0, 0.0], [0.0, 1.0]]),    // initial covariance
    Matrix::new([[0.01, 0.0], [0.0, 0.01]]),  // process noise
    Matrix::new([[0.1]]),                     // measurement noise
);
filter.predict(&Stationary).unwrap();
filter.update(&RangeToLandmark, Vector::new([5.5])).unwrap();
```

## Error handling

Each module family returns its own error enum; all six convert into the `CalcError` umbrella
via `From`, so a caller that spans families can hold one type. Every enum is `#[non_exhaustive]`,
`Copy`, and implements `Display` and `core::error::Error`.

| Enum | Raised by | Variants |
| --- | --- | --- |
| `LinalgError` | [Linear algebra](#linear-algebra), [Discretization](#discretization) | `Singular`, `NotPositiveDefinite`, `Underdetermined`, `NonFinite` |
| `DiffError` | [Derivatives](#derivatives-jacobians-and-hessians), [Approximation](#taylor-approximation), [Vector calculus](#vector-calculus) | `OrderZero`, `OrderUnsupported`, `StepSizeZero`, `IndexOutOfRange`, `EmptyFunctionSet` |
| `IntegrateError` | [Integration](#integration), [Gaussian tables](#gaussian-quadrature-tables), [ODE](#ode-integrators) | `IterationsZero`, `LimitsIllDefined`, `QuadratureOrderOutOfRange`, `StepSizeTooSmall`, `DidNotConverge { steps }`, `NonFinite` |
| `SolveError` | [Optimization](#least-squares-optimization), [Root finding](#root-finding) | `DidNotConverge { iters, residual }`, `NonFinite`, `InvalidBracket`, `Linalg(LinalgError)`, `Diff(DiffError)` |
| `KinematicsError` | [Kinematics](#kinematics) | `NonPositiveParameter`, `NonFinite` |
| `EstimationError` | [Estimation](#estimation) | `NotPositiveDefinite`, `NonFinite`, `Diff(DiffError)` |
| `CalcError` | umbrella | `Linalg`, `Solve`, `Integrate`, `Differentiate`, `Kinematics`, `Estimation` |

`SolveError` wraps `LinalgError` and `DiffError` (a solver step can fail in either), and both
are reachable through `core::error::Error::source`. Convert up to the umbrella with `?` or
`.into()`:

```rust
use multicalc::error::{CalcError, LinalgError};

fn solve() -> Result<(), CalcError> {
    let err: Result<(), LinalgError> = Err(LinalgError::Singular);
    err?;            // LinalgError -> CalcError via From
    Ok(())
}
```

## Internals

`utils` holds crate-internal numeric helpers (`pub(crate)`, not part of the public API), chiefly
the blocked pairwise summation used for long running sums, where rounding error grows like
`O(log n · eps)` instead of the naive `O(n · eps)`.
