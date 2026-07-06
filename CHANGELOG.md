# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-07-06

A feature release adding automatic differentiation, a linear-algebra core, nonlinear
optimization, and root finding, while staying strict `no_std`.

### Added

- **Automatic differentiation.** New `scalar` module with dual, hyper-dual, and Jet
  number types built on a shared numeric-traits abstraction. Autodiff is now the
  default backend for derivatives, gradients, Jacobians, and Hessians, with finite
  differences still available.
- **Linear algebra.** Fixed-size, const-generic `Vector` and `Matrix` types plus
  small-matrix solves: inverse, LU, Cholesky, QR, and SVD.
- **Optimization.** A nonlinear least-squares `optimization` module with Gauss-Newton,
  Levenberg-Marquardt, and trust-region solvers.
- **Root finding.** A `root_finding` module with bisection, Newton's method, and Newton
  for systems of equations.
- Pairwise summation utility for numerically stable sums, applied across iterative
  integration and approximation.
- An embedded curve-fitting example and an `embedded` feature marker that keeps the
  firmware example off host builds.
- A README for every source module, and benchmark suites split by area (calculus,
  linear algebra, optimization, root finding).

### Changed

- Autodiff replaces finite differences as the default differentiation method.
- Public integration tests moved from `src/**/test.rs` into top-level `tests/`, and
  `BENCHMARKS.md` was split into per-area files under `benches/`.

## [0.6.0] - 2026-06-28

A breaking rewrite focused on real-time latency, ease of use, maintainability, and strict `no_std`.

### Added

- Infinite and semi-infinite integration limits for the iterative integrators (use `f64::INFINITY` /
  `f64::NEG_INFINITY`), via a domain transform. Accurate for convergent, decaying integrands.
- A runnable example for every module under `examples/`, and `BENCHMARKS.md` with accuracy and
  latency figures.
- A criterion benchmark suite under `benches/`, plus deterministic work-count regression tests that
  fail if a latency optimization is reverted.
- `CalcError` implements `core::error::Error` and `Display`.

### Changed

- **`f64`-only API.** Dropped the `num_complex::ComplexFloat` generic; transcendental functions now
  come from `libm`, so the crate is genuinely `no_std`.
- Errors are a typed `CalcError` enum instead of `&'static str` strings.
- Functions and integrands are taken as generic `F: Fn(...)` (monomorphized) rather than `&dyn Fn`.
- Removed the redundant runtime "number of integrations" / "derivative order" arguments; these are
  now inferred from the const-generic array lengths.
- Renamed the solver types to `FiniteDifferenceSingle`/`Multi`, `IterativeSingle`/`Multi`, and
  `GaussianSingle`/`Multi`.
- Unified the vector-field calling convention: line and flux integrals now take
  `&[&dyn Fn(&[f64; N]) -> f64; N]`, matching curl and divergence.
- Gaussian quadrature takes the bare integrand; the tabulated weights carry the `e^{-x^2}` / `e^{-x}`
  weighting factor.
- Per-solver settings are now a plain-data `config` field (e.g. `solver.config.method = ...`) instead
  of getters and setters.
- Renamed the `heap` feature to `alloc`.
- Default iterative interval count is now 120 (a multiple of 12) so the Boole and Simpson rules stay
  exact.

### Fixed

- Gauss-Hermite and Gauss-Laguerre no longer double-apply the weighting function.
- Gauss-Laguerre multi-fold recursion (it read the Legendre table and failed to decrement the fold).
- The 3D line-integral z-component, and the reversed / equal / `NaN` limit checks (negative ranges
  such as `[-2.0, 1.0]` are now accepted).
- The linear and quadratic approximation formulas and their goodness-of-fit metrics.

### Removed

- The `num-complex` dependency, the `ComplexFloat` generic, and `f32` / complex-number support.

[Unreleased]: https://github.com/kmolan/multicalc-rust/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/kmolan/multicalc-rust/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/kmolan/multicalc-rust/releases/tag/v0.6.0
