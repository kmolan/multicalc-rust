# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `vector!` and `matrix!` macros for building [`Vector`] and [`Matrix`] literals,
  e.g. `vector![1.0, 2.0, 3.0]` and `matrix![[1.0, 2.0], [3.0, 4.0]]`. @rtmongold (#37)
- **Autodiff scalars example.** A runnable walkthrough (`examples/autodiff_scalars.rs`)
  that uses `Dual` and `HyperDual` directly (no derivator). @rtmongold (#106)
- Fixed `examples/README.md` table entry for `optimization_solvers` (was an orphan 
  bullet). @rtmongold (#117)
- **`Primal` scalar projection.** New `to_f64` / `to_f32` trait for edge use (viz, logging, etc.), kept separate from
  `Numeric`. `showcase/viz` now uses it instead of its private `Plottable` trait. @ProtoFN (#131)

### Fixed

- Fixed misspelled `examples/` link labels in `optimization/README.md`. @rtmongold

## [0.7.2] - 2026-07-09

### Added

- **f32 pseudo-inverse identity tests.** The SVD pseudo-inverse is now checked against
  the Moore–Penrose conditions in f32, complementing the existing f64 coverage.
  @KSHITIZ6341 (#110)
- **Gauss-Newton example.** A runnable walkthrough (`examples/optimization_solvers.rs`)
  and accompanying docs for the Gauss-Newton solver. @tapheret2 (#108)
- A pull-request template and a workflow that greets first-time issue and PR authors.
- Two showcase demo GIFs (1 kHz IK servo, Newton fractal) in the crate README.

### Changed

- Refreshed crates.io metadata (description, keywords, categories) for discoverability.
- Renamed `CONTRIBUTIONS.md` to `CONTRIBUTING.md` and rewrote it with a host-only
  quickstart and contribution workflow.

## [0.7.1] - 2026-07-07

A tooling and infrastructure release. The library API is unchanged; the repository
is now a Cargo workspace with cross-validation, embedded, and visualization crates
alongside the core `multicalc` crate.

### Added

- **Workspace layout.** The repository is now a Cargo workspace; the library lives in
  `crates/multicalc`, with supporting crates and tools split out. (#90)
- **Embedded smoke tests.** A `no_std` firmware crate (`crates/embedded-smoke`) that
  builds and runs core APIs on Cortex-M targets, guarding embedded compatibility in CI.
  (#92, #100)
- **Oracle.** Cross-validation fixtures for linear algebra, optimization, and quadrature,
  generated from reference implementations, with tests checking multicalc against them.
  (#93)
- **multicalc-viz showcase.** Rerun-based visualization examples under `showcase/viz`
  (Fourier Ferris, gradient marbles, IK servo, Newton fractal, live/recorded curve fit),
  plus demo reels in the README. (#97)
- **Nightly CI and expanded build matrix.** A nightly workflow and additional target
  builds. (#94)

### Changed

- Benchmark suites are unified under a single `benchmarks` target: use criterion
  substring filters (`cargo bench -- linear_algebra`) instead of `--bench <suite>`.
  @rtmongold (#96)
- Improved the README project description and switched image sources to direct GitHub
  links so they render off-crates.io. (#91, #98)
- Miscellaneous CI cleanup. (#95)

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

[Unreleased]: https://github.com/kmolan/multicalc-rust/compare/v0.7.1...HEAD
[0.7.1]: https://github.com/kmolan/multicalc-rust/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/kmolan/multicalc-rust/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/kmolan/multicalc-rust/releases/tag/v0.6.0
