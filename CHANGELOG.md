# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Capsule and cylinder geoms are measured rather than refused.** `multicalc-mjcf` works a body's
  mass and inertia out of the two primitives Menagerie leans on hardest for link geometry, from
  their closed forms — a cylinder as a disc across and a bar along, a capsule as that barrel plus
  the two hemispheres capping it, carried out to where they sit. Every number is checked against
  MuJoCo's own compile of the same geom. A size that is not a radius and a half-length is still
  refused. @Thiago316316 (#313)

- **A shape can state where its axis starts and stops.** `fromto` gives a capsule or a cylinder its
  length, its place and its facing from the two ends of its axis, which is how link geometry is
  usually written, and it comes down a `<default>` block like any other geom setting. Ends given
  alongside a `pos`, ends in the same place, and ends on a form this loader has not checked against
  the compiler are all refused by name. @Thiago316316 (#313)

- **A record of what a model file was loaded without.** `RigidBodyModel::ignored` names the
  top-level sections `multicalc-mjcf` read nothing out of — tendons, actuators, sensors, contact
  pairs and the rest — sorted and without repeats, so a caller can tell a model that loaded whole
  from one that loaded minus the half that mattered. Anything that could change a mass is still
  refused outright rather than recorded. @Thiago316316 (#305)

### Changed

- **PolylinePath.** Cache the cumulative arc length for each waypoint in `PolylinePath`. It 
  enhances the performance of `lookahead_point`. @SummerGram (#224)

### Fixed

- Particle filters floor an underflowed zero weight at the scalar's smallest positive value before
  the next update, allowing that particle to recover when later measurements favor it.

- **Iterative Config Total Iterations** Auto-align iteration counts to the composite rules below
  12k. 

## [0.10.0] - 2026-08-09

A feature release adding signal processing, polynomials and minimum-snap trajectories, LQR and
geometric attitude control, error-state Kalman filtering and AHRS attitude filters, rigid-body
dynamics, occupancy mapping, and MuJoCo model loading.

### Added

- **Signal processing.** A `signal_processing` module: `OnePoleLowPass` by smoothing weight or
  cutoff, `BiquadCoefficients` / `Biquad` for low-pass, high-pass, band-pass, and notch shapes,
  `BiquadCascade` and `MultiChannelBiquad` with `harmonic_notch_coefficients` for a frequency and
  the multiples above it, `MovingAverage` and `RunningMedian`, `SavitzkyGolay` for value, slope, and
  bend across a window, and `Deadband` / `Hysteresis` / `SlewRateLimiter` for conditioning. Each one
  is checked at construction, so every call after that is total. Ships with a `signal_filters` demo
  and fixtures checked against SciPy. @kmolan (#249)
- **Polynomials.** A `polynomial` module: `Polynomial` evaluating value and derivatives in one pass,
  `RealRoots` in closed form up to the fourth power with `count_real_roots` / `real_roots_in` past
  it, `PiecewisePolynomial`, and `MultivariatePolynomial` / `MultivariateTerm` held as terms so size
  grows with term count rather than degree. Operations that grow — multiply, compose, divide — take
  the output size from the caller and report when it is too small, through the new
  `PolynomialError`. Allocation-free and `no_std`. Also adds `MinimumSnapPlanner` and
  `durations_from_average_speed` to `motion`, for a smooth trajectory through waypoints.
  @kmolan (#250)
- **LQR and geometric attitude control.** `Lqr` for optimal linear state feedback, with `gain`,
  `closed_loop`, `cost_to_go`, `control` / `control_tracking`, and `certify_stability` to confirm
  the loop it closes settles. `GeometricAttitudeController` for attitude control directly on
  rotations, and `thrust_command_from_acceleration` / `ThrustCommand` for how to point and how hard
  to push, which is what joins a position loop to an attitude loop. Underneath,
  `solve_discrete_riccati` and `solve_discrete_lyapunov` in `linear_algebra`, plus `ControlError`
  variants for the gains and inertias that do not describe a usable controller. Checked against
  SciPy and exercised in a `control_loops` demo. @kmolan (#256)
- **Unscented Kalman filter.** `UnscentedKalmanFilter` handles a curved model by pushing a spread of
  points through it and rebuilding the estimate from where they land, so the model is never
  differentiated and does not have to be smooth. Tuning through `with_scaling` and
  `with_regularization`, both of which reject a value that does not describe a usable spread with
  the new `EstimationError::InvalidTuning`. Checked against FilterPy. @kmolan (#78)
- **Error-state Kalman filter.** `ErrorStateKalmanFilter` with `NominalState`, `NominalStateFn`, and
  `ImuNoise` keeps position, velocity, orientation, and the two IMU biases as a nominal state that
  the filter only ever corrects by a small error, so orientation stays a true rotation through the
  update. Ships with an `error_state_estimation` demo and fixtures for an IMU trajectory. @kmolan (#260)
- **AHRS attitude filters.** `MadgwickFilter` and `MahonyFilter` hold orientation from a rate
  sensor, an accelerometer, and optionally a magnetometer, with `step` /
  `step_without_magnetometer`, tunable gains, and reference directions set by
  `with_reference_directions`. Ships with an `attitude_filter` demo. @kmolan (#262)
- **Orientation integrator.** `ExponentialMap` turns a body's orientation forward by the turn it
  makes over the step, so what comes back is still a true rotation with nothing to scale back —
  `attitude_step` at a steady turn rate, `attitude_step_with_angular_acceleration` and
  `integrate_attitude` using the rate half way through the step. `RigidBody::stepped` moves a whole
  free body one tick forward the same way. Both are coarser than `Rk4`, which is the trade for an
  orientation that never drifts. @kmolan (#261)
- **Rigid-body inertia and the free joint.** `SpatialInertia` for a body's mass, balance point, and
  resistance to spinning, and `FreeJointState` for the pose and velocity of a body free to move in
  all six directions, with a `SpatialError` for the mass properties that do not describe a usable
  body. @kmolan (#259)
- **Rotor lag.** `RotorLag` in `plant` carries the delay between a commanded thrust and the thrust a
  rotor actually makes, as a first-order lag over a fixed timestep, with `stepped`, `stepped_over`,
  and `rate` for use inside an integrator. A time constant or timestep that does not describe a
  usable rotor is rejected with `PlantError`. @kmolan (#265)
- **Occupancy mapping.** A `mapping` module holding what used to live in the demos: `OccupancyMap`
  for size, placement, blocked cells, and how far a beam travels, `MutableOccupancyMap` for marking
  points, lines, and circles, `DynamicOccupancyGrid` for large heap-backed maps under `alloc`, and
  `ScanGeometry` for the directions a forward-facing scan points. `MonteCarloLocalizer` with
  `BeamModel` and `InitialParticleCloud` moves into `estimation` alongside the `ConstantTurnAndSpeed`
  and `DirectMeasurement` models and `residual_with_wrapped_angles`. @kmolan (#270)
- **Symmetric eigendecomposition.** `Matrix::symmetric_eigendecomposition` gives the eigenvalues of
  a symmetric matrix, largest first, together with the directions belonging to them, by rotating
  away the off-diagonal entries a pair at a time. `clamped` raises every eigenvalue to a floor and
  rebuilds the matrix, which is what turns a covariance that has drifted below zero back into one a
  filter can keep using. A matrix that does not read the same across the diagonal is rejected with
  the new `LinalgError::NotSymmetric`. @kmolan (#255)
- **MuJoCo model loading.** A `multicalc-mjcf` crate reading one free-floating body out of an MJCF
  file, working the mass out from the shapes where the file does not state it, checked against
  MuJoCo's own compile of the same file. Ships with a `model_ingestion` demo. @kmolan (#241)
- `Vector2D` / `Vector3D` / `Vector6D` and `Matrix2D` / `Matrix3D` / `Matrix4D` / `Matrix6D`
  aliases for the vector and matrix sizes used most, so a call site writes `Vector3D<T>` rather
  than `Vector<3, T>`. Each names the same type as the long spelling, so the two mix freely and
  nothing existing has to change. @kmolan (#241)
- `Numeric` gains `cbrt`, `expm1`, `ln_1p`, `log2`, `log10`, `ceil`, `trunc`, `clamp`, `wrap_to_pi`,
  `to_radians`, `to_degrees`, and `is_infinite`, implemented for `Dual`, `HyperDual`, and `Jet`
  alongside the plain floats. @birchmd (#245)
- Add `Matrix` methods: `trace`, `frobenius_norm`, `from_diagonal` @birchmd (#191)
- Add `Vector` methods: `normalize`, `normalized`, `map` @birchmd (#191)
- Add impl Div<T> where T: Numeric for `Matrix` and `Vector` @birchmd (#191)
- Add `SO2` methods `norm` and `normalized` @elias-taufer (#239)
- `Quaternion::from_two_vectors`, `Quaternion::rotation_angle_to`, and `Quaternion::inverse_transform_point`. @TomPlanche (#214)
- Add `Default` trait implementation for spatial types `Quaternion`, `SO2`, `SO3`, `SE2`, `SE3`, `Twist`, `Wrench`. @elias-taufer (#244)
- `vector!` and `matrix!` take a repeat form, so `vector![1.0; 3]` and `matrix![[1.0; 2]; 2]` build
  a filled vector or matrix without spelling out every entry. @kirloo (#268)
- A `3d_drone_flight` showcase demo flying a lap in three dimensions, on the `x2` MuJoCo model,
  with the 3D IMU, magnetic compass, satellite navigation, and height rangefinder sims it needs.
  @kmolan (#274)
- Documentation for `Matrix::symmetric_positive_definite`. @kmolan (#236)

### Changed

- Apply `#[must_use]` consistently across `multicalc` (and `multicalc-mjcf`) on
  value-returning accessors, builders, and helpers, so discarded results warn under
  `unused_must_use`. Engine constructors and returns already covered by `Result` /
  `Vector` / `Matrix` stay unmarked. @rtmongold (#251)
- Apply `#[must_use]` consistently across demos and host tools (`testkit`, `qa`) on
  value-returning accessors, queries, and helpers, so discarded results warn under
  `unused_must_use`. Returns already covered by `Result` / `Vector` / `Matrix`
  stay unmarked; `embedded-smoke` is excluded. @rtmongold (#258)
- `RandomSource::standard_normal` is now generic, uses the Marsaglia polar method to avoid
  trigonometry functions, and caches values to save work. @kirloo (#223)
- The Lie-group small-angle switch now uses a threshold derived per series rather than one shared
  constant, so the left Jacobian of `SO3`, its inverse, and the `SE3` Q matrix each change over at
  the angle their own truncation error justifies. A `bootstrap_thresholds` run in `qa` sweeps the
  angle range to derive them, and the tests check both sides of each threshold against mpmath.
  @solus161 (#271)
- `GUIDE.md` is split into per-topic tutorials under `crates/multicalc/tutorials/`, so a reader
  lands on the one page they need. @kmolan (#263)

### Fixed

- LU factorization now distinguishes an exactly singular pivot from one that is negligible
  relative to the matrix, rejecting unreliable solves and inverses while preserving a nonzero
  determinant. @LunaMeerkats (#207)
- Matrix constructors for `SO2` and `SO3` now reject reflections, shears, and scaling, while `SE2`
  and `SE3` also reject non-homogeneous bottom rows and non-finite translations. @LunaMeerkats (#209)
- `Matrix::expm` now rejects non-finite inputs before entering its scaling loop. @LunaMeerkats (#202)
- `ResamplingScheme::resample_indices` now leaves output indices untouched for empty weights
  instead of panicking. @LunaMeerkats (#204)
- `zoh` and `van_loan` now reject negative or non-finite timesteps before constructing their
  augmented matrices. @LunaMeerkats (#203)
- Silence `unused_must_use` in `embedded-smoke`'s LQR identity check by discading the Lyapunov
  certificate after `expect`. @rtmongold (#272)
- Install QEMU per bare-metal matrix target so thumb legs do not apt-upgrade shared modules under
  a cached `qemu-system-arm` after Ubuntu QEMU bumps. @rtmongold (#273)
- Discard the filtered value in the `signal_filters` demo so it builds under `unused_must_use`.
  @kmolan (#253)

### Removed

- `SolveError::DidNotConverge` no longer carries a `residual` field, and the `Primal` bound it
  forced on the root-finding and optimization paths that return it is gone. @solus161 (#246)
- The `curve_fit_record` demo, the demos' CSV sink, and `demos/plot.py`, replaced by the Rerun
  sink the showcase demos use. @kmolan (#274)

## [0.9.0] - 2026-07-26

A feature release adding state estimation, feedback control, wheeled kinematics, and
waypoint paths, plus a prelude and one-call entry points for the calculus core.

### Added

- Const-evaluable solver constructors, AutoDiff `new()`, and Gaussian quadrature node
  tables / `nodes`. @rtmongold (#168)
- Add `Vector`/`Matrix` `get` / `get_mut`, `Matrix::try_row` / `try_column`, and
  `Matrix::as_mut_slice_rows`. @rtmongold (#185)
- **Kinematics.** A `kinematics` module with `DifferentialDrive`, a unicycle model, and
  dead-reckoning odometry, plus a `kinematics` demo. @kmolan (#170)
- **State estimation.** An `estimation` module with `KalmanFilter` and
  `ExtendedKalmanFilter` (the latter differentiating its nonlinear models for the
  Jacobians each step), a `CovarianceUpdate` choice for how the covariance is recomputed,
  cross-validation fixtures, and an `estimation` demo. @kmolan (#175)
- **Particle filter.** A bootstrap/SIR `ParticleFilter` with pluggable resampling and
  measurement likelihood, behind the `alloc` feature. @kmolan (#188)
- **Control.** A `control` module with `Pid`, a one-pole derivative filter
  (`OnePoleLowPass`), and the pure-pursuit path-following law. @kmolan (#179)
- **Follow-the-gap.** A reactive obstacle-avoidance controller in `control`, with an
  `avoidance` demo. @kmolan (#188)
- **Waypoint paths.** A `motion` module with `PolylinePath` and its arc-length,
  closest-point, and lookahead queries. @kmolan (#179)
- **Random numbers.** A `random` module with the `RandomSource` trait and the seedable
  `Pcg32` generator, giving uniform and normal draws without a heap or `std`. @kmolan (#188)
- **Prelude and one-call functions.** `multicalc::prelude` for the traits and short
  functions worth importing together, plus `derivative`, `second_derivative`, `partial`,
  and `integral`, which need no configured backend. @kmolan (#198)
- **General N×N determinant and inverse.** Sizes past 4×4 now factor through LU instead of
  being unsupported; small sizes keep their closed forms. @kirloo (#184)
- **3D surface flux integral.** `flux_integral_3d` and `flux_integral_3d_custom`.
  @mubasharameen485-cloud (#172)
- Property tests that LU and QR factorizations reconstruct their input matrix.
  @yvoolab (#189)
- A localization and obstacle-avoidance lap demo, with simulated lidar, an occupancy grid,
  a particle-filter localizer, latency benchmarks, and embedded smoke fixtures.
  @kmolan (#192)
- docs.rs now builds with all features so gated items appear in the published docs.
  @kmolan (#197)

### Fixed

- Iterative integrators return `IntegrateError::LimitsIllDefined` instead of `NaN`
  if `classify` fails after validation. @rtmongold (#171)
- RK45 now handles zero-dimensional states without producing NaN norms.
  @terminalchai (#174)
- 2×2/3×3/4×4 `Matrix::inverse` reject near-singular inputs via an
  `EPSILON`-scaled det check instead of exact `det == 0`. @rtmongold (#181)
- Spatial small-angle Taylor cutoffs use `30 · EPSILON` and `(30 · EPSILON)²`
  instead of fixed `1e-6` / `1e-12`. @rtmongold (#190)
- The iterative and Gaussian integrators check their samples and return
  `IntegrateError::NonFinite`, instead of letting a blown-up integrand propagate into a
  garbage result. @SAY-5 (#180)
- The multi-variable integrators bounds-check the variable index and return the new
  `IntegrateError::IndexOutOfRange`. @Devil1716 (#178)

### Changed

- Keep panicking `Index` / `IndexMut` as the ergonomic Matrix/Vector access path.
  Remove panicking `row` / `column` (use `try_row` / `try_column`). @rtmongold (#185)
- **Vector-field functions renamed and re-exported.** `curl::get_2d`,
  `line_integral::get_2d`, and the rest are now `curl_2d`, `line_integral_2d`, and so on,
  imported straight from `vector_field`. @kmolan (#199)
- **Kalman filter takes a `KalmanModel`.** The four model matrices are one named struct
  instead of four positional constructor arguments. @kmolan (#198)
- The crate root re-exports the differentiation, integration, and approximation types, so
  they can be named without their module paths. @kmolan (#198)
- Rewrote the test suite for readability: table-driven cases, named expectations, and
  shared helpers. @kmolan (#200)
- Golden fixture regeneration runs from a script in a local virtualenv with locked,
  hash-checked dependencies and a pinned timestamp, so a rerun produces the same bytes.
  @kmolan (#201)
- Dropped CI caching, which was causing flaky pipeline failures. @kmolan (#173)

## [0.8.0] - 2026-07-16

A feature release adding spatial math (quaternions, Lie groups, twists/wrenches),
ODE solvers, matrix-based discretization, and per-module error enums.

### Added

- `vector!` and `matrix!` macros for building [`Vector`] and [`Matrix`] literals,
  e.g. `vector![1.0, 2.0, 3.0]` and `matrix![[1.0, 2.0], [3.0, 4.0]]`. @rtmongold (#37)
- **Autodiff scalars example.** A runnable walkthrough (`examples/autodiff_scalars.rs`)
  that uses `Dual` and `HyperDual` directly (no derivator). @rtmongold (#106)
- Fixed `examples/README.md` table entry for `optimization_solvers` (was an orphan 
  bullet). @rtmongold (#117)
- **`Primal` scalar projection.** New `to_f64` / `to_f32` trait for edge use (viz, logging, etc.), kept separate from
  `Numeric`. `showcase/viz` now uses it instead of its private `Plottable` trait. @ProtoFN (#131)
- Property tests for linear/quadratic approximation invariants (exactness on
  matching-degree polynomials, metrics consistency). @rtmongold (#130)
- Property tests that autodiff and central finite-difference derivatives agree on random
  polynomial compositions. @rtmongold (#129)
- Opt-in Kahan compensated summation for iterative integration and approximation metrics
  via `.with_kahan_summation()`; pairwise remains the default. @rtmongold (#134)
- **RISC-V embedded smoke.** `embedded-smoke` runs on `riscv32imc-unknown-none-elf`
  under QEMU `virt`, gated with its own flash/stack budgets. @rtmongold (#140)
- **Quaternion type.** A `spatial::quaternion` module with unit-quaternion rotations.
  @kmolan (#128)
- **Lie groups.** A `spatial::lie` module with `SO(2)`, `SE(2)`, `SO(3)`, and `SE(3)`
  (exp/log maps, composition, and group action), plus an `examples/lie_groups.rs`
  walkthrough. @kmolan (#133)
- **Twists and wrenches.** `spatial::twist` and `spatial::wrench` screw-theory types.
  @kmolan (#137)
- **Discretization and matrix exponential.** A new `discretization` module and a
  `linear_algebra::expm` matrix exponential. @kmolan (#137)
- **ODE solvers.** A new `ode` module with a fixed-step `RK4` and an adaptive `RK45`
  integrator, plus an `examples/ode.rs` walkthrough and an `ode` benchmark. @kmolan (#135)
- **Extended `Numeric` trait.** Inverse-trig (`asin`/`acos`/`atan`/`atan2`), hyperbolic
  (`sinh`/`cosh`/`tanh`), and `copysign`/`floor`/`signum`/`hypot`/`powf`/`mul_add`/
  `recip`, so autodiff scalar types support them. @kmolan (#120)
- Property tests for numerical integration invariants. @kirloo (#122)
- **Showcase demos.** A 3D arm IK demo and a 2D arm IK demo (renamed from `ik_servo`),
  with a CSV/Rerun sink abstraction in the `demos` crate. @kmolan (#145)

### Changed

- **Per-module error enums.** Fallible operations now return family-specific enums
  (`LinalgError`, `DiffError`, `IntegrateError`, `SolveError`), all convertible into the
  `CalcError` umbrella, replacing the monolithic error codes. @kmolan (#136)
- Faster Jacobians for large inputs: the multi-variable Jacobian now uses a single column
  solve instead of per-column solves. @kmolan (#138)
- Reorganized the benchmark suites and harness. @kmolan (#143)
- Refreshed the crate README and updated its links and media. @kmolan (#144, #146, #147)
- Internal refactor of the QA test and oracle scaffolding. @kmolan (#141)
- Updated the ignored advisories in `deny.toml`. @kmolan (#142)

### Fixed

- Fixed misspelled `examples/` link labels in `optimization/README.md`. @rtmongold
- Corrected the `CONTRIBUTING.md` link from the README. @kirloo (#121)

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

[Unreleased]: https://github.com/kmolan/multicalc-rust/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/kmolan/multicalc-rust/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/kmolan/multicalc-rust/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/kmolan/multicalc-rust/compare/v0.7.2...v0.8.0
[0.7.2]: https://github.com/kmolan/multicalc-rust/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/kmolan/multicalc-rust/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/kmolan/multicalc-rust/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/kmolan/multicalc-rust/releases/tag/v0.6.0
