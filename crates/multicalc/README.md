# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**Scientific computing that fits on a microcontroller, built and tested from scratch in one integrated package. Estimation, control, kinematics, Lie groups, calculus, autodiff and linear algebra in stable no_std Rust with no heap, no panics and no unsafe.**

## Why use it

- **1 kHz loop rates:** No heap, fixed-size types, bounded work per call. Results in a full robotics control loop at 1 kHz.
- **Tested on six embedded targets:** Every commit is built and tested on **six targets**:
  the `x86_64` and `aarch64` Linux hosts and on four bare-metal ABIs (`thumbv7em` soft-float,
  `thumbv7em` hardware-FPU, `thumbv6m`, and `riscv32imc`), running the real math under QEMU.
  `no_std`, no-alloc, and no-panic rules hold on each target.
- **Measured against external references:** Each module's results are verified against established libraries like `numpy`, `scipy`, and `filterpy` fixtures within ~1 ulp, thus validating the rust
  implementation. See the
  [benchmarks](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks).
- **Pure safe and panic-free.** `#![forbid(unsafe_code)]`, no C dependencies, and `unwrap`/
  `panic` denied on library paths; every fallible call returns a typed error. Types are fixed-size
  and stack-allocated, and iteration counts are bounded.


## What it does

### Robotics and control

- [Estimation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/estimation.md): linear, extended, and unscented `KalmanFilter`s (autodiff Jacobians, no hand-derived ones; the unscented one needs no derivatives at all), an `ErrorStateKalmanFilter` that fuses an IMU with position and heading fixes, `MahonyFilter` and `MadgwickFilter` for attitude estimation, and a `ParticleFilter` for nonlinear, non-Gaussian problems (`alloc` only), with a `Monte Carlo Localization` built on top of it.
- [Control](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/control.md): `Pid` control, infinite horizon `Lqr`,; `GeometricAttitudeController` for drones, the `pure pursuit` path-following law; and `FollowTheGap` reactive obstacle avoidance.
- [Spatial math](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/spatial-quaternions-and-lie-groups.md): `Quaternion`, the `SO2`/`SE2`/`SO3`/`SE3` Lie groups with left/right Jacobians and inverses, and `Twist`/`Wrench` spatial algebra in `[v; ω]` — Plücker transforms, 6×6 adjoints, motion and force cross products, and `SpatialInertia` momentum, bias wrench, energy and composition.
- [Rigid-body dynamics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/rigid-body-dynamics.md): `RigidBody` for a single body, `ArticulatedBody` for a jointed robot — inverse dynamics, the joint-space inertia matrix and forward dynamics (RNEA, CRBA, ABA) with armature, viscous damping and Coulomb friction, checked against Pinocchio. Both read from an MJCF or URDF file with `multicalc-robot-model`.
- [Plant](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/plant.md): What sits between a command and the force a body actually feels — `MultirotorMixer` shares a wanted lift and turn out across the rotors, and `RotorLag` models the moment a rotor takes to catch up to what it was asked for.
- [Kinematics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/kinematics.md): `KinematicTree` for revolute/prismatic/continuous/fixed/floating chains and forward and inverse kinematics, generic over the scalar with autodiff, built by hand or read from an MJCF or URDF file; a damped-least-squares SE(3) pose solver with joint limits and null-space redundancy resolution.
- [Collision checking](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/kinematics.md#collision-checking): `CollisionQuery` for sphere/capsule proximity — primitives on tree frames against each other and against world-fixed obstacles, with pair exclusions and fixed capacities.
- [Motion](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/motion.md): `PolylinePath` for waypoint paths with arc-length, closest-point, and lookahead queries, `MinimumSnapPlanner` for the smoothest trajectory through them, and `MotionProfilePlanner` for jerk-limited point-to-point moves with multi-axis synchronization.
- [Mapping](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/mapping.md): 2D `OccupancyGrid` and `ScanGeometry`
### Core math

- [Automatic differentiation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/scalars-and-automatic-differentiation.md): Exact autodiff of any order (total and partial), plus Jacobian and Hessian matrices.
- [Linear algebra](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/linear-algebra.md): fixed-size, stack-allocated `Matrix` and `Vector` with LU, Cholesky, column-pivoted QR, SVD, symmetric eigendecomposition, and the matrix exponential `expm`. General N×N determinant and inverse, pseudo-inverse, eigenvalue clamping, `solve_discrete_riccati` and `solve_discrete_lyapunov`. Plus [borrowed views](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/linear-algebra.md#borrowed-views) — `MatrixView` / `VectorView` and their `Mut` counterparts — where transpose, submatrix, row/column/diagonal, and splitting are stride arithmetic on someone else's buffer rather than a copy, and no operation panics.
- [Least-squares optimization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/least-squares-optimization.md): `LevenbergMarquardt` and `GaussNewton` solvers for nonlinear curve fitting.
- [Root finding](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/root-finding.md): bracketed bisection and Newton solvers for scalar equations and square systems, with an optional damped line search.
- [Polynomials](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/polynomials.md): `Polynomial` for evaluation with any number of derivatives in one pass, arithmetic, calculus, fitting and real roots; `PiecewisePolynomial` for curves made of pieces; and `MultivariatePolynomial` for several variables with symbolic partial derivatives.
- [Integration](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/integration.md): iterative Newton-Cotes rules (Boole, Simpson, Trapezoidal) and Gaussian quadrature (Legendre, Hermite, Laguerre) over finite, semi-infinite, and infinite limits.
- [ODE integrators](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/ode-integrators.md): fixed-step `Rk4` and adaptive `Rk45` (Dormand-Prince 5(4)) with PI step control and dense output, plus `ExponentialMap`, which is a purely orientation integrator.
- [Discretization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/discretization.md): zero-order hold, Van Loan, and discrete white-noise models for continuous-time linear systems.
- [Signal processing](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/signal-processing.md): `Biquad` low-pass, high-pass, band-pass, and notch filters; with cascades, motor-harmonic notches, and per-channel filtering. Plus `MovingAverage`, `RunningMedian`, `SavitzkyGolay` smoothing, `Deadband`, `Hysteresis` and `SlewRateLimiter` conditioning.
- [Vector calculus](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/vector-calculus.md): curl, divergence, and line and flux integrals.
- [Approximation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/taylor-approximation.md): linear and quadratic Taylor models with goodness-of-fit metrics.
- [Random](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/random.md): `Pcg32` and the `RandomSource` trait, a seedable `no_std` generator for the particle filter and for stochastic models.

## Quick start

Two formulas, written once, carried through six modules, each step feeding the next:

```rust
use multicalc::prelude::*;
use multicalc::{Hessian, Jacobian, KalmanFilter, KalmanModel, Matrix, Newton, SE3, SO3, Vector, constant };
use multicalc::{scalar_fn, scalar_fn_vec};

fn main() -> Result<(), CalcError> {
    // Written once, evaluated at f64 here and at an autodiff number wherever a derivative is asked
    // for — the formula text never changes.
    let f = scalar_fn!(|x| x * x * x - constant(2.0) * x);                     // f(x)    = x³ - 2x
    let g = scalar_fn!(|v: &[f64; 2]| v[0] * v[0] * v[1] + v[0].sin()); // g(x, y) = x²y + sin x

    // Derivatives — exact, by forward-mode autodiff. No step size, no truncation error.
    let single_point = 2.0_f64;
    let slope = derivative(&f, single_point);                // f'(2)  = 10
    let bend = second_derivative(&f, single_point);          // f''(2) = 12

    let point = [1.0_f64, 2.0];
    let x_index = 0;
    let dg_dx = partial(&g, x_index, &point)?;

    // The derivative matrices of those same two formulas.
    let hessian = Hessian::new().evaluate(&g, &point)?;      // 2x2 second derivatives
    let both = scalar_fn_vec!(|v: &[f64; 2]| [
        v[0] * v[0] * v[1] + v[0].sin(),
        v[0] * v[0] * v[0] - constant(2.0) * v[0],
    ]);
    let jacobian = Jacobian::new().evaluate(&both, &point)?; // 2x2 first derivatives

    // Integration — f again, this time over an interval.
    let limits = [0.0, 2.0];
    let area = integral(&|x: f64| f.eval(x), limits)?;       // ∫₀² f = 0

    // Linear algebra — solve H·x = b with the Hessian computed three lines up.
    let b = Vector::new([1.0, 2.0]);
    let x = hessian.solve(b)?;

    // Root finding — Newton on the same f, its derivative supplied by autodiff.
    let initial_guess = 2.0;
    let root = Newton::new().solve(&f, initial_guess)?.root; // √2 ≈ 1.41421356

    // Rigid-body motion — SO(3)/SE(3), generic over the scalar like everything above.
    let quarter_turn_about_z = Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]);
    let translation = Vector::new([1.0, 2.0, 3.0]);
    let start = Vector::new([1.0, 0.0, 0.0]);

    let pose = SE3::from_parts(SO3::exp(quarter_turn_about_z), translation);
    let moved = pose.act(start);                  // rotate, then translate → (1, 3, 3)

    // Estimation — a Kalman filter recovering the velocity it never measures.
    let initial_state = Vector::new([0.0, 0.0]);  // [position, velocity]
    let initial_covariance = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    let model = KalmanModel {
        state_transition: Matrix::new([[1.0, 1.0], [0.0, 1.0]]),
        measurement_model: Matrix::new([[1.0, 0.0]]),        // position only
        process_noise: Matrix::new([[0.01, 0.0], [0.0, 0.01]]),
        measurement_noise: Matrix::new([[0.1]]),
    };

    let mut filter = KalmanFilter::new(initial_state, initial_covariance, model);
    filter.predict();

    let measurement = Vector::new([1.0]);         // the target moved about 1 m
    filter.update(measurement)?;
    let velocity = filter.state()[1];             // recovered, though never measured

    Ok(())
}
```

Every fallible call propagates with `?`: each module has its own error enum, and all of them
convert into the `CalcError` umbrella, so one return type covers a program that mixes modules.

## Full tutorial

Refer to the [tutorials](https://github.com/kmolan/multicalc-rust/tree/main/crates/multicalc/tutorials) for a comprehensive tutorial for each module. They show the full imports,
expected outputs in comments, error-path notes, and pointers to runnable demos. Start there when you need the complete picture of a feature.

## Accuracy

Verified against external-library fixtures (`mpmath`, `numpy`, `scipy`, `filterpy`) in
the `multicalc-qa` crate, with per-module tables generated from those fixtures. See
[benchmarks/README.md](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/README.md)
for the index, or go straight to
[calculus](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/calculus.md),
[linear_algebra](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/linear_algebra.md),
[optimization](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/optimization.md),
[ode](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/ode.md),
[estimation](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/estimation.md),
or [root_finding](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/root_finding.md).

## Runnable demos

Runnable, self-contained programs for each module live in the
[`demos/`](https://github.com/kmolan/multicalc-rust/tree/main/demos) crate. See
[demos/README.md](https://github.com/kmolan/multicalc-rust/blob/main/demos/README.md). Run one
with:

```sh
cargo run -p multicalc-demos --example <name>
```

## Feature flags

- `alloc` (off by default): enables the heap-based methods for inputs too large for the stack.
  See [Heap allocation](#heap-allocation).

## Heap allocation

The library allocates nothing by default: every type is fixed-size and lives on the stack. Turning
on `alloc` pulls in `extern crate alloc` and unlocks exactly two things:

- `estimation::ParticleFilter`, whose cloud of samples is sized at runtime and so cannot be a
  fixed-size stack type.
- `numerical_derivative::jacobian::Jacobian::get_on_heap`, which returns a `Vec<Vec<_>>` for
  Jacobians too large to sit on the stack. The stack-allocated `get` is always available.

Nothing else changes: `no_std`, `forbid(unsafe_code)`, and the no-panic rules hold either way, and
the feature never pulls in `std`.

## MSRV and edition

Edition 2024, minimum supported Rust version **1.85**.

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
