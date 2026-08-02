# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Scientific computing that fits on a microcontroller, built and tested from scratch in one integrated package. Estimation, control, kinematics, Lie groups, calculus, autodiff and linear algebra in stable no_std Rust with no heap, no panics and no unsafe.**

https://github.com/user-attachments/assets/93dee114-67f6-4124-a20d-88a8be50da6f

*A reel of the live [showcase demos](demos#live-showcases): a 2D robot running particle filter localization + EKF sensor fusion + obstacle avoidance over a 1kHz loop rate; then an 8-link SE(3) arm tracking a moving 3D pose. Every number on
screen is measured live, inside a 1 ms tick.*

## Highlights

- **1 kHz loop rates:** No heap, fixed-size types, bounded work per call. Results in a full robotics control loop at 1 kHz.
- **Tested on six embedded targets:** Every commit is built and tested on **six targets**:
  the `x86_64` and `aarch64` Linux hosts and on four bare-metal ABIs (`thumbv7em` soft-float,
  `thumbv7em` hardware-FPU, `thumbv6m`, and `riscv32imc`), running the real math under QEMU.
  `no_std`, no-alloc, and no-panic rules hold on each target.
- **Measured against external references:** Each module's results are verified against established libraries like `numpy`, `scipy`, and `filterpy` fixtures within ~1 ulp, thus validating the rust
  implementation. See the
  [benchmarks](benchmarks).
- **Pure safe and panic-free.** `#![forbid(unsafe_code)]`, no C dependencies, and `unwrap`/
  `panic` denied on library paths; every fallible call returns a typed error. Types are fixed-size
  and stack-allocated, and iteration counts are bounded.


## What it does

### Robotics and control

- [Estimation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#estimation): linear, extended, and unscented `KalmanFilter`s (autodiff Jacobians, no hand-derived ones; the unscented one needs no derivatives at all), an `ErrorStateKalmanFilter` that fuses an IMU with position and heading fixes, `MahonyFilter` and `MadgwickFilter` for attitude estimation, and a `ParticleFilter` for nonlinear, non-Gaussian problems (`alloc` only).
- [Control](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#control): `Pid` control, infinite horizon `Lqr`,; `GeometricAttitudeController` for drones, the `pure pursuit` path-following law; and `FollowTheGap` reactive obstacle avoidance.
- [Spatial math](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#spatial-quaternions-and-lie-groups): `Quaternion`, the `SO2`/`SE2`/`SO3`/`SE3` Lie groups for 2D and 3D rotations and rigid-body transforms with left and right Jacobians and their inverses on all four, and `Twist`/`Wrench` screw-theory types.
- [Rigid-body inertia and the free joint](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#rigid-body-inertia-and-the-free-joint): `SpatialInertia` for a body's mass, balance point, and resistance to spinning, and `FreeJointState` for a body free to move in all six directions — loadable straight from MuJoCo model files with `multicalc-mjcf`.
- [Rigid-body dynamics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#rigid-body-dynamics-and-rotor-mixing): `RigidBody` computes single body ridig body dynamics.
- [Rotor mixing](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#rigid-body-dynamics-and-rotor-mixing): `MultirotorMixer` works out how hard each rotor has to push to give a flying machine the lift and turn it was asked for, and says when a rotor was asked for more than it has.
- [Kinematics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#kinematics): differential-drive and unicycle maps between wheel and body motion, with exact SE(2) odometry.
- [Motion](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#motion): `PolylinePath` for waypoint paths with arc-length, closest-point, and lookahead queries, and `MinimumSnapPlanner` for the smoothest trajectory through them.

### Core math

- [Automatic differentiation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#scalars-and-automatic-differentiation): Exact autodiff of any order (total and partial), plus Jacobian and Hessian matrices.
- [Linear algebra](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#linear-algebra): fixed-size, stack-allocated `Matrix` and `Vector` with LU, Cholesky, column-pivoted QR, SVD, symmetric eigendecomposition, and the matrix exponential `expm`. General N×N determinant and inverse, pseudo-inverse, eigenvalue clamping, `solve_discrete_riccati` and `solve_discrete_lyapunov`.
- [Least-squares optimization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#least-squares-optimization): `LevenbergMarquardt` and `GaussNewton` solvers for nonlinear curve fitting.
- [Root finding](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#root-finding): bracketed bisection and Newton solvers for scalar equations and square systems, with an optional damped line search.
- [Polynomials](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#polynomials): `Polynomial` for evaluation with any number of derivatives in one pass, arithmetic, calculus, fitting and real roots; `PiecewisePolynomial` for curves made of pieces; and `MultivariatePolynomial` for several variables with symbolic partial derivatives.
- [Integration](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#integration): iterative Newton-Cotes rules (Boole, Simpson, Trapezoidal) and Gaussian quadrature (Legendre, Hermite, Laguerre) over finite, semi-infinite, and infinite limits.
- [ODE integrators](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#ode-integrators): fixed-step `Rk4` and adaptive `Rk45` (Dormand-Prince 5(4)) with PI step control and dense output, plus `ExponentialMap`, which is a purely orientation integrator.
- [Discretization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#discretization): zero-order hold, Van Loan, and discrete white-noise models for continuous-time linear systems.
- [Signal processing](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#signal-processing): `Biquad` low-pass, high-pass, band-pass, and notch filters; with cascades, motor-harmonic notches, and per-channel filtering. Plus `MovingAverage`, `RunningMedian`, `SavitzkyGolay` smoothing, `Deadband`, `Hysteresis` and `SlewRateLimiter` conditioning.
- [Vector calculus](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#vector-calculus): curl, divergence, and line and flux integrals.
- [Approximation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#taylor-approximation): linear and quadratic Taylor models with goodness-of-fit metrics.
- [Random](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#random): `Pcg32` and the `RandomSource` trait, a seedable `no_std` generator for the particle filter and for stochastic models.

## Quick start

Two formulas, written once, carried through six modules, each step feeding the next:

```rust
use multicalc::prelude::*;
use multicalc::{Hessian, Jacobian, KalmanFilter, KalmanModel, Matrix, Newton, SE3, SO3, Vector, c};
use multicalc::{scalar_fn, scalar_fn_vec};

fn main() -> Result<(), CalcError> {
    // Written once, evaluated at f64 here and at an autodiff number wherever a derivative is asked
    // for — the formula text never changes.
    let f = scalar_fn!(|x| x * x * x - c(2.0) * x);                     // f(x)    = x³ - 2x
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
        v[0] * v[0] * v[0] - c(2.0) * v[0],
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

Every fallible call propagates with `?`: each module has its own error enum, and all of them convert into the `CalcError` umbrella, so one return type covers a program that mixes modules.

## Full tutorial

Refer to the [guide](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md) for a comprehensive tutorial for each module. It shows the full imports,
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

## Documentation

- **[Guide](crates/multicalc/GUIDE.md)**: A worked section for every module: imports, a runnable
  snippet, error paths, and a demo pointer.
- **[Crate README](crates/multicalc/README.md)** / **[API docs](https://docs.rs/multicalc)**: the
  crates.io page and full API reference, with notes on `no_std`, error handling, and heap allocation.
- **[Examples](demos#start-here)**: Self-contained, self-checking programs for each module in the
  `demos/` crate. Run one with `cargo run -p multicalc-demos --example <name>`.
- **[Benchmarks](benchmarks)**: Per-module accuracy tables and latency measurements, generated
  from the QA fixtures and checked in CI.
- **[Live showcases](demos#live-showcases)**: Five animated Rerun demos, led by a robot that boots not
  knowing where it is, finds itself on a known map with a particle filter, then fuses wheel odometry,
  an IMU, and GPS to lap a course of obstacles on lidar alone. The others are an 8-link SE(3) arm
  tracking a moving 3D pose, a Newton fractal, Fourier epicycles drawing Ferris, and gradient-driven
  marbles, each streaming live-measured speed and accuracy.
- **[QA crate](tools/qa)**: `multicalc-qa` holds the CI-enforced accuracy fixtures and generates the [benchmarks](benchmarks) tables from them.

## Repository layout

The published library crate lives in [`crates/multicalc`](crates/multicalc); the repository
root is a Cargo workspace. Runnable demos live in the dev-only [`demos/`](demos) crate (basics and
live Rerun showcases), and [`tools/embedded-smoke`](tools/embedded-smoke) runs `multicalc` on the
four bare-metal targets (three Cortex-M targets + `riscv32imc`) under QEMU every PR.
[`crates/multicalc-mjcf`](crates/multicalc-mjcf) reads MuJoCo model files into multicalc types, and
[`third_party/menagerie`](third_party/menagerie) holds the model files it is tested against, under
their own upstream licences.

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
