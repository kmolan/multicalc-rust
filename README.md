# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Scientific math for real-time embedded systems, in `no_std` stable Rust with near-zero dependencies: state
estimation, control, kinematics, and Lie groups on a calculus, autodiff, and linear-algebra core in one integrated package.
No heap, no panics, no `unsafe` - from a 64-bit server down to a bare-metal microcontroller.**

https://github.com/user-attachments/assets/93dee114-67f6-4124-a20d-88a8be50da6f

*A reel of the live [showcase demos](demos#live-showcases): a 2D robot running particle filter localization + EKF sensor fusion + obstacle avoidance over a 1kHz loop rate — then an 8-link SE(3) arm tracking a moving 3D pose. Every number on
screen is measured live, inside a 1 ms tick.*

## Highlights

- **1 kHz loop rate.** The lead showcase localizes a differential-drive
  robot against a known map with a 2,000-particle filter over 61 noisy lidar beams, then runs a
  5-state extended Kalman filter fusing 100 Hz wheel odometry, a 200 Hz IMU, and 20 Hz GPS while a
  Follow-the-Gap controller laps a course of obstacles — predict, fuse, and plan every millisecond.
- **Exercise the same math from a server to a microcontroller.** Every commit is built and tested on **six targets**:
  the `x86_64` and `aarch64` Linux hosts and on four bare-metal ABIs (`thumbv7em` soft-float,
  `thumbv7em` hardware-FPU, `thumbv6m`, and `riscv32imc`), running the real math under QEMU.
  `no_std`, no-alloc, and no-panic rules hold on each target.
- **Fast, and measured.** Each module's results are verified against established libraries like `numpy`, `scipy`, and `filterpy` fixtures within ~1 ulp, thus validating the rust
  implementation. See the
  [benchmarks](benchmarks).
- **Exact derivatives, not estimates.** Differentiation, Jacobians, Hessians, Newton steps, and
  Levenberg-Marquardt fits use forward-mode automatic differentiation, so derivatives are exact
  to machine precision; finite differences remain available for black-box functions. The extended
  Kalman filter's Jacobians come from autodiff — none are hand-derived.
- **Pure safe and panic-free.** `#![forbid(unsafe_code)]`, no C dependencies, and `unwrap`/
  `panic` denied on library paths; every fallible call returns a typed error. Types are fixed-size
  and stack-allocated, and iteration counts are bounded.
- **One dependency.** `no_std`, no heap by default, with transcendentals from
  [`libm`](https://crates.io/crates/libm).


## What it does

### Robotics and control

- [Estimation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#estimation): linear and extended `KalmanFilter`s (autodiff Jacobians, no hand-derived ones) and a `ParticleFilter` for nonlinear, non-Gaussian problems (`alloc` only).
- [Control](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#control): `Pid` with anti-windup and a filtered derivative, a one-pole low-pass, the `pure_pursuit_curvature` path-following law, and `FollowTheGap` reactive obstacle avoidance over a range scan.
- [Spatial math](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#spatial-quaternions-and-lie-groups): `Quaternion`, the `SO2`/`SE2`/`SO3`/`SE3` Lie groups for 2D and 3D rotations and rigid-body transforms with left and right Jacobians and their inverses on all four, and `Twist`/`Wrench` screw-theory types.
- [Kinematics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#kinematics): differential-drive and unicycle maps between wheel and body motion, with exact SE(2) odometry.
- [Motion](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#motion): `PolylinePath`, a stack-allocated waypoint path with arc-length, closest-point, and lookahead queries.

### Core math

- [Automatic differentiation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#scalars-and-automatic-differentiation): Exact autodiff of any order (total and partial), plus Jacobian and Hessian matrices.
- [Linear algebra](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#linear-algebra): fixed-size, stack-allocated `Matrix` and `Vector` with LU, Cholesky, column-pivoted QR, SVD, and the matrix exponential `expm`: solves, general N×N determinant and inverse, pseudo-inverse, and condition number.
- [Least-squares optimization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#least-squares-optimization): `LevenbergMarquardt` and `GaussNewton` solvers for nonlinear curve fitting.
- [Root finding](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#root-finding): bracketed bisection and Newton solvers for scalar equations and square systems, with an optional damped line search.
- [Integration](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#integration): iterative Newton-Cotes rules (Boole, Simpson, Trapezoidal) and Gaussian quadrature (Legendre, Hermite, Laguerre) over finite, semi-infinite, and infinite limits.
- [ODE integrators](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#ode-integrators): fixed-step `Rk4` and adaptive `Rk45` (Dormand-Prince 5(4)) with PI step control and dense output.
- [Discretization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#discretization): zero-order hold, Van Loan, and discrete white-noise models for continuous-time linear systems.
- [Vector calculus](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#vector-calculus): curl, divergence, and line and flux integrals.
- [Approximation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#taylor-approximation): linear and quadratic Taylor models with goodness-of-fit metrics.
- [Random](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md#random): `Pcg32` and the `RandomSource` trait, a seedable `no_std` generator for the particle filter and for stochastic models.

## Quick start

Two formulas, written once, carried through six modules — each step feeding the next:

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
    let slope = derivative(&f, 2.0_f64);                     // f'(2)  = 10
    let bend = second_derivative(&f, 2.0_f64);               // f''(2) = 12
    let dg_dx = partial(&g, 0, &[1.0_f64, 2.0])?;            // ∂g/∂x at (1, 2)

    // The derivative matrices of those same two formulas.
    let hessian = Hessian::new().evaluate(&g, &[1.0, 2.0])?;            // 2x2 second derivatives
    let both = scalar_fn_vec!(|v: &[f64; 2]| [
        v[0] * v[0] * v[1] + v[0].sin(),
        v[0] * v[0] * v[0] - c(2.0) * v[0],
    ]);
    let jacobian = Jacobian::new().evaluate(&both, &[1.0, 2.0])?;       // 2x2 first derivatives

    // Integration — f again, this time over an interval.
    let area = integral(&|x: f64| f.eval(x), [0.0, 2.0])?;   // ∫₀² f = 0

    // Linear algebra — solve H·x = b with the Hessian computed three lines up.
    let x = hessian.solve(Vector::new([1.0, 2.0]))?;

    // Root finding — Newton on the same f, its derivative supplied by autodiff.
    let root = Newton::new().solve(&f, 2.0)?.root;           // √2 ≈ 1.41421356

    // Rigid-body motion — SO(3)/SE(3), generic over the scalar like everything above.
    let turn = SO3::exp(Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]));  // 90° about z
    let pose = SE3::from_parts(turn, Vector::new([1.0, 2.0, 3.0]));
    let moved = pose.act(Vector::new([1.0, 0.0, 0.0]));      // rotate, then translate → (1, 3, 3)

    // Estimation — a Kalman filter recovering the velocity it never measures.
    let mut filter = KalmanFilter::new(
        Vector::new([0.0, 0.0]),                 // initial state [position, velocity]
        Matrix::new([[1.0, 0.0], [0.0, 1.0]]),   // initial covariance
        KalmanModel {
            state_transition: Matrix::new([[1.0, 1.0], [0.0, 1.0]]),
            measurement_model: Matrix::new([[1.0, 0.0]]),    // position only
            process_noise: Matrix::new([[0.01, 0.0], [0.0, 0.01]]),
            measurement_noise: Matrix::new([[0.1]]),
        },
    );
    filter.predict();
    filter.update(Vector::new([1.0]))?;          // the target moved about 1 m
    let velocity = filter.state()[1];            // recovered, though never measured

    Ok(())
}
```

Every fallible call propagates with `?`: each module has its own error enum, and all of them
convert into the `CalcError` umbrella, so one return type covers a program that mixes modules.

Import with `use multicalc::prelude::*;` for the traits and one-call functions, plus the types you
need from the crate root. The one-call functions (`derivative`, `partial`, `integral`) cover the
common case; the strategy objects behind them are how you pick a different method or step size.

## Tutorial

The [guide](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/GUIDE.md) is a comprehensive tutorial for each module. It shows the full imports,
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
