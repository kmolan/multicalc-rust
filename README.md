# multicalc

[![Website](https://img.shields.io/badge/website-kmolan.github.io-e8683a)](https://kmolan.github.io/multicalc-rust/)
[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**The whole robotics stack, on bare metal.** Estimation, planning, kinematics, dynamics and control
in one `no_std` Rust crate. No heap, no panics, no unsafe.

https://github.com/user-attachments/assets/ed45ccb5-ca95-4e4b-8399-27d09284b220

*The [live showcases](demos#live-showcases). Each number on screen is measured live, inside a 1 ms
tick.*

## Highlights

- **1 kHz control loops.** Fixed-size types, no heap, and bounded work for each call.
- **Tested on real silicon.** Arm Cortex-M4F, Arm Cortex-M33 and RISC-V boards. CI also tests the
  `x86_64` and `aarch64` hosts and four bare-metal targets under QEMU.
- **Checked against reference libraries.** 183 checks against NumPy, SciPy, MuJoCo, Pinocchio,
  mpmath and FilterPy. See the [benchmarks](benchmarks).
- **Heap only on request.** The default build needs no allocator. The `alloc` feature adds the particle
  filter, Monte Carlo localization and `DynamicOccupancyGrid`.
- **Safe and panic-free.** `#![forbid(unsafe_code)]`, no C dependencies, and a typed error from each
  fallible call.

## What it does

### Robotics

| Module | Contents |
| --- | --- |
| [Estimation](crates/multicalc/tutorials/estimation.md) | Linear, extended and unscented `KalmanFilter`, `ErrorStateKalmanFilter`, `MahonyFilter`, `MadgwickFilter`; `ParticleFilter` and `MonteCarloLocalizer` (`alloc`) |
| [Control](crates/multicalc/tutorials/control.md) | `Pid`, `Lqr`, `GeometricAttitudeController`, pure pursuit, `FollowTheGap`, `ComputedTorqueController`, joint and Cartesian impedance, `JointPdController` |
| [Spatial math](crates/multicalc/tutorials/spatial-quaternions-and-lie-groups.md) | `Quaternion`, `SO2`/`SE2`/`SO3`/`SE3` with Jacobians, `Twist`, `Wrench`, `SpatialInertia` |
| [Rigid-body dynamics](crates/multicalc/tutorials/rigid-body-dynamics.md) | `RigidBody`, `ArticulatedBody`: RNEA, CRBA, ABA, momentum observer. Models from MJCF or URDF |
| [Kinematics](crates/multicalc/tutorials/kinematics.md) | `KinematicTree`, forward and inverse kinematics, `CollisionQuery` |
| [Plant](crates/multicalc/tutorials/plant.md) | `MultirotorMixer`, `RotorLag`, `PositionServo` |
| [Motion](crates/multicalc/tutorials/motion.md) | `PolylinePath`, `MinimumSnapPlanner`, `MotionProfilePlanner` |
| [Mapping](crates/multicalc/tutorials/mapping.md) | `OccupancyGrid`, `DynamicOccupancyGrid` (`alloc`), `LogOddsGrid`, `DistanceField`, `CostGrid` |
| [Planning](crates/multicalc/tutorials/planning.md) | `GridPlanner` (Dijkstra, A\*, Theta\*), `Rrt`, `RrtStar`, `Prm`. Caller-owned workspaces, no allocation |
| [MLP inference](crates/multicalc/tutorials/mlp-inference.md) | `Layer`, `Activation`, `ParameterCursor` for trained policies |

### Core math

| Module | Contents |
| --- | --- |
| [Autodiff](crates/multicalc/tutorials/scalars-and-automatic-differentiation.md) | Exact derivatives (to order 6 in one variable, 3 in several), `Jacobian`, `Hessian` |
| [Linear algebra](crates/multicalc/tutorials/linear-algebra.md) | `Matrix`, `Vector`: LU, Cholesky, QR, SVD, eigendecomposition, `expm`, Riccati and Lyapunov solvers |
| [Least squares](crates/multicalc/tutorials/least-squares-optimization.md) | `LevenbergMarquardt`, `GaussNewton` |
| [Root finding](crates/multicalc/tutorials/root-finding.md) | Bisection, Brent, Newton, damped line search |
| [Polynomials](crates/multicalc/tutorials/polynomials.md) | `Polynomial`, `PiecewisePolynomial`, `MultivariatePolynomial` |
| [Integration](crates/multicalc/tutorials/integration.md) | Newton-Cotes, Gauss-Legendre, Gauss-Hermite, Gauss-Laguerre |
| [ODE](crates/multicalc/tutorials/ode-integrators.md) | `Rk4`, `Rk45`, `ExponentialMap` |
| [Discretization](crates/multicalc/tutorials/discretization.md) | Zero-order hold, Van Loan, discrete white noise |
| [Signal processing](crates/multicalc/tutorials/signal-processing.md) | `Biquad`, `MovingAverage`, `RunningMedian`, `SavitzkyGolay`, `SlewRateLimiter` |
| [Vector calculus](crates/multicalc/tutorials/vector-calculus.md) | Curl, divergence, line and flux integrals |
| [Approximation](crates/multicalc/tutorials/taylor-approximation.md) | Linear and quadratic Taylor models |
| [Random](crates/multicalc/tutorials/random.md) | `Pcg32`, `RandomSource` |

## Quick start

```rust
use multicalc::prelude::*;
use multicalc::{KalmanFilter, KalmanModel, Matrix, Newton, SE3, SO3, Vector, constant, scalar_fn};

fn main() -> Result<(), CalcError> {
    // One formula: evaluated at f64, and at a dual number for exact derivatives.
    let f = scalar_fn!(|x| x * x * x - constant(2.0) * x);   // x³ − 2x

    let slope = derivative(&f, 2.0);                          // 10, exact
    let root = Newton::new().solve(&f, 2.0)?.root;            // √2
    let area = integral(&|x: f64| f.eval(x), [0.0, 2.0])?;    // 0

    // A quarter turn about z, then a translation.
    let rotation = SO3::exp(Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]));
    let pose = SE3::from_parts(rotation, Vector::new([1.0, 2.0, 3.0]));
    let moved = pose.act(Vector::new([1.0, 0.0, 0.0]));      // (1, 3, 3)

    // A Kalman filter recovers a velocity that it does not measure.
    let model = KalmanModel {
        state_transition: Matrix::new([[1.0, 1.0], [0.0, 1.0]]),
        measurement_model: Matrix::new([[1.0, 0.0]]),
        process_noise: Matrix::new([[0.01, 0.0], [0.0, 0.01]]),
        measurement_noise: Matrix::new([[0.1]]),
    };
    let mut filter = KalmanFilter::new(Vector::new([0.0, 0.0]), Matrix::new([[1.0, 0.0], [0.0, 1.0]]), model);
    filter.predict();
    filter.update(Vector::new([1.0]))?;
    let velocity = filter.state()[1];

    Ok(())
}
```

Each module has its own error enum, and each one converts into `CalcError`.

## Documentation

- **[Website](https://kmolan.github.io/multicalc-rust/)**: demos, silicon timings, latency and accuracy charts.
- **[Tutorials](crates/multicalc/tutorials/README.md)**: one page for each module.
- **[API docs](https://docs.rs/multicalc)**: the full API reference.
- **[Demos](demos)**: a self-checking example for each module, and seven live showcases. Run one with
  `cargo run -p multicalc-demos --example <name>`.
- **[Benchmarks](benchmarks)**: accuracy tables for each module, and latency.

## Repository layout

| Path | Contents |
| --- | --- |
| [`crates/multicalc`](crates/multicalc) | The library |
| [`crates/multicalc-robot-model`](crates/multicalc-robot-model) | MJCF and URDF reader, and a `model_viewer` |
| [`demos`](demos) | Examples and live showcases |
| [`tools/qa`](tools/qa) | Accuracy fixtures and the benchmark tables |
| [`tools/embedded-smoke`](tools/embedded-smoke) | Bare-metal tests under QEMU |
| [`tools/hilshare`](tools/hilshare) | Tests on real boards |
| [`site`](site) | The website |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The crates, the domains, and the layer graph |

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
