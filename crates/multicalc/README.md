# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**Robotics for microcontrollers.** Sense, plan, estimate and control in `no_std` Rust in one
integrated package. Tested on real hardware: run the same code from your laptop to a Cortex-M4.

## Why use it

- **1 kHz loop rates:** no heap, fixed-size types and bounded work per call. A full robotics
  control loop runs at 1 kHz.
- **Tested on six targets:** each commit on `main` is built and tested on the `x86_64` and
  `aarch64` Linux hosts and on four bare-metal ABIs (`thumbv7em` soft-float, `thumbv7em`
  hardware-FPU, `thumbv6m`, `riscv32imc`). The bare-metal targets run the real math under QEMU.
  The `no_std`, no-alloc and no-panic rules hold on each target.
- **Measured against external references:** each module is checked against fixtures from
  `mpmath`, `numpy`, `scipy`, `filterpy`, MuJoCo and Pinocchio, each to a stated tolerance. See
  the [benchmarks](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/README.md).
- **Safe and panic-free:** `unsafe_code` is forbidden, there are no C dependencies, and
  `unwrap`/`expect`/`panic` are denied on library paths. Each fallible call returns a typed error.
  Types are fixed-size and stack-allocated, and iteration counts are bounded.
- **Builds in seconds:** a clean build of the whole crate takes about 2.5 s, debug, release or
  `thumbv7em`, with one dependency, `libm`. Generic code compiles in your crate, only for the
  scalar types that you use. (i7-12650H, Rust 1.100.)

## What it does

### Robotics and control

- [Estimation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/estimation.md): `KalmanFilter`, `ExtendedKalmanFilter` (autodiff Jacobians) and `UnscentedKalmanFilter` (no derivatives), an `ErrorStateKalmanFilter` that fuses an IMU with position and heading fixes, `MahonyFilter` and `MadgwickFilter` for attitude, and a `ParticleFilter` for nonlinear, non-Gaussian problems with `MonteCarloLocalizer` on top of it (both `alloc` only).
- [Control](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/control.md): `Pid` control, infinite horizon `Lqr`, `GeometricAttitudeController` for drones, the pure-pursuit path-following law (`pure_pursuit_curvature`), and `FollowTheGap` reactive obstacle avoidance. Model-based torque control on any `ArticulatedBody`: `ComputedTorqueController` (feedback linearization through `H(q)`), `JointImpedanceController` and `CartesianImpedanceController` (spring-damper in joint and tool space), and `JointPdController` (PD with optional gravity compensation).
- [Spatial math](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/spatial-quaternions-and-lie-groups.md): `Quaternion`, the `SO2`/`SE2`/`SO3`/`SE3` Lie groups with left/right Jacobians and inverses, and `Twist`/`Wrench` spatial algebra in `[v; ω]` — Plücker transforms, 6×6 adjoints, motion and force cross products, and `SpatialInertia` momentum, bias wrench, energy and composition.
- [Rigid-body dynamics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/rigid-body-dynamics.md): `RigidBody` for a single body, `ArticulatedBody` for a jointed robot — inverse dynamics, the joint-space inertia matrix and forward dynamics (RNEA, CRBA, ABA) with armature, viscous damping and Coulomb friction, checked against Pinocchio, and a `MomentumObserver` for contact detection. Both read from an MJCF or URDF file with `multicalc-robot-model`.
- [Plant](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/plant.md): Actuator models between a command and the wrench a body feels — `MultirotorMixer` maps a wrench to rotor thrusts and back, `RotorLag` a rotor's first-order thrust lag, and `PositionServo` a position-commanded joint's second-order servo, discretized exactly.
- [Kinematics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/kinematics.md): `KinematicTree` for revolute/prismatic/continuous/fixed/floating chains and forward and inverse kinematics, generic over the scalar with autodiff, built by hand or read from an MJCF or URDF file; a damped-least-squares SE(3) pose solver with joint limits and null-space redundancy resolution.
- [Collision checking](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/kinematics.md#collision-checking): `CollisionQuery` for sphere/capsule proximity — primitives on tree frames against each other and against world-fixed obstacles, with pair exclusions and fixed capacities.
- [Motion](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/motion.md): `PolylinePath` for waypoint paths with arc-length, closest-point, and lookahead queries, `MinimumSnapPlanner` for the smoothest trajectory through them, and `MotionProfilePlanner` for jerk-limited point-to-point moves with multi-axis synchronization.
- [Mapping](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/mapping.md): bit-packed `OccupancyGrid` and heap `DynamicOccupancyGrid`, `LogOddsGrid` integrating scans in log-odds, `ScanGeometry` with whole-scan casting, exact Euclidean `DistanceField` with its gradient, and `CostGrid` inflation.
- [Planning](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/planning.md): `GridPlanner` for Dijkstra, A\*, weighted A\* and any-angle Theta\*, priced by `UniformCost` or `CostmapCost`; `Rrt`, `RrtStar` and `Prm` over a `StateSpace`. Caller-owned workspaces, no allocation.
- [MLP inference](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/mlp-inference.md): `Layer` and `Activation` run a trained MLP policy forward over borrowed `MatrixView`/`VectorView` parameters, and `ParameterCursor` walks a flat export one layer at a time, with no copy and no allocation.

### The core, also usable alone

Built on its own `no_std` math core: autodiff, linear algebra and solvers.

- [Automatic differentiation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/scalars-and-automatic-differentiation.md): `Dual`, `HyperDual` and `Jet` scalars for exact forward-mode derivatives (total to order 6, partial to order 3), finite differences of any order, and Jacobian and Hessian matrices.
- [Linear algebra](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/linear-algebra.md): fixed-size, stack-allocated `Matrix` and `Vector` with LU, Cholesky, column-pivoted QR, SVD, symmetric eigendecomposition, and the matrix exponential `expm`. General N×N determinant and inverse, pseudo-inverse, eigenvalue clamping, zero-copy `MatrixView` / `VectorView`, `solve_discrete_riccati` and `solve_discrete_lyapunov`.
- [Least-squares optimization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/least-squares-optimization.md): `LevenbergMarquardt` and `GaussNewton` solvers for nonlinear curve fitting.
- [Root finding](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/root-finding.md): bracketed `Bisection` and `Brent`, and `Newton` and `NewtonSystem` for scalar equations and square systems, with an optional backtracking line search.
- [Polynomials](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/polynomials.md): `Polynomial` for evaluation with any number of derivatives in one pass, arithmetic, calculus, fitting and real roots; `PiecewisePolynomial` for curves made of pieces; and `MultivariatePolynomial` for several variables with symbolic partial derivatives.
- [Integration](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/integration.md): iterative Newton-Cotes rules (Boole, Simpson, Trapezoidal) and Gaussian quadrature (Legendre, Hermite, Laguerre) over finite, semi-infinite, and infinite limits.
- [ODE integrators](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/ode-integrators.md): fixed-step `Rk4` and adaptive `Rk45` (Dormand-Prince 5(4)) with PI step control and dense output, plus `ExponentialMap` for orientation on SO(3).
- [Discretization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/discretization.md): zero-order hold, Van Loan, and discrete white-noise models for continuous-time linear systems.
- [Signal processing](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/signal-processing.md): `Biquad` low-pass, high-pass, band-pass, and notch filters, with cascades, motor-harmonic notches, and per-channel filtering. Plus `MovingAverage`, `RunningMedian`, `SavitzkyGolay` smoothing, `Deadband`, `Hysteresis` and `SlewRateLimiter` conditioning.
- [Vector calculus](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/vector-calculus.md): curl, divergence, and line and flux integrals.
- [Approximation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/taylor-approximation.md): linear and quadratic Taylor models with goodness-of-fit metrics.
- [Random](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/random.md): `Pcg32` and the `RandomSource` trait, a seedable `no_std` generator for the particle filter and for stochastic models.

## Quick start

### A robot arm

```rust
use multicalc::prelude::*;
use multicalc::{ArticulatedBody, ComputedTorqueController, Joint, JointParent, JointReference};
use multicalc::{KinematicTree, Matrix, SE3, SO3, SpatialInertia, Vector};

fn main() -> Result<(), CalcError> {
    // A two-link arm: hinges about y, links 0.5 m long along x, 1 kg each.
    let hinge_axis = Vector::new([0.0, 1.0, 0.0]);
    let elbow_origin = SE3::from_parts(SO3::identity(), Vector::new([0.5, 0.0, 0.0]));
    let shoulder = Joint::revolute(hinge_axis, SE3::identity());
    let elbow = Joint::revolute(hinge_axis, elbow_origin);
    let tree = KinematicTree::<2, 2, f64>::try_from_joints(
        &[shoulder, elbow],
        &[JointParent::World, JointParent::Joint(0)],
    )?;

    let mass = 1.0;
    let centre_of_mass = Vector::new([0.25, 0.0, 0.0]);
    let rotational_inertia = Matrix::from_diagonal([0.01; 3]);
    let link = SpatialInertia::new(mass, centre_of_mass, rotational_inertia)?;
    let gravity = Vector::new([0.0, 0.0, -9.81]);
    let arm = ArticulatedBody::new(tree, &[Some(link), Some(link)], gravity)?;

    // Held level, G(q) by RNEA: shoulder −m·g·(0.25 + 0.75), elbow −m·g·0.25.
    let level = Vector::new([0.0, 0.0]);
    let holding = arm.gravity_torque_at(&level)?;
    assert!((holding[0] + 9.81).abs() < 1e-12);
    assert!((holding[1] + 2.4525).abs() < 1e-12);

    // One control tick: computed torque from the measured state toward the level pose.
    let controller = ComputedTorqueController::from_natural_frequency(20.0, 1.0)?;
    let measured_position = Vector::new([0.1, -0.1]);
    let measured_velocity = Vector::zeros();
    let reference = JointReference::at_rest(level);
    let torque = controller.torque_at(&arm, &measured_position, &measured_velocity, &reference)?;

    Ok(())
}
```

[`multicalc-robot-model`](https://crates.io/crates/multicalc-robot-model) builds the same
`ArticulatedBody` from an MJCF or URDF file.

### Exact derivatives

Two formulas, written once, carried through six modules:

```rust
use multicalc::prelude::*;
use multicalc::{Hessian, Jacobian, KalmanFilter, KalmanModel, Matrix, Newton, SE3, SO3, Vector, constant};
use multicalc::{scalar_fn, scalar_fn_vec};

fn main() -> Result<(), CalcError> {
    // Each formula evaluates at f64 and at any autodiff scalar.
    let f = scalar_fn!(|x| x * x * x - constant(2.0) * x);              // f(x)    = x³ - 2x
    let g = scalar_fn!(|v: &[f64; 2]| v[0] * v[0] * v[1] + v[0].sin()); // g(x, y) = x²y + sin x

    // Derivatives: exact forward-mode autodiff, with no step size and no truncation error.
    let single_point = 2.0_f64;
    let slope = derivative(&f, single_point);                // f'(2)  = 10
    let bend = second_derivative(&f, single_point);          // f''(2) = 12

    let point = [1.0_f64, 2.0];
    let x_index = 0;
    let dg_dx = partial(&g, x_index, &point)?;

    // The derivative matrices of the same two formulas.
    let hessian = Hessian::new().evaluate(&g, &point)?;      // 2x2 second derivatives
    let both = scalar_fn_vec!(|v: &[f64; 2]| [
        v[0] * v[0] * v[1] + v[0].sin(),
        v[0] * v[0] * v[0] - constant(2.0) * v[0],
    ]);
    let jacobian = Jacobian::new().evaluate(&both, &point)?; // 2x2 first derivatives

    // Integration of f over an interval.
    let limits = [0.0, 2.0];
    let area = integral(&|x: f64| f.eval(x), limits)?;       // ∫₀² f = 0

    // Linear algebra: solve H·x = b with the Hessian from above.
    let b = Vector::new([1.0, 2.0]);
    let x = hessian.solve(b)?;

    // Root finding: Newton on f, with the derivative from autodiff.
    let initial_guess = 2.0;
    let root = Newton::new().solve(&f, initial_guess)?.root; // √2 ≈ 1.41421356

    // Rigid-body motion: SO(3)/SE(3), generic over the scalar.
    let quarter_turn_about_z = Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]);
    let translation = Vector::new([1.0, 2.0, 3.0]);
    let start = Vector::new([1.0, 0.0, 0.0]);

    let pose = SE3::from_parts(SO3::exp(quarter_turn_about_z), translation);
    let moved = pose.act(start);                  // rotate, then translate → (1, 3, 3)

    // Estimation: a Kalman filter estimates the velocity from position measurements only.
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
    let velocity = filter.state()[1];             // not measured directly

    Ok(())
}
```

Each module family has its own error enum, and each enum converts into `CalcError`. Thus `?` works
in a function that mixes modules. See [Error handling](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/error-handling.md).

## Tutorials

The [tutorials](https://github.com/kmolan/multicalc-rust/tree/main/crates/multicalc/tutorials)
have one page for each module: the full imports, expected outputs, error paths and demo pointers.
Each page compiles as a doctest.

## Accuracy

The `multicalc-qa` crate checks each module against external-library fixtures (`mpmath`, `numpy`,
`scipy`, `filterpy`, MuJoCo, Pinocchio). The generated per-module tables give the operation, the
tolerance and the reference. See the
[benchmarks index](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks/README.md).

## Runnable demos

The [`demos/`](https://github.com/kmolan/multicalc-rust/tree/main/demos) crate has a runnable
program for each module. See
[demos/README.md](https://github.com/kmolan/multicalc-rust/blob/main/demos/README.md). To run one:

```sh
cargo run -p multicalc-demos --example <name>
```

## Feature flags

- `alloc` (off by default): adds `extern crate alloc` and the heap types listed below.

The library allocates nothing by default. Each type is fixed-size and lives on the stack. The
`alloc` feature adds:

- `ParticleFilter` and its `GaussianLikelihood`, `Likelihood` and `ResamplingScheme`, because the
  particle count is set at runtime.
- `MonteCarloLocalizer`, `BeamModel`, `InitialParticleCloud` and `LikelihoodFieldModel`, which
  build on the particle filter.
- `DynamicOccupancyGrid`, a heap-backed grid for large maps.
- `Jacobian::evaluate_on_heap`, which returns a `Vec<Vec<T>>` for Jacobians too large for the
  stack. The stack-allocated `Jacobian::evaluate` is always available.

The `no_std`, `forbid(unsafe_code)` and no-panic rules hold with or without `alloc`. The feature
never pulls in `std`.

## MSRV and edition

Edition 2024, minimum supported Rust version **1.92**.

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
