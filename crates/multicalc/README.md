# multicalc

[![On crates.io](https://img.shields.io/crates/v/multicalc.svg)](https://crates.io/crates/multicalc)
![Downloads](https://img.shields.io/crates/d/multicalc?style=flat-square)
[![CI](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/kmolan/multicalc-rust/actions/workflows/ci.yml)
[![Docs](https://docs.rs/multicalc/badge.svg)](https://docs.rs/multicalc)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**Robotics for microcontrollers.** Sense, plan, estimate and control in `no_std` Rust in one
integrated package. Tested on real hardware: run the same code from your laptop to a Cortex-M4.

## Highlights

- **Tested on real silicon.** Pinocchio-exact robot dynamics, checked on each run, and a full
  1 kHz arm control loop on a 100 MHz Cortex-M4F. CI also tests the `x86_64` and `aarch64` hosts
  and four bare-metal targets under QEMU.
- **1 kHz control loops.** Fixed-size types, no heap, and bounded work for each call.
- **Checked against reference libraries.** 183 checks against NumPy, SciPy, MuJoCo, Pinocchio,
  mpmath and FilterPy. See the
  [benchmarks](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks).
- **Safe and panic-free.** `#![forbid(unsafe_code)]`, no C dependencies, and a typed error from each
  fallible call.
- **Builds in seconds.** A clean build takes about 2.5 s (debug, release or `thumbv7em`), with one
  dependency, `libm` (i7-12650H, Rust 1.100).

## Quick start

### A robot arm

```rust
use multicalc::prelude::*;
use multicalc::{ArticulatedBody, Joint, JointParent, JointPdController, JointReference};
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

    // Forward kinematics: held level, the elbow sits 0.5 m out along x.
    let level = Vector::new([0.0, 0.0]);
    let measured_velocity = Vector::zeros();
    let state = arm.tree().forward_kinematics(&level)?;
    let elbow_position = state.pose(1).map(|pose| pose.translation());
    assert_eq!(elbow_position, Some(Vector::new([0.5, 0.0, 0.0])));

    // One tick of PD with gravity compensation: τ = kp⊙e + kd⊙ė + G(q), G(q) by RNEA.
    let position_gains = Vector::new([400.0, 400.0]);
    let velocity_gains = Vector::new([40.0, 40.0]);
    let controller =
        JointPdController::new(position_gains, velocity_gains)?.with_gravity_compensation(true);
    let reference = JointReference::at_rest(level);
    let torque = controller.torque(&arm, &state, &level, &measured_velocity, &reference)?;

    // On the reference, e = 0 and ė = 0, so τ = G(q): shoulder −m·g·(0.25 + 0.75), elbow −m·g·0.25.
    assert!((torque[0] + 9.81).abs() < 1e-12);
    assert!((torque[1] + 2.4525).abs() < 1e-12);

    Ok(())
}
```

[`multicalc-robot-model`](https://crates.io/crates/multicalc-robot-model) builds the same
`ArticulatedBody` from an MJCF or URDF file.

### Math, poses and filters

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
    let initial_covariance = Matrix::new([[1.0, 0.0], [0.0, 1.0]]);
    let mut filter = KalmanFilter::new(Vector::new([0.0, 0.0]), initial_covariance, model);
    filter.predict();
    filter.update(Vector::new([1.0]))?;
    let velocity = filter.state()[1];

    Ok(())
}
```

Each module has its own error enum, and each one converts into `CalcError`.

## What it does

### Robotics

| Module | Contents |
| --- | --- |
| [Estimation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/estimation.md) | Linear, extended and unscented `KalmanFilter`, `ErrorStateKalmanFilter`, `MahonyFilter`, `MadgwickFilter`; `ParticleFilter` and `MonteCarloLocalizer` (`alloc`) |
| [Control](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/control.md) | `Pid`, `Lqr`, `GeometricAttitudeController`, pure pursuit, `FollowTheGap`, `ComputedTorqueController`, joint and Cartesian impedance, `JointPdController` |
| [Spatial math](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/spatial-quaternions-and-lie-groups.md) | `Quaternion`, `SO2`/`SE2`/`SO3`/`SE3` with Jacobians, `Twist`, `Wrench`, `SpatialInertia` |
| [Rigid-body dynamics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/rigid-body-dynamics.md) | `RigidBody`, `ArticulatedBody`: RNEA, CRBA, ABA, momentum observer. Models from MJCF or URDF |
| [Kinematics](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/kinematics.md) | `KinematicTree`, forward and inverse kinematics, `CollisionQuery` |
| [Plant](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/plant.md) | `MultirotorMixer`, `RotorLag`, `PositionServo` |
| [Motion](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/motion.md) | `PolylinePath`, `MinimumSnapPlanner`, `MotionProfilePlanner` |
| [Mapping](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/mapping.md) | `OccupancyGrid`, `DynamicOccupancyGrid` (`alloc`), `LogOddsGrid`, `DistanceField`, `CostGrid` |
| [Planning](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/planning.md) | `GridPlanner` (Dijkstra, A\*, Theta\*), `Rrt`, `RrtStar`, `Prm` |
| [MLP inference](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/mlp-inference.md) | `Layer`, `Activation`, `ParameterCursor` for trained policies |

### The core, also usable alone

| Module | Contents |
| --- | --- |
| [Autodiff](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/scalars-and-automatic-differentiation.md) | `Dual`, `HyperDual` and `Jet` scalars for exact derivatives, `Jacobian`, `Hessian` |
| [Linear algebra](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/linear-algebra.md) | `Matrix`, `Vector`: LU, Cholesky, QR, SVD, eigendecomposition, `expm`, Riccati and Lyapunov solvers |
| [Least squares](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/least-squares-optimization.md) | `LevenbergMarquardt`, `GaussNewton` |
| [Root finding](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/root-finding.md) | Bisection, Brent, Newton, damped line search |
| [Polynomials](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/polynomials.md) | `Polynomial`, `PiecewisePolynomial`, `MultivariatePolynomial` |
| [Integration](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/integration.md) | Newton-Cotes, Gauss-Legendre, Gauss-Hermite, Gauss-Laguerre |
| [ODE](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/ode-integrators.md) | `Rk4`, `Rk45`, `ExponentialMap` |
| [Discretization](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/discretization.md) | Zero-order hold, Van Loan, discrete white noise |
| [Signal processing](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/signal-processing.md) | `Biquad`, `MovingAverage`, `RunningMedian`, `SavitzkyGolay`, `SlewRateLimiter` |
| [Vector calculus](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/vector-calculus.md) | Curl, divergence, line and flux integrals |
| [Approximation](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/taylor-approximation.md) | Linear and quadratic Taylor models |
| [Random](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/random.md) | `Pcg32`, `RandomSource` |

## Documentation

[Tutorials](https://github.com/kmolan/multicalc-rust/tree/main/crates/multicalc/tutorials) ·
[Demos](https://github.com/kmolan/multicalc-rust/tree/main/demos) ·
[Website](https://kmolan.github.io/multicalc-rust/) ·
[Benchmarks](https://github.com/kmolan/multicalc-rust/tree/main/benchmarks) ·
[Architecture](https://github.com/kmolan/multicalc-rust/blob/main/ARCHITECTURE.md) ·
[Contributing](https://github.com/kmolan/multicalc-rust/blob/main/CONTRIBUTING.md)

## Acknowledgements

The least-squares solvers and QR factorization port MINPACK (`lmder`, `lmpar`, `qrfac`, `qrsolv`;
Moré, Garbow, Hillstrom), after Moré (1978) and Nocedal & Wright, *Numerical Optimization*.

## Citation

```bibtex
@software{kathail_multicalc,
  author = {Kathail, Anmol},
  title  = {multicalc: robotics for microcontrollers},
  url    = {https://github.com/kmolan/multicalc-rust},
}
```

## License

multicalc is licensed under the MIT license.

## Contact

anmolkathail@gmail.com
