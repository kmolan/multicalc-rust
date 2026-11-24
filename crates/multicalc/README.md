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
- **Builds in seconds.** A clean build of the whole crate takes about 2.5 s, debug, release or
  `thumbv7em`, with one dependency, `libm`. Generic code compiles in your crate, only for the
  scalar types that you use. (i7-12650H, Rust 1.100.)

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
| [Planning](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/planning.md) | `GridPlanner` (Dijkstra, A\*, Theta\*), `Rrt`, `RrtStar`, `Prm`. Caller-owned workspaces, no allocation |
| [MLP inference](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/mlp-inference.md) | `Layer`, `Activation`, `ParameterCursor` for trained policies |

### The core, also usable alone

Built on its own `no_std` math core: autodiff, linear algebra and solvers.

| Module | Contents |
| --- | --- |
| [Autodiff](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc/tutorials/scalars-and-automatic-differentiation.md) | Exact derivatives (to order 6 in one variable, 3 in several), `Jacobian`, `Hessian` |
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
