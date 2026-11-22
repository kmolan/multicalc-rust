# multicalc-py

Python bindings for [`multicalc`](../multicalc), built with PyO3 and maturin.

This is a workspace-internal crate (`publish = false`): host-only development bindings,
not published to crates.io or PyPI on their own. Import name is `multicalc_py`.

Requires CPython **3.10+**. On **3.14**, the crate needs PyO3 **0.27+** (already pinned
in `Cargo.toml`).

## Develop

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip maturin pytest
maturin develop
```

`maturin develop` needs an active venv (or a `.venv` in this directory / a parent). Prefer
`python -m pytest` so you do not pick up a system `pytest`.

## Test

```bash
python -m pytest tests/ -v
cargo clippy -p multicalc-py --all-targets -- -D warnings
```

PR CI runs the same path on Python 3.12 (`maturin develop` + `pytest`) and
`cargo clippy -p multicalc-py --all-targets -- -D warnings`.

## What is exposed

Host `f64` only. Const-generic sizes are fixed in the class name. This is a slice of
the crate-root API, not every `multicalc` type or method.

| Area | Python names |
| --- | --- |
| Basics | `version()` |
| Errors | `CalcError`, `LinalgError`, `DiffError`, `IntegrateError`, `SolveError`, `KinematicsError`, `SpatialError`, `EstimateError`, `SignalError`, `ControlError`, `DynamicsError`, `PlantError`, `MappingError`, `MotionError`, `PolynomialError` |
| Linear algebra | `Vector2`/`3`/`4`/`6`, `Matrix2`/`3`/`4`/`6` (solve, Cholesky, LU, SVD) |
| Spatial | `Quaternion`, `SO2`, `SO3`, `SE2`, `SE3`, `Twist`, `Wrench`, `SpatialInertia`, `FreeJointState` |
| Control | `Pid`, `Lqr2x1`, `GeometricAttitudeController`, `FollowTheGap5`, `FollowTheGapOutput`, `ThrustCommand`, `pure_pursuit_curvature`, `thrust_command_from_acceleration` |
| Polynomials | `Polynomial2`…`Polynomial8`, `PiecewisePolynomial2`, `MultivariatePolynomial2` |
| Calculus | `derivative`, `second_derivative`, `partial`, `integral` |
| Roots | `bisection`, `brent`, `newton` |
| ODE | `rk4_step`, `rk45_solve`, `exponential_map_attitude_step` (2-state) |
| Discretization | `zoh`, `van_loan`, `q_discrete_white_noise` |
| Signal | `OnePoleLowPass`, `Biquad`, `Deadband`, `SlewRateLimiter`, `MovingAverage4`, `RunningMedian5` |
| Random | `Pcg32` |
| Estimation | `KalmanFilter2x1`, `MadgwickFilter`, `MahonyFilter`, `ParticleFilter2x2` |
| Kinematics / motion / dynamics | `DifferentialDrive`, `KinematicTree2`, `PolylinePath8x2`, `RigidBody` |
| Plant / mapping | `MultirotorMixer4`, `RotorLag4`, `ScanGeometry5`, `DynamicOccupancyGrid` |
| Autodiff / optimization | `Dual`, `HyperDual`, `Jet7`, `GaussNewton2x2`, `LevenbergMarquardt2x2` |

Python residuals and ODE right-hand sides are evaluated as `f64` (finite differences).
`Dual` / `HyperDual` / `Jet7` are separate types; they are not passed through those callbacks.

## Example

```python
import multicalc_py

print(multicalc_py.version())

a = multicalc_py.Vector4([1.0, 2.0, 3.0, 4.0])
b = multicalc_py.Vector4([4.0, 3.0, 2.0, 1.0])
assert a.dot(b) == 20.0

pid = multicalc_py.Pid(2.0, 1.0, 0.0, 0.01)
assert abs(pid.update(1.0, 0.0) - 2.01) < 1e-12
```
