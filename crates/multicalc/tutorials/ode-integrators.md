# ODE integrators

Initial-value solvers for `y' = f(t, y)` systems, generic over the state dimension.

- `Rk4`: fixed-step classical Runge–Kutta. `Rk4::step` advances one step; `Rk4::integrate`
  runs a fixed number of steps with a per-step callback.
- `Rk45`: adaptive Dormand–Prince 5(4) with PI step control and cubic-Hermite dense output.
  `solve` integrates to a target time, `solve_on_grid` fills requested sample times via dense
  output, and `for_each_step` exposes each accepted step. Tolerances are set with `with_rtol`
  and `with_atol`.
- `ExponentialMap`: turns an orientation forward by the turn it makes over the step, so what comes
  back is still a true rotation and there is nothing to scale back. `attitude_step` takes a steady
  turn rate, `attitude_step_with_angular_acceleration` uses the rate half way through the step
  instead, and `integrate_attitude` runs a fixed number of steps asking a callback for the rate as
  it goes.

```rust
use multicalc::{Rk4, Rk45};
use multicalc::{Vector, Vector2D};

// Harmonic oscillator y'' = -y as the first-order system [position, velocity].
let f = |_t: f64, y: &Vector2D| Vector::new([y[1], -y[0]]);
let y0 = Vector::new([1.0, 0.0]);

let start_time = 0.0;
let timestep = 0.1;
let y1 = Rk4::step(&f, start_time, &y0, timestep);      // one fixed step

// Adaptive solve over one full period returns to the start [1, 0].
let one_period = core::f64::consts::TAU;
let yf = Rk45::default().solve(&f, start_time, &y0, one_period).unwrap();
assert!((yf[0] - 1.0).abs() < 1e-6 && yf[1].abs() < 1e-6);
```

Dense output samples a whole grid in one pass, and `for_each_step` lets you track a conserved
quantity as the solver runs:

```rust
use multicalc::Rk45;
use multicalc::{Vector, Vector2D};

let f = |_t: f64, y: &Vector2D| Vector::new([y[1], -y[0]]);
let y0 = Vector::new([1.0, 0.0]);
let relative_tolerance = 1e-9;
let absolute_tolerance = 1e-12;
let solver = Rk45::default()
    .with_rtol(relative_tolerance)
    .with_atol(absolute_tolerance);

let start_time = 0.0;
let times = [0.5, 1.0, 2.0, 3.0];
let mut out = [Vector2D::zeros(); 4];
solver.solve_on_grid(&f, start_time, &y0, &times, &mut out).unwrap();
```

Stepping an orientation is the one case where the state is not a plain list of numbers. Four
orientation numbers stepped by `Rk4` or `Rk45` slowly leave unit length and have to be scaled back,
and that scaling is a correction with no physical meaning. `ExponentialMap` avoids the problem
instead of correcting it: the turn made over the step is composed onto the orientation, which keeps
it a true rotation to within rounding.

The trade is accuracy. `attitude_step` takes the turn rate as steady across the step, so its error
shrinks in proportion to the step size; `attitude_step_with_angular_acceleration` and
`integrate_attitude` use the rate half way through and shrink with the square of it. Both are
coarser than `Rk4`, which shrinks with the fourth power — so pick by whether a drifting orientation
or a coarser step hurts more for the run at hand.

```rust
use multicalc::ode::ExponentialMap;
use multicalc::spatial::SO3;
use multicalc::{Vector, Vector3D};

// A body turning at a steady 1 rad/s about its own z axis, for a quarter turn.
let steady = |_time: f64, _orientation: SO3<f64>| Vector::new([0.0, 0.0, 1.0]);
let steps = 100;
let timestep = core::f64::consts::FRAC_PI_2 / steps as f64;

let facing = ExponentialMap::integrate_attitude(
    &steady,
    0.0,
    SO3::identity(),
    timestep,
    steps,
    |_time, _orientation| {},
).unwrap();

// The x axis has swung onto the y axis, and the orientation is still a true rotation.
let swung: Vector3D<f64> = facing.act(Vector::new([1.0, 0.0, 0.0]));
assert!((swung[1] - 1.0).abs() < 1e-12);
assert!((facing.quaternion().norm() - 1.0).abs() < 1e-14);
```

Errors: the adaptive solver returns [`IntegrateError`](error-handling.md): `StepSizeTooSmall`,
`DidNotConverge { steps }`, or `NonFinite`. `attitude_step` and `attitude_step_with_angular_acceleration` have no error path — each step is a fixed handful of small products. `integrate_attitude` returns [`IntegrateError`](error-handling.md) (`NonFinite` or `NonPositiveTimestep`) since it takes a timestep and drives many steps. Full demo (harmonic oscillator plus an acrobot,
a tumbling quadrotor, and an N-body model):
[ode.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/ode.rs).


---

[Back to the tutorial index](README.md)
