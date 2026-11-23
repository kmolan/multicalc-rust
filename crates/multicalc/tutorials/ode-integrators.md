# ODE integrators

Initial-value solvers for `y' = f(t, y)` systems, generic over the state dimension.

- `Rk4`: fixed-step classical Runge–Kutta. `Rk4::step` advances one step; `Rk4::integrate`
  runs a fixed number of steps with a per-step callback.
- `Rk45`: adaptive Dormand–Prince 5(4) with PI step control and cubic-Hermite dense output.
  - `solve` integrates to a target time.
  - `solve_on_grid` fills sorted sample times from the dense output, with no allocation.
  - `for_each_step` gives each accepted step to a callback.
  - `with_rtol` (default `1e-6`) and `with_atol` (default `1e-9`) set the tolerances.
    `with_first_step`, `with_min_step`, `with_max_step` and `with_max_steps` set the step limits.
- `ExponentialMap`: the Lie-group integrator on SO(3), `R ← R · exp(ω·dt)` with `ω` in the body
  frame.
  - `attitude_step` holds `ω` constant over the step. It is first order.
  - `attitude_step_with_angular_acceleration` uses the midpoint rate `ω + α·dt/2`. It is second
    order.
  - `integrate_attitude` runs a fixed number of midpoint steps. It gets `ω` from a callback at the
    start and at the half step, and gives each node to an observer.

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

Dense output fills a time grid in one pass:

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

A quaternion state under `Rk4` or `Rk45` drifts off unit norm and needs renormalization.
`ExponentialMap` composes the step rotation onto the orientation, so the result stays on SO(3) to
rounding. The cost is order: first or second order, against fourth order for `Rk4`.

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

Errors: the fallible calls return [`IntegrateError`](error-handling.md).

- `Rk4::step` and `Rk4::integrate` have no error path.
- `Rk45` returns `LimitsIllDefined` for a NaN or zero-length span, or for a `solve_on_grid` time
  grid of the wrong length, out of range or out of order. It returns `NonFinite`,
  `StepSizeTooSmall` below `min_step`, or `DidNotConverge { steps }` when `max_steps` runs out.
- `attitude_step` and `attitude_step_with_angular_acceleration` have no error path and do not
  validate their input. Non-finite input can give a non-finite orientation. A non-positive `dt`
  steps backward or not at all.
- `integrate_attitude` returns `NonFinite` for a non-finite `dt` or rate, and
  `NonPositiveTimestep` for `dt ≤ 0`.

Full demo (harmonic oscillator, an acrobot, a quadrotor in free tumble and an N-body model):
[ode.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/ode.rs).


---

[Back to the tutorial index](README.md)
