# Kinematics

Maps between wheel motion and body motion for a differential drive, and pose integration on SE(2).
Fixed-size, no allocation, no panics, and generic over the `Numeric` scalar.

The body motion is deliberately 2-DOF, not 3. A differential drive has exactly two degrees of
freedom `(v, ω)` and exactly two wheels, so the map between them is a bijection and both round trips
are exact identities. There is no lateral term to silently drop.

- `DifferentialDrive`: the geometry, a wheel radius and a track width. Constructing it is the only fallible
  operation in the module; with the geometry checked once, every map below is total.
- `WheelVelocities` / `BodyTwist`: motion per second, related by `forward` and `inverse`. A
  `BodyTwist` is the se(2) twist a differential drive can realise, with the lateral term dropped.
- `WheelRotations` / `BodyArc`: motion over one tick, related by `forward_arc` and `inverse_arc`.
  `WheelRotations` is what an encoder reports; a `BodyArc` is arc length and heading change, the
  exponential coordinates of the relative pose.
- `integrate`: advances an `SE2` pose along the exact constant-twist arc.
- `Unicycle`: the same plant as an ODE right-hand side, for `Rk4`/`Rk45`.
- `OdometryStep`: the process model as a `VectorFn`, for autodiff Jacobians.

```rust
use multicalc::kinematics::integrate;
use multicalc::{BodyTwist, DifferentialDrive, WheelVelocities};
use multicalc::Dual;
use multicalc::SE2;

let wheel_radius = 0.036_f64;   // 36 mm
let track_width = 0.235;        // 235 mm between the wheels
let drive = DifferentialDrive::new(wheel_radius, track_width).unwrap();

// Wheel velocities to a body twist, and back exactly.
let wheel_speeds = WheelVelocities::new(10.0, 10.0);        // rad/s on each wheel
let twist = drive.forward(wheel_speeds);                    // v = 0.36 m/s, ω = 0

let body_motion = BodyTwist::new(0.36, 0.0);                // m/s forward, rad/s turn
let wheels = drive.inverse(body_motion);                    // back to (10, 10)

// The encoder path: distance travelled -> wheel rotation -> body arc -> pose.
let left_travel = 0.01;    // metres rolled by each wheel
let right_travel = 0.012;
let rotations = drive.wheel_rotations_from_travel(left_travel, right_travel);

let start = SE2::identity();
let pose = integrate(start, drive.forward_arc(rotations));

// Autodiff straight through an odometry step: d(pose)/d(arc length).
let arc_length = Dual::variable(0.4);   // the quantity being differentiated
let turn_rate = Dual::constant(0.3);
let duration = Dual::constant(1.0);

let step = integrate(
    SE2::<Dual<f64>>::identity(),
    BodyTwist::new(arc_length, turn_rate).integrate_over(duration),
);
let dx_ds = step.translation()[0].deriv;
```

Because `integrate` is built on `SE2::exp`, a straight line (ω = 0) is handled by the same code path
as an arc, with no `1/ω` to blow up: the value and its derivative stay finite at exactly zero
curvature. The arc is exact for a constant twist at any step size, so the modelling error is the
zero-order hold on the wheel velocities rather than integration error.

Full demo:
[kinematics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/kinematics.rs).


---

[Back to the tutorial index](README.md)
