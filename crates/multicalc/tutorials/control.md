# Control

Feedback controllers and steering laws. Everything is fixed-size, does not allocate, and is generic
over the `Numeric` scalar, so the same code runs at `f32` on a microcontroller. Angles are radians in
the robot body frame, from the +x axis, positive counter-clockwise. Each controller validates its
configuration at construction.

- `Pid`: three gains and a fixed timestep.
  - `with_output_limits` clamps the output. Conditional integration stops integral windup: while
    the output is saturated and the error pushes it further into the limit, the integral holds.
  - `with_derivative_filter` puts a one-pole low-pass on the derivative term, which makes a D gain
    usable on a noisy measurement. `OnePoleLowPass` and sharper filters are in
    [Signal processing](signal-processing.md).
  - The derivative acts on the measurement, not on the error, so a setpoint step gives no
    derivative kick. For a constant setpoint, the two are equal.
  - `set_gains` retunes with no output step. `resume_from` takes over from a command from another
    source with no output step.
- `Lqr`: discrete infinite-horizon LQR from `state_transition`, `input_model`, `state_cost` and
  `input_cost`. `new` solves the discrete Riccati equation once. `control` and `control_tracking`
  are then matrix-vector products. `certify_stability` proves that the closed loop is stable; it is
  a design-time check, not a per-tick one.
  `with_input_limits` sets per-input limits. `control_limited` and `control_tracking_limited`
  return a `LimitedOutput`: the clamped input and a flag for each input that hit a limit. The law
  has no integrator, so the clamp causes no windup. If an outer `Pid` or an integral state feeds
  the reference, hold that integrator while its input is limited.
- `ComputedTorqueController`: model-based joint tracking. The PD term goes through the joint-space
  inertia, so the tracking error obeys `ë + kd⊙ė + kp⊙e = 0` when the model is exact. The law is one
  RNEA pass per tick: `inverse_dynamics` at the reference acceleration. Coulomb friction is fed
  forward at the desired rate, so measurement noise about zero cannot flip its sign. Viscous
  damping stays at the measured rate, where it is linear and cancels the model exactly.
  `from_natural_frequency` sets the gains from a closed-loop bandwidth and a damping ratio.
- `JointImpedanceController`: a joint-space spring-damper on top of full bias compensation. The
  model keeps its natural inertia, so the joint is compliant, not stiff. Zero stiffness on an axis
  makes that axis free, so the gains must be non-negative, not positive.
- `JointPdController`: joint-space PD, with optional gravity compensation. Use it when the model is
  not reliable. It is the torque-side view of the position-controlled hardware that
  [`PositionServo`](plant.md) models. A discrete high-gain PD has a stability limit that the sample
  rate sets, so tune the gains for the loop rate.
- `CartesianImpedanceController`: a six-axis spring-damper at a tool frame, mapped to joint torque
  through `Jᵀ`. It does no inverse kinematics and does not invert `J`, so it costs one matrix
  product. Near a singularity, it loses authority in some directions and does not command large
  joint motions. `JacobianFrame::Body` makes the stiffness axes tool-fixed; `JacobianFrame::World`
  makes them base-fixed. An optional null-space posture term holds a preferred configuration without
  an effect on the tool, at the cost of one damped pseudo-inverse per tick.
- `GeometricAttitudeController`: rigid-body attitude control on SO(3), so it has no singular
  orientation and no angle wrap. It cancels the gyroscopic torque `ω × (J·ω)` and feeds forward the
  target body rate and its derivative, so it tracks a moving target.
- `thrust_command_from_acceleration`: converts a desired acceleration and heading into an attitude
  and a collective thrust. A body with unidirectional thrust must tilt to accelerate sideways. This
  function connects a position loop to the attitude loop.
  `thrust_command_from_acceleration_with_tilt_limit` also caps the tilt and reports the cap through
  `tilt_bound()`.
- `pure_pursuit_curvature`: the pure-pursuit curvature `κ = 2·sin(α)/L_d` toward a lookahead point,
  in body-frame coordinates. `Curvature::to_body_twist` converts it into a command at a given
  forward speed.
- `FollowTheGap`: reactive avoidance over a forward range scan. It is const-generic on the beam
  count, so the work buffer is on the stack and the beam geometry is fixed at compile time. It
  numbers its beams with the same formula as [`ScanGeometry`](mapping.md), so a scan and the
  steering from it agree beam for beam.

```rust
use multicalc::{FollowTheGap, Pid, pure_pursuit_curvature};
use multicalc::Vector;
use multicalc::SE2;

// A speed loop: PID on the forward speed, output limited, derivative filtered.
let proportional_gain = 2.0_f64;
let integral_gain = 1.0;
let derivative_gain = 0.05;
let timestep = 0.01;
let lowest_output = -1.0;
let highest_output = 1.0;
let derivative_filter_weight = 0.2;

let mut speed_loop = Pid::new(proportional_gain, integral_gain, derivative_gain, timestep)
    .unwrap()
    .with_output_limits(lowest_output, highest_output)
    .unwrap()
    .with_derivative_filter(derivative_filter_weight)
    .unwrap();

let setpoint = 0.4;      // m/s wanted
let measurement = 0.35;  // m/s measured
let command = speed_loop.update(setpoint, measurement);

// Steer toward a point 2 m ahead and 1 m to the left: a left turn, so positive curvature.
let pose = SE2::identity();
let target = Vector::new([2.0, 1.0]);
let lookahead_distance = 2.0;
let curvature = pure_pursuit_curvature(pose, target, lookahead_distance).unwrap();

let forward_speed = 0.4;
let twist = curvature.to_body_twist(forward_speed);

// Reactive avoidance over a 31-beam scan.
let field_of_view = 2.0 * core::f64::consts::PI / 3.0;   // 120°
let max_range = 4.0;
let chassis_width = 0.5;
let free_range_threshold = 0.5;   // a beam at or beyond this range is free
let cruise_speed = 0.4;

let follower: FollowTheGap<31, f64> = FollowTheGap::try_new(
    field_of_view,
    max_range,
    chassis_width,
    free_range_threshold,
    cruise_speed,
)
.unwrap();

// A clear scan drives straight ahead at cruise speed.
let goal_angle = 0.0;
let clear_scan = [4.0; 31];
let output = follower.compute(&clear_scan, goal_angle).unwrap();
assert!(output.heading().abs() < 1e-12);

// A wall all round stops, and reports it.
let walled_in = [0.2; 31];
let blocked = follower.compute(&walled_in, goal_angle).unwrap();
assert!(blocked.is_blocked());
assert_eq!(blocked.body_twist().linear(), 0.0);
```

```rust
use multicalc::SO3;
use multicalc::control::{GeometricAttitudeController, Lqr, thrust_command_from_acceleration};
use multicalc::linear_algebra::{Matrix, Vector};

// A double integrator, one input, 0.1 s timestep.
let state_transition = Matrix::<2, 2>::new([[1.0, 0.1], [0.0, 1.0]]);
let input_model = Matrix::<2, 1>::new([[0.005], [0.1]]);
let state_cost = Matrix::<2, 2>::identity();
let input_cost = Matrix::<1, 1>::new([[1.0]]);

let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();

// Once, at startup: prove that the closed loop is stable.
let certificate = controller.certify_stability().unwrap();
assert!(certificate.cholesky().is_ok());

let bounded = controller
    .with_input_limits(Vector::new([-0.5]), Vector::new([0.5]))
    .unwrap()
    .control_limited(Vector::new([10.0, 0.0]));
assert!(bounded.output()[0].abs() <= 0.5);

// Each tick after that is one matrix-vector product.
let mut state = Vector::new([1.0, 0.0]);
for _ in 0..400 {
    let input = controller.control(state);
    state = state_transition * state + input_model * input;
}
assert!(state.norm() < 1e-6);

// Attitude: a body tilted about x gets a restoring torque.
let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
let attitude_controller = GeometricAttitudeController::new(6.0, 1.2, inertia).unwrap();
let level = SO3::<f64>::identity();
let still = Vector::new([0.0, 0.0, 0.0]);
let tipped = SO3::exp(Vector::new([0.1, 0.0, 0.0]));
let torque = attitude_controller.torque(tipped, still, level, still, still);
assert!(torque[0] < 0.0);

// A flying body must tilt toward +x to accelerate along +x.
let gravity = 9.81;
let facing_along_x = 0.0;
let command =
    thrust_command_from_acceleration(Vector::new([2.0, 0.0, 0.0]), facing_along_x, gravity)
        .unwrap();
assert!(command.thrust_acceleration() > gravity);

// That attitude is the attitude-loop target; a body already there needs no torque.
let settled = attitude_controller.torque(
    command.attitude(),
    still,
    command.attitude(),
    still,
    still,
);
assert!(settled.norm() < 1e-14);
```

`Lqr::new` (Riccati) and `certify_stability` (discrete Lyapunov `A_clᵀ·P·A_cl − P + I = 0`) each
cost `O(n³)` per iteration over an iteration budget. Run them at startup or on the bench, never in
the loop. `certify_stability` returns:

- `LinalgError::DidNotConverge` (in `ControlError::Linalg`) if the closed loop is not stable. A
  solution exists only for a stable loop.
- `LinalgError::NotPositiveDefinite` if rounding leaves the solution without a Cholesky factor.
  The identity weight makes the solution positive definite for each stable loop, for any
  `state_cost`.

## Model-based torque control

The four model-based laws take the model, the measured state and a reference, and return one torque
per joint. The model is any fixed-base [`ArticulatedBody`](rigid-body-dynamics.md), for example a
manipulator, a leg or a gantry. The caller passes the model on each call; the controller does not
hold it. Computed torque is the tracking law: `inverse_dynamics` at the reference acceleration
`q̈_d + kd⊙ė + kp⊙e` cancels the model inertia, and the error obeys `ë + kd⊙ė + kp⊙e = 0`. The
example drives a one-link pendulum to a setpoint, with the forward dynamics of the crate for the
plant:

```rust
use multicalc::control::{ComputedTorqueController, JointReference};
use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::spatial::{SE3, SpatialInertia};

// One hinge about y, a 2 kg link with its centre of mass 0.5 m out along x.
let hinge = Joint::revolute(Vector::new([0.0, 1.0, 0.0]), SE3::<f64>::identity());
let tree = KinematicTree::<1, 1, f64>::try_from_joints(&[hinge], &[JointParent::World])?;
let link = SpatialInertia::new(
    2.0,
    Vector::new([0.5, 0.0, 0.0]),
    Matrix::from_diagonal([0.01, 0.01, 0.01]),
)?;
let body = ArticulatedBody::new(tree, &[Some(link)], Vector::new([0.0, 0.0, -9.81]))?;

// The gains are a closed-loop bandwidth and a damping ratio: kp = w^2, kd = 2*zeta*w.
let controller = ComputedTorqueController::<1, f64>::from_natural_frequency(10.0, 1.0)?;
let reference = JointReference::at_rest(Vector::new([0.6]));

let timestep = 0.001;
let mut position = Vector::zeros();
let mut velocity = Vector::zeros();
for _ in 0..2000 {
    let torque = controller.torque_at(&body, &position, &velocity, &reference)?;
    let acceleration = body.forward_dynamics_at(&position, &velocity, &torque)?;
    velocity = velocity + acceleration.scale(timestep);
    position = position + velocity.scale(timestep);
}
assert!((position[0] - 0.6).abs() < 1e-6);
# Ok::<(), multicalc::CalcError>(())
```

`JointImpedanceController` at the same setpoint behaves differently. Its gains define a
spring-damper, not an error decay rate, and the mechanism keeps its own inertia. Thus an external
force displaces it by `external / stiffness`, and it stays there. This property makes it safe for
work near people, and it is the reason that it is a separate type and not a flag.

## Cascaded flight control

A multirotor uses two loops at two rates:

1. The outer loop holds position. It compares the position with the reference and outputs a
   desired acceleration.
2. `thrust_command_from_acceleration` converts that acceleration into an attitude and a thrust.
3. The inner loop, `GeometricAttitudeController`, holds that attitude. It runs several times faster
   than the outer loop, which is why it is a separate block.

`demos/examples/basics/control_loops.rs` flies a set of waypoints and returns home with this
arrangement, on a path from the minimum-snap planner in [Motion](motion.md). In that demo, the
attitude target is constant between outer-loop updates, so the target body rate is not fed forward.
This move-and-hold arrangement tracks well at usual speeds. A feedforward of the target rate, from
the third and fourth derivatives of the path, gives the last part of the tracking accuracy.

## Follow-the-Gap

`FollowTheGap::compute` makes two passes over the scan:

1. It sanitizes the scan. A non-finite or non-positive beam is a dropped return and reads as free
   space at maximum range.
2. It finds each run of consecutive beams at or above `free_range_threshold`. It discards a run
   whose two bounding returns are closer than `chassis_width`. It scores each other run by
   `span − goal_bias · |aim − goal_angle|`.

The `span` is the usable arc of the run, pulled in from each bounded edge by the angle that the
half-width of the robot subtends at the range of that edge. `aim` is `goal_angle` clamped into that
arc. Together they keep the sides of the robot clear of the obstacles at the gap edges. The output
yaw rate is `steering_gain · heading`, and the forward speed scales with the frontal clearance.

The width test is in metres, not in beams: the same angular gap is wide enough at 4 m but too
narrow at 0.4 m. The law of cosines across the two bounding returns gives the gap width. A run that
reaches an end of the field of view has no bounding return on that side, so it counts as open.

The method is purely reactive, with no map and no memory, so it can oscillate in a three-sided
pocket. When no run is both free and wide enough, it returns a stopped twist with `is_blocked()`
set. The recovery policy, for example a rotation in place until a gap opens, is the responsibility
of the caller.

Full demos:
[control_loops.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/control_loops.rs),
[avoidance.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/avoidance.rs)
and
[2d_localization_obstacle_avoidance.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/2d_localization_obstacle_avoidance.rs)
(a full lap of a marked course, with map-based localization and fusion of odometry, an IMU and GPS).

---

[Back to the tutorial index](README.md)
