# Control

Feedback controllers and steering laws: a PID with anti-windup and a filtered derivative, an
optimal linear feedback law with a check that the loop it closes settles, attitude control for a
rigid body, the piece that joins those last two together, four model-based torque laws for a
jointed model — computed torque, joint impedance, joint PD, and Cartesian impedance at a tool — the
pure-pursuit path-following law, and Follow-the-Gap reactive avoidance. The
derivative filter itself lives in [Signal processing](signal-processing.md). Fixed-size,
no allocation, no panics, and generic over the `Numeric` scalar, so the same code runs at `f32` on a
microcontroller.

The four model-based laws are the only ones here that take a
[`dynamics::ArticulatedBody`](rigid-body-dynamics.md) — any fixed-base jointed model, not just an
arm — and the model is passed per call rather than held, so a caller who does not need them does not
pay for it.

Angles are radians in the robot body frame, measured from the forward (+x) axis and positive
counter-clockwise. Every controller is configured once, with the configuration validated up front,
and every call after that is total.

- `Pid`: three gains and a fixed timestep. `with_output_limits` clamps the output and stops the
  integral winding up against the clamp; `with_derivative_filter` puts a low-pass on the derivative
  term, which is what makes a D gain usable on a noisy measurement — see
  [Signal processing](signal-processing.md) for the filter itself and for sharper ones. The derivative
  acts on the measurement, so a jump in the setpoint does not send a spike through the derivative
  gain; when the setpoint holds still the two are the same thing. `set_gains` retunes without the
  output stepping, and `resume_from` takes over from a command driven some other way without a step
  either.
- `Lqr`: an optimal linear feedback law. Give it how the state moves, how the input pushes it, and
  what state error and input effort cost; it solves for the best trade-off once and hands back a
  gain. `control` and `control_tracking` are then matrix-vector products. `certify_stability`
  checks the loop actually settles — a design-time check, not a per-tick one.
- `ComputedTorqueController`: model-based joint tracking. The PD term is driven through the
  joint-space inertia, so the tracking error obeys `ë + kd⊙ė + kp⊙e = 0` when the model is exact.
  One recursive-Newton-Euler pass per tick — `inverse_dynamics` evaluated at the reference
  acceleration is the whole law. Coulomb friction is fed forward at the desired rate so it cannot
  flip sign on measurement noise about zero; viscous damping stays at the measured rate, where it is
  linear and cancels the model exactly. `from_natural_frequency` sets the gains from a closed-loop
  bandwidth and damping ratio.
- `JointImpedanceController`: a spring-damper in joint space on top of full bias compensation. The
  model keeps its natural inertia, which is what makes it compliant rather than stiff. Zero stiffness
  on an axis means free along it, so gains are non-negative rather than positive.
- `JointPdController`: joint-space PD, optionally with gravity cancelled. What to reach for when the
  model is not trusted, and the torque-side view of the position-controlled hardware
  [`PositionServo`](plant.md) models. A discrete high-gain PD has a sample-rate-bounded stability
  limit — the gains belong to the loop rate.
- `CartesianImpedanceController`: a six-axis spring-damper at a tool frame, mapped to joint torque by
  the Jacobian transpose. It never solves inverse kinematics and never inverts `J`, so it costs one
  matrix product rather than an iteration, and near a singularity it loses the ability to push in
  some direction rather than producing large joint motions. `JacobianFrame::Body` makes the
  stiffness axes tool-fixed; `JacobianFrame::World` makes them base-fixed. An optional null-space
  posture term holds a comfortable configuration without disturbing the tool, at the cost of a
  damped pseudo-inverse per tick.
- `GeometricAttitudeController`: attitude control for a rigid body, worked on rotations rather than
  on angles, so there is no orientation it breaks down at and no wrap-around to handle. It cancels
  the body's own gyroscopic torque and follows the target's turn rate, so it tracks a moving target.
- `thrust_command_from_acceleration`: what joins the two. A body whose rotors only push one way
  cannot accelerate sideways without tipping first, so a wanted acceleration is really two commands
  — an attitude to reach and a push to apply once there. Give it what a position loop wants and
  which way the body should face, and it hands back both.
- `pure_pursuit_curvature`: the exact `κ = 2·sin(α)/L_d` steering curvature toward a lookahead
  point, written in body-frame coordinates. `Curvature::to_body_twist` turns it into a command at a
  chosen speed.
- `FollowTheGap`: reactive avoidance over a forward range scan. Const-generic on the beam count,
  so the working buffer is stack-allocated and the beam geometry is fixed at compile time. Its beams
  are numbered by the same formula as [`ScanGeometry`](mapping.md), so a scan and the steering
  worked out from it always agree beam for beam.

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

let setpoint = 0.4;      // m/s we want
let measurement = 0.35;  // m/s we have
let command = speed_loop.update(setpoint, measurement);

// Steering toward a point 2 m ahead and 1 m to the left: a left turn, so positive curvature.
let pose = SE2::identity();
let target = Vector::new([2.0, 1.0]);
let lookahead_distance = 2.0;
let curvature = pure_pursuit_curvature(pose, target, lookahead_distance).unwrap();

let forward_speed = 0.4;
let twist = curvature.to_body_twist(forward_speed);

// Reactive avoidance over a 31-beam scan.
let field_of_view = 2.0 * core::f64::consts::PI / 3.0;   // 120°
let max_range = 4.0;
let robot_radius = 0.5;
let clearance = 0.5;     // a gap must beat this to count as free
let cruise_speed = 0.4;

let follower: FollowTheGap<31, f64> =
    FollowTheGap::try_new(field_of_view, max_range, robot_radius, clearance, cruise_speed).unwrap();

// A clear scan drives straight ahead at cruise speed.
let goal_angle = 0.0;
let clear_scan = [4.0; 31];
let output = follower.compute(&clear_scan, goal_angle).unwrap();
assert!(output.heading().abs() < 1e-12);

// A wall all round stops, and says why.
let walled_in = [0.2; 31];
let blocked = follower.compute(&walled_in, goal_angle).unwrap();
assert!(blocked.is_blocked());
assert_eq!(blocked.body_twist().linear(), 0.0);
```

```rust
use multicalc::SO3;
use multicalc::control::{GeometricAttitudeController, Lqr, thrust_command_from_acceleration};
use multicalc::linear_algebra::{Matrix, Vector};

// A cart carrying its speed forward, pushed by one input, at a 0.1 s timestep.
let state_transition = Matrix::<2, 2>::new([[1.0, 0.1], [0.0, 1.0]]);
let input_model = Matrix::<2, 1>::new([[0.005], [0.1]]);
let state_cost = Matrix::<2, 2>::identity();
let input_cost = Matrix::<1, 1>::new([[1.0]]);

let controller = Lqr::new(state_transition, input_model, state_cost, input_cost).unwrap();

// Once, at startup: check the loop this closes actually settles.
let certificate = controller.certify_stability().unwrap();
assert!(certificate.cholesky().is_ok());

// Every tick after that is one matrix-vector product.
let mut state = Vector::new([1.0, 0.0]);
for _ in 0..400 {
    let input = controller.control(state);
    state = state_transition * state + input_model * input;
}
assert!(state.norm() < 1e-6);

// Attitude: a body tipped about x is pushed back the other way.
let inertia = Matrix::<3, 3>::from_diagonal([0.02, 0.02, 0.04]);
let attitude_controller = GeometricAttitudeController::new(6.0, 1.2, inertia).unwrap();
let level = SO3::<f64>::identity();
let still = Vector::new([0.0, 0.0, 0.0]);
let tipped = SO3::exp(Vector::new([0.1, 0.0, 0.0]));
let torque = attitude_controller.torque(tipped, still, level, still, still);
assert!(torque[0] < 0.0);

// Joining the two: a flying body asked to speed up along x has to tip that way first.
let gravity = 9.81;
let facing_along_x = 0.0;
let command =
    thrust_command_from_acceleration(Vector::new([2.0, 0.0, 0.0]), facing_along_x, gravity)
        .unwrap();
assert!(command.thrust_acceleration() > gravity);

// That attitude is what the attitude loop is given, and a body already there needs no torque.
let settled = attitude_controller.torque(
    command.attitude(),
    still,
    command.attitude(),
    still,
    still,
);
assert!(settled.norm() < 1e-14);
```

## Model-based torque control

The four take the model, the measured state and a reference, and hand back one torque per joint. The
model is any [`ArticulatedBody`](rigid-body-dynamics.md) — a manipulator, a leg, a gantry. Computed
torque is the tracking law of the four: `inverse_dynamics` evaluated at the reference acceleration
`q̈_d + kd⊙ė + kp⊙e`, which cancels the model's inertia and leaves the error obeying
`ë + kd⊙ė + kp⊙e = 0`. Driving a one-link pendulum to a setpoint, closing the loop with the crate's
own forward dynamics:

```rust
use multicalc::control::{ComputedTorqueController, JointReference};
use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::spatial::{SE3, SpatialInertia};

// One hinge about y, a 2 kg link balancing half a metre out along x.
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

Swapping `ComputedTorqueController` for `JointImpedanceController` at the same setpoint gives a very
different machine: the gains describe a spring-damper the mechanism hangs on rather than an error
decay rate, and it keeps its own inertia, so a push moves it by `external / stiffness` and it stays
there. That is the property that makes it safe to work beside, and the reason it is a separate type
rather than a flag.

Both the Riccati solve behind `Lqr::new` and the certificate behind `certify_stability` cost
`O(n³)` per pass over a budget of passes. They belong at startup or on the bench, never inside the
loop; the gain they produce is what the loop uses. The certificate reports that a loop does not
settle by failing to find an answer at all, which is the honest verdict rather than a number that
looks plausible. A certificate that is found but is not positive definite means the state cost
cannot see every direction the state can move in — the loop may still settle, but this check cannot
say so.

The three fit together as two loops at two rates, which is how a flying machine is normally built.
The outer one holds position: it takes where the body is against where it should be and says what
acceleration would close the gap. `thrust_command_from_acceleration` turns that into an attitude to
reach and a push to apply, because a body whose rotors only push one way has to tip before it can
accelerate sideways. The inner one holds that attitude. The inner loop runs several times faster
than the outer, which is exactly why the attitude controller is its own block rather than folded
into a single law. `demos/examples/basics/control_loops.rs` flies a set of waypoints and returns
home on this arrangement, with the path coming from
[Motion](motion.md)'s minimum-snap planner.

One limit worth stating: the attitude loop is given a target that holds still between outer-loop
updates, so the target's own turning is not fed forward. That is the usual move-and-hold
arrangement and it tracks a path well at ordinary speeds; reading the target's turn rate out of a
path's third and fourth derivatives is what buys the last of the tracking accuracy, and it is not
part of this module yet.

`FollowTheGap` makes two passes over the scan. First it cleans the scan: a beam that is non-finite
or non-positive counts as a dropped return and reads as free space at maximum range. Then it finds
every run of consecutive beams above the free-range threshold, throws out any run whose two bounding
returns are closer together than the chassis width, and scores the rest by
`span − goal_bias · |aim − goal_angle|`. The `span` is the run's usable arc, pulled in from each
bounded edge by the angle the robot's half-width covers at that edge's range; `aim` is the goal
angle clamped into that arc. Together they keep the robot's sides clear of the obstacles that form
the gap. It then steers toward the winning run with a yaw rate of `steering_gain · heading` and a
forward speed that scales with how far the path ahead is clear.

Measuring the gap in metres rather than in beams is what makes the width test meaningful: the same
angular gap is wide enough to pass at 4 m but too narrow at 0.4 m, and the law of cosines across the
two bounding returns settles it directly. A run that reaches either end of the field of view has no
bounding return on that side, so it counts as open: the sensor saw nothing out there, and inventing
a wall would stop the robot on no evidence.

It is a purely reactive method: with no map and no memory, it can dither in a three-sided pocket.
When no run is both clear and wide enough, it returns a stopped twist with `is_blocked()` set rather
than inventing a heading. The recovery policy — rotating in place until a gap opens, say — is left to
the caller.

Full demos:
[control_loops.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/control_loops.rs),
[avoidance.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/avoidance.rs)
and
[2d_localization_obstacle_avoidance.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/2d_localization_obstacle_avoidance.rs)
(a full lap of a marked course, localizing on a map and fusing odometry, an IMU, and GPS).


---

[Back to the tutorial index](README.md)
