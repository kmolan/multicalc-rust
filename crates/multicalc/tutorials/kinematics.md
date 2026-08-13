# Kinematics

Maps between wheel motion and body motion for a differential drive, pose integration on SE(2), and
robots built from joints. Fixed-size, no allocation, no panics, and generic over the `Numeric`
scalar.

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

## Robots built from joints

A `KinematicTree` is a jointed robot model: joints in topological order, each attached to the world
or to an earlier joint. Storage is a fixed array of `MAX_JOINTS` joints plus a runtime length, so the
model is `Copy` and needs no heap.

- `Joint`: one single-degree-of-freedom joint. `Joint::revolute`, `Joint::prismatic` and
  `Joint::fixed` build one; the `with_*` methods set the anchor it rotates about, the reference
  configuration, the travel limits, and the armature, damping and friction loss a later dynamics pass
  reads.
- `JointParent`: what a joint is attached to, either `World` or an earlier joint by index.
- `KinematicTree`: the model. `try_from_joints` and `push` are the only fallible calls; with the
  model validated once, every query afterwards is total.
- `KinematicTreeState`: the world pose of every joint frame for one configuration, returned by
  `forward_kinematics` and indexed by joint.
- `KinematicJacobian`: how each joint's rate moves a chosen frame — six rows, three of
  straight-line motion then three of turning, one column per joint. `geometric_jacobian_at` builds
  one from a configuration; `geometric_jacobian` reuses poses you already solved for.
- `InverseKinematics`: the other direction — the joint readings that put a chosen frame at a pose
  you name. Steps toward the target, damping the step near a pose where the arm loses a direction
  of motion, keeping every reading inside its travel.
- `SecondaryObjective`: what an arm with joints to spare should do with the freedom the task
  leaves it — hold a comfortable posture, or stay off its limits.

Every joint takes a configuration slot, welds included, so joint index and configuration index agree.

```rust
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3};

// A planar two-link arm: both joints rotate about z, each link reaches one unit along x, and a
// weld carries the tool frame at the far end.
let about_z = Vector::new([0.0, 0.0, 1.0]);
let along_x = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));

let tree = KinematicTree::<3, f64>::try_from_joints(
    &[
        Joint::revolute(about_z, SE3::identity()),
        Joint::revolute(about_z, along_x),
        Joint::fixed(along_x),
    ],
    &[
        JointParent::World,
        JointParent::Joint(0),
        JointParent::Joint(1),
    ],
)
.unwrap();

// Shoulder at +90 degrees, elbow straight.
let configuration = Vector::new([core::f64::consts::FRAC_PI_2, 0.0, 0.0]);
let state = tree.forward_kinematics(&configuration).unwrap();
let tool = state.pose(2).unwrap().translation();

assert!(tool[0].abs() < 1e-12);
assert!((tool[1] - 2.0).abs() < 1e-12);
```

Readings are counted from each joint's reference configuration (`with_zero_offset`, MuJoCo's `ref`),
so an encoder zero that differs from the model zero is handled by the model rather than by a constant
buried in whatever reads it.

The whole path is generic over the scalar, so pushing a `Dual` through it gives the exact derivative
of any frame's pose with respect to any joint reading, with nothing hand-derived.

Full demo:
[forward_kinematics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/forward_kinematics.rs).

## Working backwards from a pose

```rust
use multicalc::kinematics::{
    InverseKinematics, InverseKinematicsTermination, Joint, JointParent, KinematicTree,
};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3};

// The same planar two-link arm, with each hinge limited to a half turn either way.
let about_z = Vector::new([0.0, 0.0, 1.0]);
let along_x = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
let hinge = |origin| Joint::revolute(about_z, origin).with_limits(-3.14, 3.14);

let tree = KinematicTree::<3, f64>::try_from_joints(
    &[hinge(SE3::identity()), hinge(along_x), Joint::fixed(along_x)],
    &[
        JointParent::World,
        JointParent::Joint(0),
        JointParent::Joint(1),
    ],
)
.unwrap();

// Put the tool at (1, 1): reachable, with the elbow at a right angle.
let target = SE3::from_parts(SO3::identity(), Vector::new([1.0, 1.0, 0.0]));
let seed = Vector::new([0.3, 0.3, 0.0]);

let solver = InverseKinematics::<3, f64>::new();
let report = solver.solve(&tree, 2, target, &seed).unwrap();

assert_eq!(report.termination, InverseKinematicsTermination::Converged);
assert!(report.position_error < 1e-6);

// The answer is not unique — which of the two elbow poses you land on depends on the guess you
// started from. Check the pose it reached, not the readings it chose.
let reached = tree.forward_kinematics(&report.joint_positions).unwrap();
let tool = reached.pose(2).unwrap().translation();
assert!((tool[0] - 1.0).abs() < 1e-6 && (tool[1] - 1.0).abs() < 1e-6);
```

An arm with more joints than the task needs can move without disturbing the frame it is holding.
`with_secondary_objective` says what to do with that freedom — `PreferredPosture` drifts toward a
set of readings you like, `JointLimitMargin` drifts toward the middle of each joint's travel — and
the task is unaffected either way.

Full demo:
[3d_arm_ik.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/3d_arm_ik.rs).


---

[Back to the tutorial index](README.md)
