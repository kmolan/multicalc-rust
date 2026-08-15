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
  reads; `Joint::continuous` builds one with no travel limit, wrapping past ±π instead of stopping.
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

Every joint takes at least one configuration slot, welds included — joint index and configuration
index agree for every joint except a floating one, which takes seven.

```rust
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3};

// A planar two-link arm: both joints rotate about z, each link reaches one unit along x, and a
// weld carries the tool frame at the far end.
let about_z = Vector::new([0.0, 0.0, 1.0]);
let along_x = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));

let tree = KinematicTree::<3, 3, f64>::try_from_joints(
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

## Continuous joints

A revolute joint with no stated limit is merely never clamped. A continuous joint states the
unboundedness: it can carry no limit at all, and its reading is treated as periodic wherever
configuration distance is measured — shortest arc about ±π rather than the raw difference. Wheels,
turrets and spinning wrist rolls; `multicalc-mjcf` maps a hinge with no resolved range to one.

```rust
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;

let about_z = Vector::new([0.0, 0.0, 1.0]);
let tree = KinematicTree::<1, 1, f64>::try_from_joints(
    &[Joint::continuous(about_z, SE3::identity())],
    &[JointParent::World],
)
.unwrap();

// Readings are not folded back: 7 rad is 7 rad.
let state = tree.forward_kinematics(&Vector::new([7.0])).unwrap();
assert!((state.pose(0).unwrap().rotation().log()[2] - (7.0 - 2.0 * core::f64::consts::PI)).abs() < 1e-12);

// 3 rad to -3 rad is 2*pi - 6 the short way, not 6.
let distance = tree.configuration_distance(&Vector::new([3.0]), &Vector::new([-3.0]));
assert!((distance - (2.0 * core::f64::consts::PI - 6.0)).abs() < 1e-12);
```

`configuration_distance` applies that per joint kind across the whole model: plain difference at a
revolute or prismatic joint, shortest arc at a continuous one, translation plus rotation log at a
floating one, nothing at a weld. `SecondaryObjective::PreferredPosture` uses the same shortest-arc
error, so posture bias at a continuous joint drives the short way round.

## Floating joints

A body free to move in every direction — MJCF's `<freejoint>`, URDF's `floating` — is a joint too,
just a wider one: `Joint::floating` takes no axis or anchor, because none applies. Its reading is
seven numbers, a position and a unit quaternion, rather than the usual one, and its rate is six
rather than one — see `FreeJointState`. A floating joint may only be a tree's first joint, since it
describes a body's connection to the world rather than to another joint.

`KinematicTree` carries a second capacity for this: `KinematicTree<MAX_JOINTS, MAX_CONFIG, T>`,
where `MAX_CONFIG` bounds the configuration vector rather than the joint count. A tree with no
floating joint sets both to the same number; one with a floating base sets `MAX_CONFIG` six higher.

```rust
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;

// One joint, one slot, seven configuration numbers.
let tree = KinematicTree::<1, 7, f64>::try_from_joints(
    &[Joint::floating(SE3::identity())],
    &[JointParent::World],
)
.unwrap();

// Position first, then a scalar-first quaternion: at rest, facing the world axes.
let state = tree
    .forward_kinematics(&Vector::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    .unwrap();
assert!(state.pose(0).unwrap().translation().norm() < 1e-12);
```

## Loading a model from a file

Building a model joint by joint is fine for a two-link arm. For a real robot, read it out of the
file the manufacturer's model ships as. `multicalc-mjcf` reads MuJoCo MJCF files — the format the
MuJoCo Menagerie models use — into the same types.

```rust,ignore
use multicalc_mjcf::load_path;

let model = load_path(std::path::Path::new("third_party/menagerie/franka_emika_panda/panda.xml"))?;

// Eleven bodies: seven turning joints, two sliding fingers, and two welds.
assert_eq!(model.body_count(), 11);

// The arm alone: the chain running from the world down to the hand.
let arm = model.kinematic_tree_to::<9>("hand")?;
let state = arm.forward_kinematics(&Vector::zeros())?;
```

(`multicalc-mjcf` is not a dependency of `multicalc` itself — see its own
[README](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc-mjcf/README.md) for a
demo that actually compiles.)

The travel limits, the reference reading, and the armature, damping and friction figures come across
with the model, so a solver holds each joint inside the travel the file states and a later dynamics
pass has the numbers it needs. What the reader passed over is listed in `model.ignored()`, and
anything that could change a mass is refused rather than ignored — see the
[crate README](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc-mjcf/README.md)
for the part of the format it reads and which models load.

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

let tree = KinematicTree::<3, 3, f64>::try_from_joints(
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

## Enumerating branches

One DLS solve converges to whichever branch its seed falls into. `MultiStartInverseKinematics`
runs up to `MAX_STARTS` solves and keeps the distinct converged configurations, deduplicated by
`configuration_distance` against a threshold you set. Seeds come from an array you supply, or from
`solve_seeded`, which runs the base seed unperturbed and then jitters draws from any
`RandomSource` — uniform across a joint's range where it has limits, `base ± jitter_span` where it
does not.

```rust
use multicalc::kinematics::{Joint, JointParent, KinematicTree, MultiStartInverseKinematics};
use multicalc::linear_algebra::Vector;
use multicalc::random::Pcg32;
use multicalc::spatial::{SE3, SO3};

// Six hinges alternating x/y on 0.25 m links, tool welded 0.25 m past the last: 6 DOF against a
// 6-DOF task, so its solutions are discrete branches.
let about_x = Vector::new([1.0, 0.0, 0.0]);
let about_y = Vector::new([0.0, 1.0, 0.0]);
let link = SE3::from_parts(SO3::identity(), Vector::new([0.0, 0.0, 0.25]));

let mut tree = KinematicTree::<7, 7, f64>::new();
for index in 0..6 {
    let axis = if index % 2 == 0 { about_x } else { about_y };
    let origin = if index == 0 { SE3::identity() } else { link };
    let parent = if index == 0 {
        JointParent::World
    } else {
        JointParent::Joint(index - 1)
    };
    tree.push(Joint::revolute(axis, origin), parent).unwrap();
}
tree.push(Joint::fixed(link), JointParent::Joint(5)).unwrap();

let posture = Vector::new([0.3, 0.6, -0.4, 0.9, 0.2, -0.5, 0.0]);
let target = tree.forward_kinematics(&posture).unwrap().pose(6).unwrap();

let solver = MultiStartInverseKinematics::<8, 7, f64>::new();
let mut source = Pcg32::<f64>::new(11);
let report = solver
    .solve_seeded(&tree, 6, target, &posture, &mut source, 6)
    .unwrap();

assert_eq!(report.attempts(), 6);
assert!(report.len() >= 2); // several configurations reach the same pose

// Branch continuity: hold to whichever solution is nearest the configuration already commanded.
let nearest = report.closest_to(&tree, &posture).unwrap();
assert!(tree.configuration_distance(&nearest.joint_positions, &posture) < 1e-6);
```

Nothing here is exhaustive. Without a closed-form solver there is no enumeration of the full
solution set — multi-start finds some branches, probabilistically from jittered seeds or
deterministically from chosen ones. Note also that dedup runs through `configuration_distance`, so
readings 2π apart are two branches on a `Revolute` joint and one on a `Continuous` one.

## Classifying a singularity

`smallest_singular_value` gives σ_min — the distance to a rank deficiency, and what the solver
ramps its damping against. `classify_singularity` names the degenerate direction instead:
`Positional` where the lost twist direction is predominantly translational, `Rotational` where it
is predominantly angular, `Mixed` where neither half carries two thirds, `None` above the
threshold. Not wrist/elbow/shoulder — the same classifier runs on planar chains, legs and
floating-base trees.

```rust
use multicalc::kinematics::{
    JacobianFrame, Joint, JointParent, KinematicTree, SingularityKind,
};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;

let about_x = Vector::new([1.0, 0.0, 0.0]);
let about_y = Vector::new([0.0, 1.0, 0.0]);
let about_z = Vector::new([0.0, 0.0, 1.0]);

// A gantry: three slides plus hinges about x and y. Rank 5 against a 6-DOF task, degenerate in
// omega_z.
let tree = KinematicTree::<6, 6, f64>::try_from_joints(
    &[
        Joint::prismatic(about_x, SE3::identity()),
        Joint::prismatic(about_y, SE3::identity()),
        Joint::prismatic(about_z, SE3::identity()),
        Joint::revolute(about_x, SE3::identity()),
        Joint::revolute(about_y, SE3::identity()),
        Joint::fixed(SE3::identity()),
    ],
    &[
        JointParent::World,
        JointParent::Joint(0),
        JointParent::Joint(1),
        JointParent::Joint(2),
        JointParent::Joint(3),
        JointParent::Joint(4),
    ],
)
.unwrap();

let jacobian = tree
    .geometric_jacobian_at(&Vector::zeros(), 5, JacobianFrame::World)
    .unwrap();

assert_eq!(
    jacobian.classify_singularity(1e-2).unwrap(),
    SingularityKind::Rotational
);
```

The threshold is the solver's own damping threshold: at or above it, the chain reads as full rank.
Only the direction belonging to σ_min is classified, so under a nullity above one — any chain with
fewer than six actuated DOF — the reported direction is an arbitrary member of the degenerate
subspace.

---

[Back to the tutorial index](README.md)
