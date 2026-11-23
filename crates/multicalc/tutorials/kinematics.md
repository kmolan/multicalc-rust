# Kinematics

Differential-drive kinematics, pose integration on SE(2), and jointed robots. Fixed-size, no
allocation, no panics, and generic over the `Numeric` scalar.

## Differential drive

The body twist has two DOF, `(v, ω)`, the same count as the wheels. The map is a bijection, so
both round trips are exact. With wheel radius `r` and track width `b`:

```text
v = r·(ω_R + ω_L)/2        ω = r·(ω_R − ω_L)/b
```

- `DifferentialDrive::new(wheel_radius, track_width)`: the only fallible call in this part. After
  it checks the geometry, each map is total.
- `WheelVelocities` / `BodyTwist`: rates, related by `forward` and `inverse`. `BodyTwist` is the
  se(2) twist restricted to `(v, ω)`; it has no lateral field. `to_tangent` gives `[v, 0, ω]`.
  `project_tangent` drops the lateral component of an se(2) tangent, and `tangent_slip` returns it.
- `WheelRotations` / `BodyArc`: increments over one tick, related by `forward_arc` and
  `inverse_arc`. `WheelRotations` is the encoder reading. `BodyArc` is arc length and heading
  change, the exponential coordinates of the relative pose.
- `wheel_rotations_from_travel` / `wheel_travel` convert between wheel rotation and distance.
- `integrate(pose, arc)`: `pose · exp([Δs, 0, Δθ])`, the exact constant-twist arc.
  `odometry_step` composes `forward_arc` and `integrate`.
- `Unicycle`: the plant `[ẋ, ẏ, θ̇] = [v·cos θ, v·sin θ, ω]` as an ODE right-hand side for
  `Rk4`/`Rk45`.
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

// Encoder path: wheel travel -> wheel rotation -> body arc -> pose.
let left_travel = 0.01;    // metres rolled by each wheel
let right_travel = 0.012;
let rotations = drive.wheel_rotations_from_travel(left_travel, right_travel);

let start = SE2::identity();
let pose = integrate(start, drive.forward_arc(rotations));

// Autodiff through an odometry step: d(pose)/d(arc length).
let arc_length = Dual::variable(0.4);
let turn_rate = Dual::constant(0.3);
let duration = Dual::constant(1.0);

let step = integrate(
    SE2::<Dual<f64>>::identity(),
    BodyTwist::new(arc_length, turn_rate).integrate_over(duration),
);
let dx_ds = step.translation()[0].deriv;
```

`integrate` uses `SE2::exp`, so a straight line (ω = 0) takes the same code path as an arc. There
is no `1/ω` singularity: the value and its derivative stay finite at zero curvature. The arc is
exact for a constant twist at any step size. The only model error is the zero-order hold on the
wheel velocities.

Demo:
[kinematics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/kinematics.rs).

## Robots built from joints

A `KinematicTree` holds joints in topological order. Each joint attaches to the world or to an
earlier joint. Storage is a fixed array of `MAX_JOINTS` joints and a runtime length, so the model
is `Copy` and needs no heap.

- `Joint`: one joint. `revolute`, `prismatic` and `fixed` build the 1-DOF kinds. `continuous` is a
  periodic revolute joint with no travel limit. `floating` is a 6-DOF free joint.
  The `with_*` methods set the anchor, the reference configuration, the travel limits, and the
  armature, damping and friction loss for dynamics.
- `JointParent`: `World`, or an earlier joint by index.
- `KinematicTree`: the model. `try_from_joints` and `push` validate it. After that, a query fails
  only on a bad input: a non-finite reading, an out-of-range frame index, or a state from another
  tree.
- `KinematicTreeState`: the world pose of each joint frame for one configuration, from
  `forward_kinematics`, indexed by joint.
- `KinematicJacobian`: the geometric Jacobian of a chosen frame, `6 × MAX_CONFIG` in `[v; ω]` row
  order, one column per velocity slot. Columns outside the ancestor chain of the frame are zero.
  `geometric_jacobian_at` takes a configuration; `geometric_jacobian` takes solved poses.
- `InverseKinematics`: damped least-squares IK, `Δq = J⁺_λ · e` on the body-frame pose error. The
  damping `λ` ramps in near a singularity. Steps are norm-limited and clamped to the joint limits.
- `SecondaryObjective`: a null-space objective for a redundant chain.

Each joint other than a floating joint takes one configuration slot, a weld included, so the
joint index and the configuration index agree. A floating joint takes seven slots and shifts each
later slot by six.

```rust
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3};

// Planar two-link arm: two hinges about z, 1 m links along x, a weld for the tool frame.
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

// Shoulder at +90°, elbow straight: tool at (0, 2, 0).
let configuration = Vector::new([core::f64::consts::FRAC_PI_2, 0.0, 0.0]);
let state = tree.forward_kinematics(&configuration).unwrap();
let tool = state.pose(2).unwrap().translation();

assert!(tool[0].abs() < 1e-12);
assert!((tool[1] - 2.0).abs() < 1e-12);
```

Each reading is relative to the joint reference configuration (`with_zero_offset`, the MuJoCo
`ref`). The model, not the encoder driver, holds an encoder zero offset.

The whole path is generic over the scalar. A `Dual` pass gives the exact derivative of any frame
pose with respect to any joint reading.

Demo:
[forward_kinematics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/forward_kinematics.rs).

## Continuous joints

A revolute joint with no limit is only unclamped. A `Continuous` joint is periodic: it cannot carry
limits (`ContinuousJointHasLimits`), and configuration distance at it is the shortest arc about ±π.
Typical uses are wheels, turrets and wrist rolls. `multicalc-robot-model` maps an MJCF hinge with
no resolved range and a URDF `continuous` joint to this kind.

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

// Forward kinematics does not wrap the reading.
let state = tree.forward_kinematics(&Vector::new([7.0])).unwrap();
assert!((state.pose(0).unwrap().rotation().log()[2] - (7.0 - 2.0 * core::f64::consts::PI)).abs() < 1e-12);

// 3 rad to -3 rad is 2*pi - 6 the short way, not 6.
let distance = tree.configuration_distance(&Vector::new([3.0]), &Vector::new([-3.0]));
assert!((distance - (2.0 * core::f64::consts::PI - 6.0)).abs() < 1e-12);
```

`configuration_distance` is the root-sum-square of per-joint terms, with metres and radians mixed:

| Joint kind | Term |
|---|---|
| `Revolute`, `Prismatic` | Plain difference |
| `Continuous` | Shortest arc |
| `Floating` | Translation, plus `SO3::log` of the relative rotation |
| `Fixed` | None |

`SecondaryObjective::PreferredPosture` uses the same shortest-arc error, so at a continuous joint
the posture bias goes the short way.

## Floating joints

`Joint::floating` is a 6-DOF free joint (MJCF `<freejoint>`, URDF `floating`). It takes no axis
or anchor. Its reading is seven numbers, a position and a unit quaternion, and its rate is six
numbers; see `FreeJointState`. A floating joint must be the first joint and attach to the world
(`FloatingJointNotAtRoot`).

`KinematicTree<MAX_JOINTS, MAX_CONFIG, T>` has a second capacity, `MAX_CONFIG`, for the
configuration length. Without a floating joint, set `MAX_CONFIG = MAX_JOINTS`. With a floating
base, set `MAX_CONFIG = MAX_JOINTS + 6`.

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

// Position, then a scalar-first quaternion: the identity pose.
let state = tree
    .forward_kinematics(&Vector::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    .unwrap();
assert!(state.pose(0).unwrap().translation().norm() < 1e-12);
```

## Loading a model from a file

`multicalc-robot-model` reads MuJoCo MJCF (as in the MuJoCo Menagerie) and URDF (as in most ROS
robots) into the same types. `load_path` selects the reader from the file extension.

```rust,ignore
use multicalc_robot_model::load_path;

let model = load_path(std::path::Path::new("third_party/menagerie/franka_emika_panda/panda.xml"))?;

// 11 bodies: seven hinges, two finger slides, two welds.
assert_eq!(model.body_count(), 11);

// The chain from the world to the hand.
let arm = model.kinematic_tree_to::<9, 9>("hand")?;
let state = arm.forward_kinematics(&Vector::zeros())?;
```

URDF differs from MJCF in two ways:

- URDF states a link mass or omits it, so `inertia()` returns `None` for a link with no mass. The
  MoveIt Panda is kinematics-only and states no mass.
- URDF can couple one joint to another (mimic joints). A tree of independent joints cannot express
  this, so take the arm of a model with a coupled gripper through `kinematic_tree_to`.

```rust,ignore
let model = load_path(std::path::Path::new("third_party/moveit_resources_panda/panda.urdf"))?;

// 12 bodies: the URDF has a bare frame between the last arm link and the hand.
assert_eq!(model.body_count(), 12);
assert!(model.bodies().iter().all(|body| body.inertia().is_none()));

// The second finger mimics the first, so take the chain to the hand.
let arm = model.kinematic_tree_to::<10, 10>("panda_hand")?;
```

`multicalc-robot-model` is not a dependency of `multicalc`, so these blocks do not compile here.
Its [README](https://github.com/kmolan/multicalc-rust/blob/main/crates/multicalc-robot-model/README.md)
has a demo that compiles, and lists the part of each format that it reads.

The loader carries the travel limits, the reference reading, and the armature, damping and
friction values into the model. `model.ignored()` lists the elements that the reader skipped. The
reader refuses any element that could change a mass, and does not skip it.

## Collision checking

`CollisionQuery` measures sphere and capsule clearance:

- Self-collision between primitives attached to robot frames.
- Environment collision against world-fixed primitives.

Distances are surface to surface, negative on overlap. There is no broad phase: the query checks
each pair. Size the capacities for one or two primitives per link, not for mesh geometry. Use
`exclude_pair` to drop adjacent-link pairs that always touch.

```rust
use multicalc::kinematics::{CollisionQuery, Joint, JointParent, KinematicTree, Primitive};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3};

let about_z = Vector::new([0.0, 0.0, 1.0]);
let tree = KinematicTree::<1, 1, f64>::try_from_joints(
    &[Joint::revolute(about_z, SE3::identity())],
    &[JointParent::World],
)
.unwrap();

// A 0.1 m sphere 1 m out on the link, and a 0.1 m obstacle at (0, 1, 0).
let mut query = CollisionQuery::<1, 1, 0, f64>::new();
let out_along_x = SE3::from_parts(SO3::identity(), Vector::new([1.0, 0.0, 0.0]));
let obstacle_pose = SE3::from_parts(SO3::identity(), Vector::new([0.0, 1.0, 0.0]));
query
    .push_self_primitive(0, Primitive::Sphere { radius: 0.1 }, out_along_x)
    .unwrap();
query
    .push_environment_primitive(Primitive::Sphere { radius: 0.1 }, obstacle_pose)
    .unwrap();

// Centres sqrt(2) m apart, less both radii.
let state = tree.forward_kinematics(&Vector::zeros()).unwrap();
let report = query.check(&state).unwrap();
assert!((report.minimum_clearance - (2.0_f64.sqrt() - 0.2)).abs() < 1e-12);
assert!(report.is_clear(0.05));

// At q = π/2 the spheres coincide: negative clearance.
let turned = tree
    .forward_kinematics(&Vector::new([core::f64::consts::FRAC_PI_2]))
    .unwrap();
assert!(!query.check(&turned).unwrap().is_clear(0.0));
```

Each `Primitive` has an `SE3` pose. A capsule segment lies along the local z of its pose, as in
MJCF. `sphere_sphere_distance`, `capsule_capsule_distance` and `sphere_capsule_distance` give the
closed forms without a query.

To filter IK solutions, do not hook into the solver. Walk the `solutions()` of a
`MultiStartReport`, run `forward_kinematics` on each, and keep the first whose `check` is
`is_clear`.

## Inverse kinematics

`InverseKinematics::solve(&tree, tool_index, target, &seed)` returns a report with
`joint_positions`, `position_error`, and `termination`:

- `Converged`: the pose is within both tolerances.
- `Stalled`: the step fell below the minimum norm, at a joint limit, at a singularity, or with
  the target outside the workspace.
- `IterationBudget`: the iteration budget ran out.

A report that did not converge still carries the best configuration that the solver found. The
`with_*` builder methods set the tolerances, the iteration budget, the damping, the step limits,
the joint weights, and the limit enforcement.

```rust
use multicalc::kinematics::{
    InverseKinematics, InverseKinematicsTermination, Joint, JointParent, KinematicTree,
};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3};

// The planar two-link arm, each hinge limited to ±3.14 rad.
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

// Tool at (1, 1): reachable, elbow at a right angle.
let target = SE3::from_parts(SO3::identity(), Vector::new([1.0, 1.0, 0.0]));
let seed = Vector::new([0.3, 0.3, 0.0]);

let solver = InverseKinematics::<3, f64>::new();
let report = solver.solve(&tree, 2, target, &seed).unwrap();

assert_eq!(report.termination, InverseKinematicsTermination::Converged);
assert!(report.position_error < 1e-6);

// Two elbow branches reach this pose; the seed selects one. Check the pose, not the readings.
let reached = tree.forward_kinematics(&report.joint_positions).unwrap();
let tool = reached.pose(2).unwrap().translation();
assert!((tool[0] - 1.0).abs() < 1e-6 && (tool[1] - 1.0).abs() < 1e-6);
```

For a redundant chain, `with_secondary_objective` sets a null-space objective. It does not
change the task-frame pose.

- `PreferredPosture(q_ref)`: bias toward a reference configuration.
- `JointLimitMargin`: bias toward the middle of each joint range.

Demo:
[3d_arm_ik.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/3d_arm_ik.rs).

## Enumerating branches

One DLS solve converges to the branch of its seed. `MultiStartInverseKinematics` runs up to
`MAX_STARTS` solves and keeps the distinct converged configurations. Two solutions are the same if
their `configuration_distance` is below the `with_distinct_threshold` value.

- `solve_from_seeds` takes a seed array.
- `solve_seeded` runs the base seed first, then jittered seeds from a `RandomSource`. The jitter
  is uniform across the joint range where the joint has limits, and `base ± jitter_span` where it
  does not.

```rust
use multicalc::kinematics::{Joint, JointParent, KinematicTree, MultiStartInverseKinematics};
use multicalc::linear_algebra::Vector;
use multicalc::random::Pcg32;
use multicalc::spatial::{SE3, SO3};

// Six hinges alternating x/y on 0.25 m links, tool welded 0.25 m past the last. 6 DOF against a
// 6-DOF task, so the solutions are discrete branches.
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

// Branch continuity: take the solution nearest the current command.
let nearest = report.closest_to(&tree, &posture).unwrap();
assert!(tree.configuration_distance(&nearest.joint_positions, &posture) < 1e-6);
```

Multi-start does not enumerate the full solution set; only a closed-form solver can. It finds some
branches, at random from jittered seeds or deterministically from given seeds. Deduplication uses
`configuration_distance`, so readings 2π apart are two branches at a `Revolute` joint and one at a
`Continuous` joint.

## Classifying a singularity

`smallest_singular_value` gives σ_min, the distance to rank deficiency. The IK damping ramps
against it. `classify_singularity(threshold)` names the degenerate direction:

| `SingularityKind` | Condition |
|---|---|
| `None` | σ_min at or above the threshold |
| `Positional` | The lost direction is mostly translational |
| `Rotational` | The lost direction is mostly rotational |
| `Mixed` | Neither half carries more than two thirds |

The classes are not wrist, elbow or shoulder: the same classifier runs on planar chains, legs and
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

// A gantry: three slides and hinges about x and y. Rank 5 against a 6-DOF task, degenerate in ω_z.
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

The threshold has the same role as the IK damping threshold: at or above it, the chain counts as
full rank. The classifier
uses only the singular vector of σ_min. If the nullity is more than one, as in any chain with
fewer than six actuated DOF, the reported direction is an arbitrary member of the degenerate
subspace.

---

[Back to the tutorial index](README.md)
