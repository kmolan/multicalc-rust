# multicalc-mjcf

Reads MuJoCo MJCF model files into [`multicalc`](../multicalc)'s robot types.

This is a workspace-internal crate (`publish = false`): it ships alongside `multicalc` but is not
published to crates.io on its own.

## Example: parse the Unitree Go1, run IK

```rust
use multicalc::kinematics::{InverseKinematics, InverseKinematicsTermination, JointKind};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;
use multicalc_mjcf::load_path;

let model = load_path(std::path::Path::new(
    "third_party/menagerie/unitree_go1/go1.xml",
))?;

// The trunk on a free joint, plus four legs of three hinges each.
assert_eq!(model.body_count(), 13);
assert_eq!(model.movable_joint_count(), 13);
assert!(model.has_floating_base());

// <keyframe>/<actuator> aren't parsed; listed in `ignored()`.
assert!(model.ignored().iter().any(|s| s == "keyframe"));

let robot = model.kinematic_tree::<13, 19>()?;

// The trunk's floating joint reads seven numbers -- position, then a scalar-first
// quaternion -- rather than the usual one.
assert_eq!(robot.joint(0).unwrap().kind(), JointKind::Floating);

// axis/damping/range come from the "abduction" default class, two levels under "go1".
// forward_kinematics doesn't read them; they're still parsed correctly.
let front_right_hip = robot.joint(1).unwrap();
assert_eq!(front_right_hip.kind(), JointKind::Revolute);
assert_eq!(front_right_hip.damping(), 1.0);
assert_eq!(front_right_hip.limits(), Some((-0.863, 0.863)));

// The file's own "home" pose: standing, all four legs symmetric.
let standing = Vector::new([
    0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0,
    0.9, -1.8,
]);

// 6-DOF task-space IK for the front-right foot: eighteen degrees of freedom (six
// floating, twelve hinge) against the six-dimensional task, so the trunk and the other
// three legs are free to help place it.
let foot = robot.forward_kinematics(&standing)?.pose(3).unwrap();
let target = SE3::from_parts(foot.rotation(), foot.translation() + Vector::new([0.05, 0.0, 0.05]));

let report = InverseKinematics::<19>::new()
    .with_position_tolerance(1e-6)
    .solve(&robot, 3, target, &standing)?;

assert_eq!(report.termination, InverseKinematicsTermination::Converged); // 3 iterations
# Ok::<(), Box<dyn std::error::Error>>(())
```

The same `RobotModel` also emits Rust source for a build with no XML parser and no filesystem
access at runtime:

```rust
use multicalc_mjcf::{GeneratedScalar, RustSourceOptions};

let options = RustSourceOptions::new("unitree_go1")     // name of the generated fn
    .with_scalar(GeneratedScalar::F32)                    // f32 in the generated code (default; use F64 for f64)
    .with_capacity(13)                                     // KinematicTree<13, _, _>; must be >= slot count, default = exact
    .with_configuration_capacity(19);                      // the second const generic; must be >= 19 for the floating base
std::fs::write("unitree_go1.rs", model.to_rust_source(&options)?)?;
```

A model converts to a `KinematicTree` on request: `kinematic_tree` for the whole body tree,
`kinematic_tree_to` for the chain down to one named body.

## The part of the format this reads

| Construct | What is read |
| --- | --- |
| `<compiler>` | `angle`, `autolimits`, `inertiafromgeom` |
| `<default>` | class blocks and nesting, for `<geom>` and `<joint>`; `childclass` on a body reaches every body below it in the tree |
| `<include>` | followed, relative to the file that pulls it in, when the model is read from a file |
| `<worldbody>` / `<body>` | the tree, `pos` and `quat` |
| `<inertial>` | `pos`, `quat`, `mass`, `diaginertia`, `fullinertia` |
| `<joint>` | `hinge` and `slide`, with `axis`, `pos`, `range`, `limited`, `armature`, `damping`, `frictionloss`, `ref`, `springref`, `stiffness` |
| `<freejoint>` | on the top body only, marking the model as floating |
| `<geom>` | `sphere`, `ellipsoid`, `box`, `capsule` and `cylinder`, as a source of mass where a body states none of its own, including `fromto` for the two elongated shapes |

## Passed over

Tendons, equality constraints, actuators, sensors, contact pairs, keyframes, assets and the
`<option>` block are not read. Each top-level section a loaded file carried nothing useful from is
named in [`RobotModel::ignored`](src/lib.rs), sorted and without repeats, so a caller can see what a
model loaded without.

Passing over a section is only ever safe where it cannot change what a body weighs or where it
balances. Anything that could is refused by name instead — see below.

## Refused by name

- A ball joint, or any joint kind other than `hinge`/`slide`.
- A free joint anywhere but the root body.
- More than one joint on a single body.
- A mesh as a source of mass — its inertia cannot be worked out from the file alone.
- A turn written as `euler`, `axisangle`, `xyaxes` or `zaxis` — only `quat` is read.
- A joint marked (or defaulted to) limited that states no `range`.
- A body with neither stated mass properties nor shapes carrying mass.
- A shape giving both `fromto` and `pos` — they can disagree about where it sits.

## Writing a model out as Rust

`RobotModel::to_rust_source(&options) -> Result<String, MjcfError>` renders one `.rs` file
containing a single function that builds the parsed `KinematicTree` from literal values — no XML
parsing, no file I/O, so the generated code runs with `no_std`/no filesystem, e.g. on a
microcontroller. `RustSourceOptions` controls what gets generated:

| Method | Default | What it sets |
| --- | --- | --- |
| `new(function_name)` | required | name of the generated function |
| `with_scalar(GeneratedScalar::F32 \| F64)` | `F32` | float type the tree is built in (`KinematicTree<N, f32>` vs `<N, f64>`) |
| `with_capacity(n)` | `0` = exactly as many slots as the model has | `N` in `KinematicTree<N, _>`; must be `>=` the number of bodies emitted, so a caller can reserve headroom for a model that grows later |
| `with_configuration_capacity(n)` | `0` = `capacity` plus six if the model has a floating base, else `capacity` | the generated tree's second const generic, bounding the configuration vector rather than the joint count |
| `with_tip(name)` | unset = whole model | emit only the chain from the world down to body `name`, e.g. the arm without a gripper |
| `with_header(text)` | empty | a `// text` comment above the generated code (e.g. where it was generated from) |
| `with_documentation(text)` | empty | doc comment (`/// text`) on the generated function |

`tools/qa/src/bin/gen_model_source.rs` uses this to check the Franka arm's model into
`tools/embedded-smoke`, which the on-target forward-kinematics check then builds against directly.

## Licence note

The vendored model files under [`third_party/menagerie`](../../third_party/menagerie) keep their
own upstream licences; see the `LICENSE` file alongside each model.
