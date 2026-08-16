# multicalc-robot-model

Reads MuJoCo MJCF and URDF model files into [`multicalc`](../multicalc)'s robot types.

This is a workspace-internal crate (`publish = false`): it ships alongside `multicalc` but is not
published to crates.io on its own.

Both formats parse into one `RobotModel`: a topologically ordered body list carrying name, parent
index, parent-relative transform, spatial inertia where stated, and joint. Converts to a
`KinematicTree` on request — `kinematic_tree` for the whole model, `kinematic_tree_to` for the
root-to-tip chain of a named body.

## Choosing a reader

| Entry point | Dispatch |
| --- | --- |
| `load_path(path)` | file extension: `.urdf` reads URDF, anything else MJCF |
| `load_str(xml)` | root element: `<robot>` reads URDF, `<mujoco>` MJCF |
| `mjcf::load_path` / `mjcf::load_str` | explicit |
| `urdf::load_path` / `urdf::load_str` | explicit |

`RobotModel::format()` reports which reader ran. Each reader is a feature — `mjcf`, `urdf`, both on
by default. The dispatching entry points return `ModelError::FormatNotEnabled` for a reader that is
not compiled in. With neither feature the crate is the model types plus the codegen.

## Example: parse the Unitree Go1, run IK

```rust
use multicalc::kinematics::{InverseKinematics, InverseKinematicsTermination, JointKind};
use multicalc::linear_algebra::Vector;
use multicalc::spatial::SE3;
use multicalc_robot_model::load_path;

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

## Example: parse the MoveIt Panda URDF

```rust
use multicalc_robot_model::{ModelFormat, load_path};

let model = load_path(std::path::Path::new(
    "third_party/moveit_resources_panda/panda.urdf",
))?;
assert_eq!(model.format(), ModelFormat::Urdf);

// Seven arm joints, two prismatic fingers; panda_joint8 and panda_hand_joint are fixed.
assert_eq!(model.body_count(), 12);
assert_eq!(model.movable_joint_count(), 9);

// The file states no <inertial> anywhere: kinematics only. Mass is never invented.
assert!(model.bodies().iter().all(|body| body.inertia().is_none()));

// panda_finger_joint2 mimics panda_finger_joint1, which a tree cannot express, so take the
// chain to the hand instead of the whole model.
assert!(model.kinematic_tree::<16, 16>().is_err());
let arm = model.kinematic_tree_to::<10, 10>("panda_hand")?;
assert_eq!(arm.len(), 10);
# Ok::<(), Box<dyn std::error::Error>>(())
```

The same `RobotModel` also emits Rust source for a build with no XML parser and no filesystem
access at runtime:

```rust
use multicalc_robot_model::{GeneratedScalar, RustSourceOptions};

let options = RustSourceOptions::new("unitree_go1")     // name of the generated fn
    .with_scalar(GeneratedScalar::F32)                    // f32 in the generated code (default; use F64 for f64)
    .with_capacity(13)                                     // KinematicTree<13, _, _>; must be >= slot count, default = exact
    .with_configuration_capacity(19);                      // the second const generic; must be >= 19 for the floating base
std::fs::write("unitree_go1.rs", model.to_rust_source(&options)?)?;
```

## The part of MJCF this reads

| Construct | What is read |
| --- | --- |
| `<compiler>` | `angle`, `autolimits`, `inertiafromgeom`, `eulerseq` |
| `<default>` | nested class blocks for `<geom>` and `<joint>`, plus `childclass` |
| `<include>` | resolved relative to the including file |
| `<worldbody>` / `<body>` | the tree, `pos`, and any one of `quat`, `euler`, `axisangle`, `xyaxes`, `zaxis` |
| `<inertial>` | `pos`, the five orientation forms, `mass`, `diaginertia`, `fullinertia` |
| `<joint>` | `hinge`, `slide`, with `axis`, `pos`, `range`, `limited`, `armature`, `damping`, `frictionloss`, `ref`, `springref`, `stiffness` |
| `<freejoint>` | top body only |
| `<geom>` | `sphere`, `ellipsoid`, `box`, `capsule`, `cylinder`, with `fromto` on the last two |

Angles are degrees unless `<compiler angle="radian">`. `childclass` on a body applies to every body
beneath it. `<include>` is followed only by `load_path`, which has a base directory to resolve
against. Geoms are integrated for mass only where the body states no `<inertial>`; `<freejoint>` on
the top body marks the model floating.

## The part of URDF this reads

| Construct | What is read |
| --- | --- |
| `<robot>` | `name` |
| `<link>` | the tree, and `<inertial>` where present |
| `<inertial>` | `<origin xyz rpy>`, `<mass value>`, `<inertia ixx ixy ixz iyy iyz izz>` |
| `<joint>` | `revolute`, `continuous`, `prismatic`, `fixed`, with `<origin>`, `<parent>`, `<child>`, `<axis>`, `<limit lower upper>`, `<dynamics damping friction>`, `<mimic>` |

Metres and radians throughout. `<origin rpy>` is fixed-axis roll-pitch-yaw,
`R = Rz(yaw)·Ry(pitch)·Rx(roll)`. The transform sits on the joint, so a body's pose is its parent
joint's `<origin>` and the root is identity; the joint itself sits at the child link frame's origin,
so `anchor` is always zero. `<axis>` defaults to `[1, 0, 0]`, against MJCF's `[0, 0, 1]`. URDF has no
armature, `ref`, `springref` or `stiffness`, so those fields stay zero.

## What is not read

Skipped only where it cannot affect mass properties; anything that could is rejected by name.
Top-level elements a reader consumed nothing from appear in [`RobotModel::ignored`](src/lib.rs),
sorted and deduplicated.

| | Skipped | Rejected |
| --- | --- | --- |
| MJCF | tendons, equality constraints, actuators, sensors, contact pairs, keyframes, assets, `<option>` | joints other than `hinge`/`slide`; a free joint off the root; several joints on one body; a mesh as a mass source; an element stating its orientation two ways at once; a limited joint with no `range`; a body with neither inertial nor mass-bearing geoms; a geom giving both `fromto` and `pos` |
| URDF | top-level elements other than `<link>`/`<joint>` (`<transmission>`, `<gazebo>`, `<material>`); `<visual>`, `<collision>` and their `package://` meshes; `<safety_controller>` | `planar` and unrecognised joint types; `floating`; `revolute`/`prismatic` with no `<limit lower upper>`; a joint naming an undeclared link; a link claimed by two joints; zero or several root links; a cycle; an unparseable attribute; an `<inertial>` the core rejects, `mass="0"` included |

`<visual>`, `<collision>` and `<safety_controller>` are link or joint children rather than top-level
elements, so they are skipped without being listed. Soft limits always sit inside the hard `<limit>`
that is read.

## Writing a model out as Rust

`RobotModel::to_rust_source(&options) -> Result<String, ModelError>` renders one `.rs` file holding
a single function that builds the parsed `KinematicTree` from literals — no XML parsing, no file
I/O, so the output runs `no_std` and filesystem-free, e.g. on a microcontroller. It reads a
`RobotModel` and never learns which reader built it, so MJCF and URDF models emit alike.
`RustSourceOptions` controls the output:

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

The vendored models under [`third_party/menagerie`](../../third_party/menagerie) and
[`third_party/moveit_resources_panda`](../../third_party/moveit_resources_panda) keep their upstream
licences; see the `LICENSE` and `README.md` alongside each.
