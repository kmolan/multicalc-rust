# multicalc-robot-model

Reads MuJoCo MJCF and URDF model files into [`multicalc`](https://crates.io/crates/multicalc)'s
robot types, and draws any of them with the bundled `model_viewer` binary.

Both formats parse into one `RobotModel`: a topologically ordered body list carrying name, parent
index, parent-relative transform, spatial inertia where stated, joint, and the shapes the file draws
the body with. Converts on request to a `KinematicTree` (`kinematic_tree` for the whole model,
`kinematic_tree_to` for the root-to-tip chain of a named body) or to an `ArticulatedBody` given
gravity.

![Six robot models drawn side by side by model_viewer: a Skydio X2 quadrotor, a TurtleBot3 Burger,
a Franka Emika Panda arm, a Google Barkour vB quadruped, a Shadow Hand and a Unitree H1 humanoid](res/demo.jpg)

## Loading a model

| Entry point | Dispatch |
| --- | --- |
| `load_path(path)` | file extension: `.urdf` reads URDF, anything else MJCF |
| `load_str(xml)` | root element: `<robot>` reads URDF, `<mujoco>` MJCF |
| `mjcf::load_path` / `mjcf::load_str` | explicit |
| `urdf::load_path` / `urdf::load_str` | explicit |

`RobotModel::format()` reports which reader ran.

## Examples

MJCF, the Menagerie Unitree Go1:

```rust
use multicalc::kinematics::JointKind;
use multicalc_robot_model::load_path;

let model = load_path(std::path::Path::new("go1.xml"))?;

// The trunk on a free joint, plus four legs of three hinges each.
assert_eq!(model.body_count(), 13);
assert!(model.has_floating_base());
assert!(model.ignored().iter().any(|section| section == "keyframe"));

// Thirteen slots; the floating base widens the configuration to nineteen.
let robot = model.kinematic_tree::<13, 19>()?;
assert_eq!(robot.joint(0).unwrap().kind(), JointKind::Floating);

// axis/damping/range come from the "abduction" default class, two levels under "go1".
let front_right_hip = robot.joint(1).unwrap();
assert_eq!(front_right_hip.damping(), 1.0);
assert_eq!(front_right_hip.limits(), Some((-0.863, 0.863)));
# Ok::<(), Box<dyn std::error::Error>>(())
```

URDF, the MoveIt Panda:

```rust
use multicalc_robot_model::{ModelFormat, load_path};

let model = load_path(std::path::Path::new("panda.urdf"))?;
assert_eq!(model.format(), ModelFormat::Urdf);

// The file states no <inertial> anywhere: kinematics only. Mass is never invented.
assert!(model.bodies().iter().all(|body| body.inertia().is_none()));

// panda_finger_joint2 mimics panda_finger_joint1, which a tree cannot express, so take the
// chain to the hand instead of the whole model.
assert!(model.kinematic_tree::<16, 16>().is_err());
assert_eq!(model.kinematic_tree_to::<10, 10>("panda_hand")?.len(), 10);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Viewing a model

```sh
cargo run --bin model_viewer -- go1.xml
```

Draws every body at the pose the file states, each joint at its own reference value. The Rerun
viewer must be on `PATH` and version-matched to the SDK:
`cargo install rerun-cli --locked --version 0.33.1`.

| Flag | Default | Effect |
| --- | --- | --- |
| `--record <file.rrd>` | unset (live) | write a recording instead of opening a viewer |
| `--geoms visual\|collision\|all` | `visual` | groups to draw: `visual` is 0–2, MuJoCo's own visible set, `collision` is 3–4 |
| `--package-path <name>=<dir>` | none | resolve `package://<name>/…`; repeatable |
| `--frame-axes <metres>` | `0.05` | frame gnomon length, `0` for none |

`RERUN_VIZ_URL` connects to that address instead of spawning a viewer. Under WSL, where the
virtualized GPU usually cannot start one, the binary connects to the default gRPC address so a
host-side viewer picks the stream up.

The same geometry sits on the model type, so a caller can draw it any other way:
`BodyDescription::visual_geometry` gives the shapes in body axes, and `RobotModel::mesh_path`
resolves a mesh reference against the model's directory and a package map.

Meshes are logged as file references and never parsed here. Rerun loads `.obj`, `.stl`, `.glb` and
`.gltf`; a `.dae` mesh, or one whose path does not resolve, is skipped with a warning while its body
still draws its frame.

## Generating Rust source

`RobotModel::to_rust_source(&options) -> Result<String, ModelError>` renders one `.rs` file holding
a single function that builds the parsed `KinematicTree` from literals: no XML parsing, no file
I/O, so the output runs `no_std` and filesystem-free, e.g. on a microcontroller. It reads a
`RobotModel` and never learns which reader built it, so MJCF and URDF models emit alike.

| Method | Default | What it sets |
| --- | --- | --- |
| `new(function_name)` | required | name of the generated function |
| `with_scalar(GeneratedScalar::F32 \| F64)` | `F32` | float type the tree is built in |
| `with_capacity(n)` | `0` = exactly as many slots as the model has | `N` in `KinematicTree<N, _>`; must be `>=` the number of bodies emitted |
| `with_configuration_capacity(n)` | `0` = `capacity`, plus six for a floating base | the second const generic, bounding the configuration vector rather than the joint count |
| `with_tip(name)` | unset = whole model | emit only the chain from the world down to body `name`, e.g. the arm without a gripper |
| `with_header(text)` | empty | a `// text` comment above the generated code |
| `with_documentation(text)` | empty | doc comment on the generated function |

```rust
use multicalc_robot_model::{GeneratedScalar, RustSourceOptions};

let options = RustSourceOptions::new("unitree_go1")
    .with_scalar(GeneratedScalar::F32)
    .with_capacity(13)
    .with_configuration_capacity(19);
std::fs::write("unitree_go1.rs", model.to_rust_source(&options)?)?;
```

## Format coverage

Unsupported constructs are rejected by name rather than dropped, so a model never loads with wrong
mass properties. Anything that cannot affect mass is skipped instead: a top-level section (listed in
`RobotModel::ignored`, sorted and deduplicated), or an undrawable geom: an unrecognised type, or
one naming an undeclared mesh.

### MJCF

| Construct | Read |
| --- | --- |
| `<compiler>` | `angle`, `autolimits`, `inertiafromgeom`, `eulerseq`, `meshdir`, `assetdir` |
| `<default>` | nested class blocks for `<geom>`, `<joint>` and `<mesh>`, plus `childclass` |
| `<include>` | resolved relative to the including file, by `load_path` only |
| `<worldbody>` / `<body>` | the tree, `pos`, and any one of `quat`, `euler`, `axisangle`, `xyaxes`, `zaxis` |
| `<inertial>` | `pos`, the five orientation forms, `mass`, `diaginertia`, `fullinertia` |
| `<joint>` | `hinge`, `slide`, with `axis`, `pos`, `range`, `limited`, `armature`, `damping`, `frictionloss`, `ref`, `springref`, `stiffness` |
| `<freejoint>` | top body only, and it marks the model floating |
| `<geom>` | `sphere`, `ellipsoid`, `box`, `capsule`, `cylinder`, with `fromto` on the last two; plus `type`, `size`, `pos`, orientation, `group`, `rgba`, `material`, `mesh` for drawing |
| `<asset>` | `<mesh>` name (or file stem), `file`, `scale`; `<material>` name and `rgba` |
| *Skipped* | tendons, equality constraints, actuators, sensors, contact pairs, keyframes, `<option>` |
| *Rejected* | joints other than `hinge`/`slide`; a free joint off the root; several joints on one body; a mesh as a mass source; two orientation forms on one element; a limited joint with no `range`; a body with neither inertial nor mass-bearing geoms; a geom giving both `fromto` and `pos` |

Angles are degrees unless `<compiler angle="radian">`, and `childclass` applies down the subtree.
Geoms are integrated for mass only where the body states no `<inertial>`. Mesh files resolve against
`meshdir`, falling back to `assetdir`; a geom's `material` resolves through the class chain, but a
`<material>` element taking its own `rgba` from a default class does not.

### URDF

| Construct | Read |
| --- | --- |
| `<robot>` | `name` |
| `<link>` | the tree, and `<inertial>` where present |
| `<inertial>` | `<origin xyz rpy>`, `<mass value>`, `<inertia ixx ixy ixz iyy iyz izz>` |
| `<joint>` | `revolute`, `continuous`, `prismatic`, `fixed`, with `<origin>`, `<parent>`, `<child>`, `<axis>`, `<limit lower upper>`, `<dynamics damping friction>`, `<mimic>` |
| `<visual>` / `<collision>` | `<origin>`, `<geometry>` (`box`, `cylinder`, `sphere`, `mesh` with `scale`), `<material>` colour |
| `<material>` | top-level `name` and `<color rgba>`, for visuals naming one |
| *Skipped* | top-level elements other than `<link>`/`<joint>`/`<material>`, e.g. `<transmission>`, `<gazebo>`; `<safety_controller>`, a joint child and so never listed, whose soft limits always sit inside the hard `<limit>` |
| *Rejected* | `planar`, `floating` and unrecognised joint types; `revolute`/`prismatic` with no `<limit lower upper>`; a joint naming an undeclared link; a link claimed by two joints; zero or several root links; a cycle; an unparseable attribute; an `<inertial>` the core rejects, `mass="0"` included |

Metres and radians. `<origin rpy>` is fixed-axis roll-pitch-yaw, `R = Rz(yaw)·Ry(pitch)·Rx(roll)`.
The transform sits on the joint, so a body's pose is its parent joint's `<origin>`, the root is
identity, and `anchor` is always zero. `<axis>` defaults to `[1, 0, 0]`, against MJCF's `[0, 0, 1]`;
armature, `ref`, `springref` and `stiffness` have no URDF equivalent and stay zero. `<visual>` shapes
are group 0 and `<collision>` group 3, MJCF's convention, so one group filter serves both formats;
a `<mesh filename>` is kept verbatim, `package://` and all.

## Licence

Model files used by the examples and tests are third-party and keep their upstream licences; see the
`LICENSE` alongside each.
