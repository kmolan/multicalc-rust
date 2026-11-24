# multicalc-robot-model

Reads MuJoCo MJCF and URDF model files into the robot types of [`multicalc`](https://crates.io/crates/multicalc). The `model_viewer` binary draws a model file.

`multicalc` is robotics for microcontrollers: sense, plan, estimate and control in `no_std` Rust.
This crate needs `std` and runs on the host.

![model_viewer drawing a Skydio X2, a TurtleBot3 Burger, a Franka Emika Panda, a Unitree Go2, a
Shadow Hand and a Unitree H1](https://raw.githubusercontent.com/kmolan/multicalc-rust/main/crates/multicalc-robot-model/res/demo.jpg)

## Install

```sh
cargo add multicalc multicalc-robot-model
```

The two crates release together, at the same version.

## Loading a model

| Entry point | Dispatch |
| --- | --- |
| `load_path(path)` | file extension: `.urdf` reads URDF, anything else MJCF |
| `load_str(xml)` | root element: `<robot>` reads URDF, `<mujoco>` MJCF |
| `mjcf::load_path` / `mjcf::load_str` | explicit |
| `urdf::load_path` / `urdf::load_str` | explicit |

`RobotModel::format()` reports which reader ran.

## MuJoCo Menagerie

This crate loads 40 of the 71 robots in
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (commit `eea4c74`). CI
checks three of them against MuJoCo. RNEA, CRBA and ABA need a fixed base.

<details>
<summary>Each Menagerie robot</summary>

| Robot | Status | Base |
| --- | --- | --- |
| `agilex_piper` | Refused: a geom `size` the reader cannot parse | — |
| `agility_cassie` | Refused: a ball joint | — |
| `aloha` | Loads | fixed |
| `anybotics_anymal_b` | Loads | floating |
| `anybotics_anymal_c` | Loads | floating |
| `apptronik_apollo` | Refused: a body with no `<inertial>` and no geom | — |
| `arx_l5` | Refused: mass from a mesh geom | — |
| `berkeley_humanoid` | Loads | floating |
| `bitcraze_crazyflie_2` | Loads | floating |
| `booster_t1` | Loads | floating |
| `boston_dynamics_spot` | Loads | floating |
| `dynamixel_2r` | Loads | fixed |
| `fanuc_crx5ia` | Loads | fixed |
| `flexiv_rizon4` | Refused: mass from a mesh geom | — |
| `flexiv_rizon4s` | Refused: mass from a mesh geom | — |
| `flybody` | Refused: mass from a mesh geom | — |
| `fourier_n1` | Loads | floating |
| `franka_emika_panda` | Checked against MuJoCo in CI, except `panda_nohand.xml` (a body with no `<inertial>` and no geom) | fixed |
| `franka_fr3` | Refused: a body with no `<inertial>` and no geom | — |
| `franka_fr3_duo` | Refused: a body with no `<inertial>` and no geom | — |
| `franka_fr3_v2` | Refused: a body with no `<inertial>` and no geom | — |
| `google_barkour_v0` | Loads | floating |
| `google_barkour_vb` | Refused: a body with no `<inertial>` and no geom | — |
| `google_robot` | Refused: mass from a mesh geom | — |
| `hello_robot_stretch` | Refused: mass from a mesh geom | — |
| `hello_robot_stretch_3` | Refused: mass from a mesh geom | — |
| `i2rt_yam` | Refused: mass from a mesh geom | — |
| `iit_softfoot` | Refused: several joints on one body | — |
| `kinova_gen3` | Loads | fixed |
| `kuka_iiwa_14` | Loads | fixed |
| `leap_hand` | Loads | fixed |
| `low_cost_robot_arm` | Refused: mass from a mesh geom | — |
| `ms_human_700` | Refused: several joints on one body | — |
| `pal_talos` | Loads | floating |
| `pal_tiago` | Loads, except `tiago_position.xml` (a geom `size` the reader cannot parse) | floating |
| `pal_tiago_dual` | Refused: a geom `size` the reader cannot parse | — |
| `pndbotics_adam_lite` | Loads | floating |
| `rainbow_robotics_rby1` | Refused: a geom `size` the reader cannot parse | — |
| `realsense_d435i` | Refused: mass from a mesh geom | — |
| `rethink_robotics_sawyer` | Loads | fixed |
| `robot_soccer_kit` | Loads | floating |
| `robotiq_2f85` | Refused: mass from a mesh geom | — |
| `robotiq_2f85_v4` | Loads, except `mjx_2f85.xml` (a geom `size` the reader cannot parse) | fixed |
| `robotis_omy_3m` | Loads | fixed |
| `robotis_op3` | Loads | floating |
| `robotstudio_so101` | Refused: mass from a mesh geom | — |
| `seeed_rebot_devarm` | Loads | fixed |
| `shadow_dexee` | Refused: mass from a mesh geom | — |
| `shadow_hand` | Loads | fixed |
| `sharpa_wave` | Refused: mass from a mesh geom | — |
| `skydio_x2` | Checked against MuJoCo in CI | floating |
| `stanford_tidybot` | Refused: several joints on one body | — |
| `tetheria_aero_hand_open` | Refused: mass from a mesh geom | — |
| `toddlerbot_2xc` | Loads | floating |
| `toddlerbot_2xm` | Loads | floating |
| `trossen_vx300s` | Loads | fixed |
| `trossen_wx250s` | Loads | fixed |
| `trossen_wxai` | Loads | fixed |
| `trs_so_arm100` | Refused: mass from a mesh geom | — |
| `ufactory_lite6` | Refused: a non-positive inertia | — |
| `ufactory_xarm7` | Loads | fixed |
| `umi_gripper` | Refused: several joints on one body | — |
| `unitree_a1` | Loads | floating |
| `unitree_g1` | Loads | floating |
| `unitree_go1` | Checked against MuJoCo in CI | floating |
| `unitree_go2` | Loads, except `go2_mjx.xml` (a geom `size` the reader cannot parse) | floating |
| `unitree_h1` | Loads | floating |
| `unitree_z1` | Loads | fixed |
| `universal_robots_ur10e` | Loads | fixed |
| `universal_robots_ur5e` | Loads | fixed |
| `wonik_allegro` | Refused: mass from a mesh geom | — |

</details>

## Your own model, at runtime

The const generics of `kinematic_tree::<MAX_JOINTS, MAX_CONFIG>()` are a capacity, not a model
size. One binary with `kinematic_tree::<32, 38>()` loads each model of 32 bodies or fewer, with
no recompile. A larger model gives `ModelError::TreeCapacityExceeded`. A floating base needs
six more configuration slots than bodies.

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
cargo run -p multicalc-robot-model --features viewer --bin model_viewer -- go1.xml
```

Draws each body at the pose that the file states, with each joint at its reference value. Needs a
version-matched Rerun viewer on `PATH`: `cargo install rerun-cli --locked --version 0.33.1`.

| Flag | Default | Effect |
| --- | --- | --- |
| `--record <file.rrd>` | live | a recording instead of a viewer |
| `--geoms visual\|collision\|all` | `visual` | groups drawn: `visual` is 0–2, MuJoCo's visible set; `collision` is 3–4 |
| `--package-path <name>=<dir>` | none | resolves `package://<name>/…`; repeatable |
| `--frame-axes <metres>` | `0.05` | axis length, `0` for no frames |

The binary spawns a viewer, with two exceptions:

- If `RERUN_VIZ_URL` names a viewer, it connects to that viewer.
- On WSL, the virtualized GPU rarely starts a viewer. The default gRPC address reaches a viewer on
  the Windows side.

For other renderers, the geometry is on the model type: `BodyDescription::visual_geometry` in body
axes, and `RobotModel::mesh_path` against the model directory and a package map.

The viewer logs meshes as file references and never parses them. Rerun decodes `.obj`, `.stl`,
`.glb`, `.gltf` and `.dae` (`.dae` as triangles and diffuse colours, without textures). If a path
does not resolve, the viewer skips it with a warning, and the body still draws its frame. If Rerun
cannot decode a file, it fails there, and the binary does not report it.

## Generating Rust source

`RobotModel::to_rust_source(&options) -> Result<String, ModelError>` renders one `.rs` file: a
single function that builds the parsed `KinematicTree` from literals. It has no XML parsing and no
file I/O, so it runs `no_std` and with no filesystem. It reads only the `RobotModel`, so MJCF and
URDF give the same output form.

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

## Licence

The model files of the examples and tests are third-party, under their upstream licences.
