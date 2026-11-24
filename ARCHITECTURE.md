# Architecture

## Crates

| Crate | Path | Role | On crates.io |
| --- | --- | --- | --- |
| `multicalc` | [`crates/multicalc`](crates/multicalc) | The library: `no_std`, no heap by default, one dependency (`libm`) | Yes |
| `multicalc-robot-model` | [`crates/multicalc-robot-model`](crates/multicalc-robot-model) | Reads MJCF and URDF into `multicalc` types, and writes a model as Rust source. Needs `std`. The `viewer` feature adds the Rerun model viewer. | Yes, at the version of `multicalc` |
| `multicalc-py` | [`crates/multicalc-py`](crates/multicalc-py) | Python bindings | No |
| `multicalc-demos` | [`demos`](demos) | Examples and live showcases | No |
| `multicalc-qa` | [`tools/qa`](tools/qa) | Tests against reference fixtures, the fixture and table generators, and the benchmarks | No |
| `multicalc-testkit` | [`tools/testkit`](tools/testkit) | Test helpers for `multicalc` | No |
| `embedded-smoke` | [`tools/embedded-smoke`](tools/embedded-smoke) | Bare-metal tests under QEMU | No |
| `multicalc-hilshare` | [`tools/hilshare`](tools/hilshare) | Tock apps for the HILShare boards, in a separate workspace | No |

`multicalc` depends on no crate of this workspace. A microcontroller does not run
`multicalc-robot-model`: `RobotModel::to_rust_source` writes the model as one Rust function on the
host, and the firmware compiles that file. An example is
[`tools/embedded-smoke/src/franka_panda_model.rs`](tools/embedded-smoke/src/franka_panda_model.rs).

## Domains

Each domain is one directory under [`crates/multicalc/src`](crates/multicalc/src). Core is a set
of modules, not a module, so each path stays flat (`multicalc::linear_algebra`).

| Domain | Modules | Contents |
| --- | --- | --- |
| core | `scalar`, `linear_algebra`, `numerical_derivative`, `numerical_integration`, `gaussian_tables`, `optimization`, `root_finding`, `ode`, `discretization`, `signal_processing`, `spatial`, `polynomial`, `random`, `mlp_inference`, `approximation`, `vector_field`, `error` | Scalars and autodiff, linear algebra, calculus, solvers, ODE integrators, filters, Lie groups and spatial algebra, polynomials, random numbers, MLP inference |
| kinematics | `kinematics` | Kinematic trees, forward and inverse kinematics, Jacobians, collision queries, wheeled-robot maps, odometry |
| dynamics | `dynamics` | Rigid bodies, RNEA, CRBA, ABA, the momentum observer |
| estimation | `estimation` | KF, EKF, UKF, ESKF, attitude filters, the particle filter, Monte Carlo localization |
| control | `control` | PID, LQR, geometric attitude control, computed torque, impedance control, pure pursuit, follow-the-gap |
| motion | `motion` | Polyline paths, motion profiles, minimum-snap trajectories |
| plant | `plant` | Rotor lag, the position servo, the multirotor mixer |
| mapping | `mapping` | Occupancy grids, log-odds grids, distance fields, scan geometry, cost grids |
| planning | `planning` | Grid search (Dijkstra, A\*, Theta\*), RRT, RRT\*, PRM, frontiers |

`approximation` and `vector_field` are frozen.

## Layer graph

The edges are the `use crate::` paths in code. Doc links do not count.

```
core
 ├─ mapping     → core
 ├─ motion      → core
 ├─ plant       → core
 ├─ kinematics  → core
 ├─ estimation  → core, mapping
 ├─ dynamics    → core, kinematics
 ├─ control     → core, kinematics, dynamics, mapping
 └─ planning    → core, mapping, motion
```

- `estimation → mapping`: `MonteCarloLocalizer` matches a scan against an `OccupancyMap`.
- `dynamics → kinematics`: RNEA, CRBA and ABA recurse over the `KinematicTree`.
- `control → kinematics, dynamics`: computed torque and impedance control use `ArticulatedBody`.
- `control → mapping`: `FollowTheGap` numbers its beams with `mapping::beam_angle_across`.
- `planning → mapping, motion`: the planners search an `OccupancyMap` and return a
  `PolylinePath`.

No domain uses a domain below it in this list, and core uses no domain. Inside core, the one cycle
is `numerical_integration` ↔ `gaussian_tables`. To list the edges of one module:

```sh
grep -rhE "crate::[a-z_]+" crates/multicalc/src/control | grep -vE "^\s*//" \
  | grep -oE "crate::[a-z_]+" | sort -u
```

## Rules

- **New code goes to one domain.** Pose and velocity maps go to `kinematics`; forces, mass and
  inertia to `dynamics`; actuation and allocation to `plant`; feedback to `control`; state
  inference to `estimation`; references and trajectories to `motion`; maps and what a sensor
  measures against them to `mapping`; search over a map or a configuration space to `planning`;
  domain-independent math to core.
- **`error.rs` defines each domain error enum** (`KinematicsError`, `DynamicsError`, and the
  others), and the crate root re-exports it (`multicalc::KinematicsError`). Thus `CalcError`
  names each enum, and core imports no domain.
- **One Cargo feature, `alloc`.** No feature removes a domain: a clean build takes about 2–3 s,
  the generic code compiles in the user's crate, and the linker removes the code that a binary
  does not call.
- **A domain becomes its own crate only when all of these are true:** each of its modules is at
  the industry bar; its API had no break in three or more minors; its public types come only from
  core and the domains it uses; and users ask for it. Then `multicalc` re-exports it, and each
  path stays the same.
