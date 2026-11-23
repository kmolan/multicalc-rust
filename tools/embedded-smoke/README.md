# embedded-smoke

A `no_std` / `no_main` smoke binary that runs `multicalc` on four bare-metal ABIs under QEMU:
Cortex-M4 soft-float, Cortex-M4 hard-float, Cortex-M0 and `riscv32imc`. Host tests cannot cover
these targets. Each result must agree with a known answer to a tolerance, not bit for bit, because
soft-float paths round intermediates differently:

- on target, the abs/rel band of the fixture;
- between ABIs, `1e-9` relative (see [ci/README.md](../../ci/README.md#cross-abi-divergence-guard)).

A divergence beyond tolerance on any leg fails the build.

The crate is dev-only (`publish = false`, not in `default-members`), and it never builds for a host
target. It is a separate crate because it is a full bare-metal program: start-up code, a crash
handler, a custom link step, and chip-only dependencies (`cortex-m-rt`, `riscv-rt`). In
`multicalc`, these would break the host build and add chip-only dependencies to a crate that ships
to crates.io.

## Running

```sh
rustup target add thumbv7em-none-eabi thumbv7em-none-eabihf thumbv6m-none-eabi riscv32imc-unknown-none-elf
sudo apt-get install -y qemu-system-arm qemu-system-misc  # qemu-system-misc provides qemu-system-riscv32
cargo install cargo-binutils                              # provides cargo size, for the gate

cargo run -p embedded-smoke --release --target thumbv7em-none-eabi
cargo run -p embedded-smoke --release --target thumbv7em-none-eabihf
cargo run -p embedded-smoke --release --target thumbv6m-none-eabi --no-default-features
cargo run -p embedded-smoke --release --target riscv32imc-unknown-none-elf
```

Aliases: `cargo smoke-eabi`, `cargo smoke-eabihf`, `cargo smoke-m0`, `cargo smoke-riscv`.

## Targets and QEMU machine

| Target                        | Codegen                | QEMU machine          | RAM  | Flash / DRAM base |
|-------------------------------|------------------------|-----------------------|------|-------------------|
| `thumbv7em-none-eabi`         | Cortex-M4, soft-float  | `netduinoplus2`       | 64K  | `0x08000000`      |
| `thumbv7em-none-eabihf`       | Cortex-M4, hard-float  | `netduinoplus2`       | 64K  | `0x08000000`      |
| `thumbv6m-none-eabi`          | Cortex-M0, CAS-free    | `microbit` (nRF51)    | 16K  | `0x0`             |
| `riscv32imc-unknown-none-elf` | RV32IMC, soft-float    | `virt` (`-bios none`) | 256K | `0x80000000`      |

`microbit` is a real M0 core. Thus an oversized image or an out-of-ISA (ARMv7E-M) instruction
faults, as it does on silicon.

The runners and `rustflags` (`-Tlink.x`, `--nmagic`) are in `.cargo/config.toml` at the repository
root. `build.rs` supplies the memory map of each target from `memory/`.

## Checks

`src/checks.rs` holds the on-target checks against the public API. Each check asserts a known answer
to a tolerance, so a wrong result panics. All checks are `no_std`.

- A **golden** check asserts against a value from the host QA fixtures (`src/fixtures.rs`, generated
  from `tools/qa`).
- An **identity** check asserts a mathematical identity and needs no fixture.

The `full-smoke` feature (on by default) selects the set. The canary runs on all targets, and
`thumbv6m` builds with `--no-default-features` to run only the canary. The full set adds the
heavier checks on the `thumbv7em` ABIs and `riscv32imc`.

| Test | Targets | Details |
|------|---------|---------|
| `portable_path` | all (incl. `thumbv6m`) | Identity: `multicalc` vector dot product `[1,2,3,4]·[4,3,2,1] = 20`, a no-atomics library call on M0. |
| `svd_golden` | all (incl. `thumbv6m`) | Golden: singular values of a 3×3 fixture matrix vs the `svd_3x3` QA golden. Emits `SMOKE_VAL_svd_s*`. |
| `error_path_returns_err` | all (incl. `thumbv6m`) | Negative path: `lu()` of a singular matrix and `cholesky()` of an indefinite matrix return a typed `Err`, not a panic. |
| `pid_step` | all (incl. `thumbv6m`) | Identity: the first PID output is exactly `kp·e + ki·e·dt`, and the plant climbs to within 1% of the setpoint. Emits `SMOKE_VAL_pid`. |
| `polynomial_evaluate` | all (incl. `thumbv6m`) | Identity: value and first two derivatives of a degree-7 polynomial at `x = 1/2`, where every power is exact. Emits `SMOKE_VAL_poly`. |
| `polynomial_quadratic_roots` | all (incl. `thumbv6m`) | Identity: the closed-form roots of `(x−1)(x−2)` are 1 and 2. Emits `SMOKE_VAL_poly_quadratic_root`. |
| `lm_fit` | full set | Golden: Levenberg-Marquardt on the Rosenbrock least-squares problem vs the `rosenbrock` QA golden. Emits `SMOKE_VAL_lm0`. |
| `autodiff_derivative` | full set | Identity: forward-mode autodiff of `x³` at `x = 2` is 12. Emits `SMOKE_VAL_ad`. |
| `autodiff_derivative_f32` | full set | Identity in `f32`: the same derivative, where soft-float and the FPU are most likely to differ. Emits `SMOKE_VAL_ad_f32`. |
| `ode_identity` | full set | Identity: RK4 harmonic-oscillator round trip, and RK45 `y' = -y` to `e^{-1}`. |
| `lie_group_identity` | full set | Identity: SO(3)/SE(3) exp/log round trips and one known rotation. Emits `SMOKE_VAL_lie`. |
| `quadrature_identity` | full set | Identity: Gauss-Legendre order 4 of `2x` on `[0,2]` is 4. Emits `SMOKE_VAL_quad`. |
| `jacobian_identity` | full set | Identity: the Jacobian of `[x·y·z, x²+y²]` at `(1,2,3)` is `[[6,3,2],[2,4,0]]`. Emits `SMOKE_VAL_jac00`. |
| `vector_field_identity` | full set | Identity: curl of `[y,-x,2z]` is `[0,0,-2]`, and divergence is 2. Emits `SMOKE_VAL_div3d`. |
| `root_finding_golden` | full set | Golden: Newton on Wien's equation vs the `wien_newton` QA golden. Emits `SMOKE_VAL_wien_root`. |
| `kalman_filter_golden` | full set | Golden: eight predict/update steps of a constant-velocity Kalman filter vs the `kalman_filter_constant_velocity_one_dimensional` QA golden. Emits `SMOKE_VAL_kalman`. |
| `extended_kalman_filter_golden` | full set | Golden: eight EKF steps of a coordinated turn with position fixes vs the `extended_kalman_filter_coordinated_turn_fusion` QA golden. The only check that drives 5×5 autodiff Jacobians on target. Emits `SMOKE_VAL_extended_kalman`. |
| `kalman_filter_identity_f32` | full set | Identity in `f32`: two unit-measurement steps of a constant-velocity filter give exactly `[5/3, 2/3]`. Emits `SMOKE_VAL_kalman_f32`. |
| `pure_pursuit_identity` | full set | Identity: curvature is zero for a point dead ahead, and exactly `2·lateral/L²` for a point to the side. Emits `SMOKE_VAL_pure_pursuit`. |
| `follow_the_gap_identity` | full set | Identity: a clear 31-beam scan drives straight ahead at cruise speed; a wall on all sides stops the robot and reports it blocked. Emits `SMOKE_VAL_follow_the_gap`. |
| `lqr_identity` | full set | Identity: an LQR for `A = [[1, 0.1], [0, 1]]`, `B = [0.005, 0.1]` passes `certify_stability` and drives `[1, 0]` to `‖x‖ < 1e-6` in 400 steps. Emits `SMOKE_VAL_lqr`. |
| `geometric_attitude_identity` | full set | Identity: zero torque at the target attitude with zero rate; a tilt about x gives a negative x torque. Emits `SMOKE_VAL_geometric_attitude`. |
| `polynomial_roots_golden` | full set | Golden: the four real roots of `x(x−1)(x−2)(x−4)` vs the `roots_quartic_four_real` QA golden. Emits `SMOKE_VAL_poly_root0`. |
| `polynomial_sturm_identity` | full set | Identity: the six roots of a sixth-degree polynomial by Sturm sign-change counts and bisection, the deepest polynomial path on target. Emits `SMOKE_VAL_poly_sturm0`. |
| `piecewise_polynomial_golden` | full set | Golden: position, velocity and acceleration of a minimum-snap trajectory vs the `minimum_snap_three_segments_3d` QA golden. Evaluation is the hot-loop path; planning stays on the host. Emits `SMOKE_VAL_piecewise`. |
| `multivariate_identity` | full set | Identity: `3x²y + 2xy − 1` and its gradient at `(1.5, −2)`, with integer powers. Emits `SMOKE_VAL_multivariate`. |
| `polynomial_evaluate_identity_f32` | full set | Identity in `f32`: the degree-7 evaluation at single precision. Emits `SMOKE_VAL_poly_f32`. |
| `occupancy_ray_cast_identity` | full set | Identity: on a 16×16 map of 0.25 m cells, a beam from x = 0.5 meets a wall at x = 1.25 at exactly 0.75 m; the reverse beam meets nothing; the middle beam of a scan reads the same range. Covers bit-packed storage, rasterization and the cell walk. Emits `SMOKE_VAL_occupancy_ray_cast`. |
| `kinematic_tree_identity` | full set | Identity: a two-joint planar arm of unit links puts its tool at `[cos a + cos(a+b), sin a + sin(a+b), 0]`. Emits `SMOKE_VAL_forward_kinematics`. |
| `geometric_jacobian_identity` | full set | Identity: the same arm at `q = 0`, world frame: shoulder linear `v_y = 2`, elbow `v_y = 1`, shoulder `ω_z = 1`, fixed-joint column zero. Emits `SMOKE_VAL_geometric_jacobian`. |
| `franka_forward_kinematics_f32` | full set | Golden in `f32`: the Franka Panda tree from the generated `franka_panda_model.rs` puts its hand at the MuJoCo position (`kinematics/franka_panda_seven_joint`, configuration 0) to `2e-4`. Also checks the armature and damping inherited from the MJCF default class. Emits `SMOKE_VAL_franka_hand_x`. |

## Pass/fail contract

The binary runs under QEMU semihosting:

- Clean finish: `debug::exit(EXIT_SUCCESS)`, and QEMU exits 0.
- Panic (failed `assert!`/`expect`): on ARM, `panic-semihosting` (with the `exit` feature) prints
  the message and exits QEMU non-zero. On RISC-V, the `#[panic_handler]` of the crate does the same
  through `riscv-semihosting`.

> If you pipe `cargo run … | tee`, set `pipefail`, or `tee` hides the exit code. CI runs steps
> under `bash -eo pipefail`, so a panic fails the job.

## Stack high-water mark

`main.rs` paints the free stack below the entry frame with a sentinel. After the checks, it scans
for the deepest overwritten byte and prints `STACK_HWM_BYTES=<n>` for the budget gate.

- `PAINT`: the sentinel byte (`0xAA`).
- `WINDOW`: the bytes of free stack to paint and scan (6 KiB). It must exceed the deepest check and
  fit the target RAM. If the measured value reaches `WINDOW`, the binary panics. Then raise
  `WINDOW` and confirm the headroom.
- `GUARD`: the bytes just below the current SP that the paint does not touch (the live frame).

The scan uses a fixed window below the SP, so it needs no linker symbol and is identical on all
machines. If a target cannot give a stable number, remove its `STACK_HWM_BYTES` line and gate that
ABI on `.text` only.

## Size and stack gate

`ci/check_budgets.sh` reads `.text` through `cargo size -- -A` and parses `STACK_HWM_BYTES` from the
run output. It compares both against `ci/budgets.toml`. For the formula and the raise protocol, see
[ci/README.md](../../ci/README.md).

```sh
cargo run -p embedded-smoke --release --target thumbv7em-none-eabi | tee smoke.txt
bash ci/check_budgets.sh embedded-smoke thumbv7em-none-eabi smoke.txt
```

## Lints

This crate does **not** use `[lints] workspace = true`. It needs `unsafe` for the stack scan and the
exception handlers, so it sets `unsafe_code = "allow"` locally. It does not inherit the panic-free
and no-unsafe lints of the library.
