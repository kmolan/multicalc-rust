# hilshare

`multicalc` as three [Tock](https://tockos.org) apps on the three [HILShare](https://github.com/hilshare/action)
boards: the Nucleo-F411RE (Cortex-M4F at 100 MHz), the Raspberry Pi Pico 2 (Cortex-M33 at
150 MHz) and the XIAO ESP32-C3 (RV32IMC at 160 MHz). The Nucleo runs HILShare kernel `0.3.1` and
the Pico 2 kernel `0.3.0`, which give the FPU to the process, and the XIAO runs kernel `0.2.0`.
`embedded-smoke` runs the same math under QEMU; these apps run it on silicon and time it.

A dev-only crate (`publish = false`) in its own workspace. It links
[libtock-rs](https://github.com/tock/libtock-rs) at the fixed app addresses of each kernel, on a
pinned toolchain (`rust-toolchain.toml`). Build it from this directory, not from the repository
root.

## What it prints

Three apps. Each is one process, and each starts with the two counter checks. Each check runs
eight times: the first run gives the cold figure, and the best and the worst of the other seven
give the warm figures.

```
HILSHARE_START_EXECUTION
HILSHARE_FIRMWARE=0.5.0+arm
HILSHARE_CHECK name=vector_dot status=PASS us=<us> worst_us=<us> budget_us=<budget> overhead_us=<round trip> cycles=<warm> worst_cycles=<worst> cold_cycles=<cold> value=2e1
HILSHARE_CHECKS_PASSED=12/12
HILSHARE_RESULT=PASS
```

| Key | Value |
|---|---|
| `us` | The warm figure in microseconds, less `overhead_us`. The budget judges this figure |
| `worst_us` | The worst of the seven warm figures in microseconds, less `overhead_us` |
| `budget_us` | The budget of the check on this board, from `CHECKS` in `src/checks/mod.rs` |
| `overhead_us` | The round trip of the system call that the app subtracted: the warm figure of `cycle_counter_overhead` in this run. `0` on the two counter checks |
| `cycles` | The warm figure in cycles, less the round trip |
| `worst_cycles` | The worst warm figure in cycles, less the round trip |
| `cold_cycles` | The first reading in cycles, less the round trip |
| `value` | The headline scalar of the check |

| App (Cargo feature) | Checks |
|---|---|
| `arm` | The math checks; Franka Panda RNEA, CRBA and ABA against Pinocchio; the `f32` computed-torque tick with the momentum observer; the `Dual` derivative `∂τ/∂q̈ = H` |
| `drone` | The `f32` Skydio X2 tick at hover and its attitude-and-mixer part; the ESKF over the QA IMU trajectory |
| `mobile_and_policy` | The `f32` coordinated-turn EKF and pure-pursuit tick; the MLP policy in `f64` and `f32` against numpy |

elf2tab pads a Cortex-M TBF to a power of two, and the Nucleo-F411RE app region is 256 KB, so one
app cannot hold every check. Each app has a 48 KB stack: the `f64` and `Dual` dynamics frames are
up to about 20 KB.

The app reads the CPU clock from the clock driver of the kernel, `0xA0000`. If the driver is not
available, the app prints `HILSHARE_CHECK name=clock_rate status=FAIL` and
`HILSHARE_RESULT=FAIL`.

A check is `PASS` when its eight answers agreed with the known answer and `us <= budget_us`.
`HILSHARE_RESULT` is `PASS` when all the checks of the app pass. The goldens come from
`tools/embedded-smoke/src/fixtures.rs` and `src/robotics_fixtures.rs`, which `multicalc-qa`
generates. Do not write a golden by
hand.

## Building

```sh
cargo install elf2tab --version 0.13.0 --locked
tools/hilshare/build-tab.sh
```

Build through the script. It sets `RUSTFLAGS`, which replaces the `embedded-smoke` linker flags in
the `.cargo/config.toml` of the repository root. It builds each app one time for each board, at
the addresses of that kernel. It packs the three TBFs of each app in
`tools/hilshare/target/tab/multicalc-hilshare-<app>.tab`, with the ELF and the TBF of each board
beside it. The Arm builds are hard-float.

## In CI, and on the board

`.github/workflows/hilshare.yml` lints this crate and builds the TABs on each pull request, so a
change to an API that a check calls must keep this crate building.

CI does not run the app on a board. A maintainer starts the `Board` workflow by hand
(`.github/workflows/board.yml`). It builds the three TABs and runs each on the three boards through
`hilshare/action@v1`. A red run blocks nothing; a maintainer investigates it.

## Budgets

A budget is `ceil(1.1 × us)` of the figure measured on its board, in `CHECKS` in
`src/checks/mod.rs`. On the Nucleo-F411RE and the XIAO ESP32-C3, the figures repeat from run to run
of one build, so the 10% is margin for the code layout, not for noise.

On the Raspberry Pi Pico 2, a budget is `ceil(1.5 × us)` of the highest figure of the runs. The
figures of the short checks change by up to 26% from run to run of one build.

The figures of each check on each board are in
[`benchmarks/silicon.md`](../../benchmarks/silicon.md), from the runs that set the budgets.

Measure each budget again when one of these changes:

- the channel in `rust-toolchain.toml`
- the version of the HILShare kernel
- `Cargo.lock`, or the libtock-rs rev in `Cargo.toml`
- the release profile in `Cargo.toml`, `tools/hilshare/.cargo/config.toml`, or the addresses in
  `build-tab.sh`
- the checks of an app, which move the code layout of that app

Measure the budget of one check again when a change to multicalc moves the figure of that check
above its budget. The `Board` run is then red. Make the regression smaller, or raise that one
number with a line of reason and the figure of the board beside it, as `ci/README.md` does for a
size budget. Do not make the 1.1 or the 1.5 larger.

To measure: build the TABs with each budget at `u32::MAX`, start three `Board` runs, and read `us`
from each `HILSHARE_CHECK` line. Set each budget from the highest figure, write
`benchmarks/silicon.md` with `gen_silicon_table` from the same runs, and rebuild. A run of the new
TABs on each board must then give `HILSHARE_RESULT=PASS`.
