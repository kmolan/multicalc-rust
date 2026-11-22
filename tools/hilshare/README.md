# hilshare

`multicalc` as a [Tock](https://tockos.org) app on the three [HILShare](https://github.com/hilshare/action)
boards: the Nucleo-F411RE (Cortex-M4F at 100 MHz), the Raspberry Pi Pico 2 (Cortex-M33 at
150 MHz) and the XIAO ESP32-C3 (RV32IMC at 160 MHz), each on HILShare kernel `0.2.0`.
`embedded-smoke` runs the same math under QEMU; this app runs it on silicon and times it.

A dev-only crate (`publish = false`) in its own workspace. It links
[libtock-rs](https://github.com/tock/libtock-rs) at the fixed app addresses of each kernel on a
pinned toolchain (`rust-toolchain.toml`), so build it from inside this directory, not from the
repository root.

## What it prints

One process, seven checks. Each check runs eight times: the first run gives the cold figure, and
the best of the other seven gives the warm figure.

```
HILSHARE_START_EXECUTION
HILSHARE_FIRMWARE=0.4.0
HILSHARE_CHECK name=vector_dot status=PASS us=<us> budget_us=<budget> overhead_us=<round trip> cycles=<warm> cold_cycles=<cold> value=2e1
HILSHARE_CHECKS_PASSED=7/7
HILSHARE_RESULT=PASS
```

| Key | Value |
|---|---|
| `us` | The warm figure in microseconds, less `overhead_us`. The budget judges this figure |
| `budget_us` | The budget of the check on this board, from `CHECKS` in `src/checks.rs` |
| `overhead_us` | The round trip of the system call that the app subtracted: the warm figure of `cycle_counter_overhead` in this run. `0` on the two counter checks |
| `cycles` | The warm figure in cycles, less the round trip |
| `cold_cycles` | The first reading in cycles, less the round trip |
| `value` | The headline scalar of the check |

The app reads the CPU clock from the clock driver of the kernel, `0xA0000`. If the driver is not
available, the app prints `HILSHARE_CHECK name=clock_rate status=FAIL` and
`HILSHARE_RESULT=FAIL`.

A check is `PASS` when its eight answers agreed with the known answer and `us <= budget_us`.
`HILSHARE_RESULT` is `PASS` when the seven checks are. The goldens come from
`tools/embedded-smoke/src/fixtures.rs`, which `multicalc-qa` generates. Do not write a golden by
hand.

## Building

```sh
cargo install elf2tab --version 0.13.0 --locked
tools/hilshare/build-tab.sh
```

Build through the script. It sets `RUSTFLAGS`, which replaces the linker flags that the
`.cargo/config.toml` of the repository root gives for `embedded-smoke`. It builds the app one time
for each board, at the addresses of that kernel, and packs the three TBFs in
`tools/hilshare/target/tab/multicalc-hilshare.tab`, with the ELF of each board beside it.

## In CI, and on the board

`.github/workflows/hilshare.yml` lints this crate and builds the TAB on each pull request, so a
change to an API that a check calls must keep this crate building.

CI does not run the app on a board. A maintainer starts the `Board` workflow by hand
(`.github/workflows/board.yml`). It builds the TAB and runs it on the three boards through
`hilshare/action@v1`. A red run is investigated and blocks nothing.

## Budgets

A budget is `ceil(1.1 × us)` of the figure measured on its board, in `CHECKS` in
`src/checks.rs`. The figures repeat from run to run of one build, so the 10% is margin for the
code layout, not for noise.

Measured on toolchain `1.92` and HILShare kernel `0.2.0`, `us` and then `budget_us` on each board:

| Check | Nucleo-F411RE | Raspberry Pi Pico 2 | XIAO ESP32-C3 |
|---|---|---|---|
| `cycle_counter_sanity` | — | — | — |
| `cycle_counter_overhead` | — | — | — |
| `vector_dot` | — | — | — |
| `svd_singular_values` | — | — | — |
| `autodiff_derivative` | — | — | — |
| `pid_step` | — | — | — |
| `error_path_returns_err` | — | — | — |

Measure each budget again when one of these changes:

- the channel in `rust-toolchain.toml`
- the version of the HILShare kernel
- `Cargo.lock`, or the libtock-rs rev in `Cargo.toml`
- the release profile in `Cargo.toml`, `.cargo/config.toml`, or the addresses in `build-tab.sh`

Measure the budget of one check again when a change to multicalc moves the figure of that check
above its budget. The `Board` run is then red. Make the regression smaller, or raise that one
number with a line of reason and the figure of the board beside it, as `ci/README.md` does for a
size budget. Do not make the 1.1 larger.

To measure: build the TAB with each budget at `u32::MAX`, run it three times on each board, and
read `us` from each `HILSHARE_CHECK` line. Set each budget to `ceil(1.1 × us)` of the highest
figure, update the table above, and rebuild. A run of the new TAB on each board must then give
`HILSHARE_RESULT=PASS`.
