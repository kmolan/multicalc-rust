# openhil

`multicalc` as a [Tock](https://tockos.org) app, for a physical Nucleo-F411RE (Cortex-M4F at
100 MHz) running the OpenHIL kernel. `embedded-smoke` runs the same math under QEMU; this
runs it on silicon and times it.

A dev-only crate (`publish = false`) in its own workspace. It links
[libtock-rs](https://github.com/tock/libtock-rs) at the kernel's fixed app addresses
(`.cargo/config.toml`) on a pinned toolchain (`rust-toolchain.toml`), so it is built from
inside this directory and never from the repository root.

## What it prints

One process, seven checks. Each check runs eight times: the first is the cold figure, the
best of the other seven the warm one.

```
OPENHIL_START_EXECUTION
OPENHIL_CHECK name=vector_dot status=PASS us=<us> budget_us=<budget> overhead_us=<round trip> cycles=<warm> cold_cycles=<cold> value=2e1
OPENHIL_CHECKS_PASSED=7/7
OPENHIL_RESULT=PASS
```

| Key | Value |
|---|---|
| `us` | The warm figure in microseconds, less `overhead_us`. The figure the budget judges |
| `budget_us` | The check's budget, from `CHECKS` in `src/checks.rs` |
| `overhead_us` | The system-call round trip subtracted: `cycle_counter_overhead`'s warm figure from this run. `0` on the two counter checks |
| `cycles` | The warm figure as the cycle counter read it |
| `cold_cycles` | The first run's reading |
| `value` | The check's headline scalar |

A check is `PASS` when all eight answers agreed with the known one and `us <= budget_us`.
`OPENHIL_RESULT` is `PASS` when all seven are. Goldens come from
`tools/embedded-smoke/src/fixtures.rs`, which `multicalc-qa` generates; never hand-write
one.

## Building

```sh
cargo install elf2tab --version 0.13.0 --locked
tools/openhil/build-tabs.sh
```

Build through the script. It sets `RUSTFLAGS`, which replaces the linker flags the
repository root's `.cargo/config.toml` gives this target for `embedded-smoke`; a bare
`cargo build` here inherits them and stops at `cannot find linker script link.x`.

Three TABs land in `tools/openhil/target/tab/`:

| TAB | Build | A run of it |
|---|---|---|
| `multicalc-openhil-pass.tab` | default | passes |
| `multicalc-openhil-fail.tab` | `--features fail` | fails: `vector_dot` expects a wrong answer |
| `multicalc-openhil-slow.tab` | `--features slow` | fails: `vector_dot` answers correctly, over its budget |

The last two prove the two gates: a wrong answer goes red, and so does a slow one.

## In CI, and on the board

`.github/workflows/openhil.yml` lints this crate and builds the TABs on every pull request,
so a change to an API a check calls has to keep this crate building.

A run on the board is not part of CI. A maintainer starts one after a change is on `main`,
naming the `Full CI` run whose `multicalc-openhil` artifact holds the TABs. It is advisory:
a red run is investigated, and blocks nothing.

## Budgets

A budget is `ceil(1.1 × us)` of the figure measured on the board, in `CHECKS` in
`src/checks.rs`. Figures repeat exactly from run to run of one build, so the 10% is
headroom for code layout, not for noise.

Measured on a Nucleo-F411RE, toolchain `1.92`, OpenHIL kernel `0.1.1`:

| Check | `us` | `budget_us` |
|---|---|---|
| `cycle_counter_sanity` | 316.12 | 348 |
| `cycle_counter_overhead` | 15.6 | 18 |
| `vector_dot` | 15.17 | 17 |
| `svd_singular_values` | 1519.71 | 1672 |
| `autodiff_derivative` | 8.27 | 10 |
| `pid_step` | 3738.71 | 4113 |
| `error_path_returns_err` | 31.04 | 35 |

Every budget is measured again when any of these moves:

- the channel in `rust-toolchain.toml`
- the OpenHIL kernel's version
- `Cargo.lock`, or the libtock-rs rev in `Cargo.toml`
- the release profile in `Cargo.toml`, or `.cargo/config.toml`

One check's budget is measured again when a change to multicalc moves that check's figure
past it. The board run goes red on `main`. Resolve it as `ci/README.md` resolves a size
budget: shrink the regression, or raise that one number with a line of justification and
the board's figure beside it. Never widen the 1.1.

To measure: build the TABs, run `multicalc-openhil-pass.tab` on the board, and read `us`
from each `OPENHIL_CHECK` line. Set each budget to `ceil(1.1 × us)`, update the table above,
and rebuild. The pass, fail and slow TABs then have to come back pass, fail and fail.
