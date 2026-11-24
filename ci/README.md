# CI budget gate

`ci/budgets.toml` holds the `.text` and per-scalar stack budgets, keyed by `<binary>.<target>`.
`ci/check_budgets.sh <binary> <target> <smoke-out> [<build-args>]` measures the release smoke
binary. It fails if a gated row exceeds `budget × (1 + tolerance_pct/100)`.

## Budget-raise protocol

A red gate means that a change grew the `.text` or the stack of a binary past its tolerance. Fix it
in the same PR, in one of two ways:

1. **Regression:** shrink the change back under the budget. Prefer this.
2. **Legitimate growth:** raise the specific `text_bytes` or `stack_bytes_<scalar>` value in
   `ci/budgets.toml`. Give a one-line reason in the PR description. The reviewer approves the new
   baseline, so the number moves only in a diff that a human approved.

Do **not** raise `tolerance_pct`. It loosens every row at once and hides real regressions. The
budgets are per binary and per scalar, so the growth of one domain never uses the headroom of
another.

Time budgets on real hardware are a separate set in `tools/hilshare`. Its README has their protocol.

## Smoke-test harness

The on-target smoke binary is a hand-rolled `#[entry]` runner (`cortex-m-rt` on ARM, `riscv-rt` on
RISC-V). It runs all checks in sequence in one QEMU invocation. It prints machine-readable result
lines over semihosting (`hprintln!`, `SYS_WRITE0`) and exits QEMU with `debug::exit`. The budget
gate and the cross-ABI guard parse these lines from QEMU stdout.

The harness does not use `embedded-test` or `defmt-test`:

- Their harness needs the `probe-rs` host runner. The runner enumerates the tests through a
  probe-rs-specific semihosting operation, and resets the target for each `#[test]`.
- The plain `qemu-system-arm` / `qemu-system-riscv32 -kernel` runners do not drive that protocol,
  so the harness aborts before a test runs.
- The per-test reset is incompatible with the single-invocation stack high-water mark that the
  budget gate reads.
- `defmt` output needs a host-side RTT decoder. Plain-text semihosting lines need no decoder.

## Smoke fixtures (shared with the QA crate)

The smoke checks assert against goldens from the host QA fixtures, so the target and the host share
one source of truth. `tools/qa/src/bin/gen_smoke_fixtures.rs` reads the committed fixtures under
`tools/qa/fixtures` and writes `tools/embedded-smoke/src/fixtures.rs` as bit-exact
`f64::from_bits` consts. It also writes `tools/hilshare/src/robotics_fixtures.rs`, the goldens of
the robotics checks on silicon:

```
cargo run -p multicalc-qa --bin gen_smoke_fixtures
```

The Franka forward-kinematics check needs a whole model. `tools/qa/src/bin/gen_model_source.rs`
reads `third_party/menagerie/franka_emika_panda/panda.xml` through `multicalc-robot-model`. It writes
`tools/embedded-smoke/src/franka_panda_model.rs`, Rust source that builds the same tree, so the
target never reads a model file:

```
cargo run -p multicalc-qa --bin gen_model_source
```

The output of both generators is deterministic. CI runs each generator and then
`git diff --exit-code` on its output file, so a stale committed copy fails the build. If you change a
fixture, run the generator again. Never edit `fixtures.rs`, `robotics_fixtures.rs` or
`franka_panda_model.rs` by hand.

## Test-set tiering

The `full-smoke` feature of `embedded-smoke` (on by default) selects the check set:

| Target | Build | Check set |
|---|---|---|
| `thumbv7em-none-eabi`, `thumbv7em-none-eabihf`, `riscv32imc-unknown-none-elf` | default features | full set |
| `thumbv6m-none-eabi` | `--no-default-features` | canary: the portable (no-atomics) path, one golden (`svd_golden`), the no-panic negative path, `pid_step` and two polynomial identities |

The canary keeps the M0 image small, and it still runs on each PR.
`.github/workflows/matrix.yml` gives `--no-default-features` to the `thumbv6m` leg.

## Cross-ABI divergence guard

```
bash ci/check_cross_abi.sh <eabi> <eabihf> [<riscv> ...] [--subset <m0>]
```

The script compares the `SMOKE_VAL_<name>=` scalars of the smoke runs. It fails if a shared scalar
differs by more than a relative tolerance of `1e-9`. All legs run the same math on the same inputs,
so a gap beyond tolerance is a soft-float vs FPU / `libm` divergence.

- The full-keyset legs (`thumbv7em-none-eabi`, `thumbv7em-none-eabihf`, `riscv32imc`) are
  positional. They must expose an identical key set.
- The subset leg (`thumbv6m`) uses `--subset`. It contributes only the keys it prints.

Values are f64 in Rust `{:e}` format (shortest round-trip decimal). Each output comes from a
separate matrix leg, and the legs meet in the `cross-abi` job. A wider tolerance is a reviewed
change in the same PR, as a budget re-baseline is. Never widen it to hide a real divergence.

## Adding a binary or a scalar row

- New smoke crate: add a `[<binary>.<target>]` section for each target, and add the package name to
  the loop in `.github/workflows/matrix.yml`.
- New scalar row: add `stack_bytes_<scalar>` to the section, and make the smoke binary print
  `STACK_HWM_BYTES_<SCALAR>=<n>`. The bare `STACK_HWM_BYTES=<n>` line counts as `f64`. An absent or
  `0` row is report-only.
