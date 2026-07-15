# CI budget gate

`ci/budgets.toml` holds `.text` and per-scalar stack budgets, keyed by `<binary>.<target>`.
`ci/check_budgets.sh <binary> <target> <smoke-out>` measures the release smoke binary and fails
if any gated row exceeds `budget × (1 + tolerance_pct/100)`.

## Budget-raise protocol

A red gate means a change grew a binary's `.text` or stack past its tolerance. Resolve it one of
two ways, **in the same PR**:

1. **Regression** — investigate and shrink the change back under budget. Prefer this.
2. **Legitimate growth** — raise the specific `text_bytes` / `stack_bytes_<scalar>` value in
   `ci/budgets.toml`, with a one-line justification in the PR description. The reviewer approves the
   new baseline. This is the line between a **reviewed re-baseline** and **silent bloat**: the number
   moves in a diff a human signed off on.

Do **not** raise `tolerance_pct` to get under a gate — that loosens every row at once and hides real
regressions. Budgets are per-binary and per-scalar precisely so one domain's growth never spends
another's headroom; keep them tight and per-row.

## Smoke-test harness

The on-target smoke binary is a hand-rolled `cortex-m-rt` `#[entry]` runner. It runs every check
in sequence in a single QEMU invocation, prints machine-readable result lines over semihosting
(`hprintln!`, i.e. `SYS_WRITE0`), and exits QEMU with `debug::exit`. The budget gate and the
cross-ABI guard parse those lines straight from QEMU stdout.

`embedded-test`/`defmt-test` are not used. Their harness is driven by the `probe-rs` host runner:
it enumerates test cases through a probe-rs-specific semihosting operation and resets the target to
run each `#[test]` in its own invocation. Under the plain `qemu-system-arm -kernel` runner these
smoke tests use, nothing drives that protocol, so the harness aborts before any test runs. The
per-test reset model is also incompatible with the single-invocation stack high-watermark
measurement the budget gate reads. `defmt` output is avoided for the same parsing reason: its RTT
transport needs a host-side decoder, while the plain-text semihosting lines need none.

## Smoke fixtures (shared with the QA crate)

The smoke checks assert against goldens taken from the host QA fixtures, so the target and the
host share one source of truth. `tools/qa/src/bin/gen_smoke_fixtures.rs` reads the committed
fixtures under `tools/qa/fixtures/v1` and writes `tools/embedded-smoke/src/fixtures.rs` as
bit-exact `f64::from_bits` consts. Regenerate with:

```
cargo run -p multicalc-qa --bin gen_smoke_fixtures
```

The output is deterministic. CI regenerates it and runs `git diff --exit-code` on `fixtures.rs`, so
a stale checked-in copy fails the build. Change a fixture and rerun the generator; never edit
`fixtures.rs` by hand.

## Test-set tiering

The `embedded-smoke` `full-smoke` feature (on by default) selects the check set. Both `thumbv7em`
ABIs build with default features and run the full set; `thumbv6m` builds with `--no-default-features`
and runs only the canary — the portable (no-atomics) path, one golden, and the no-panic negative
path — which keeps the M0 image small while still smoke-running every PR. The target matrix threads
`--no-default-features` into the `thumbv6m` run (see `.github/workflows/matrix.yml`).

## Cross-ABI divergence guard

`ci/check_cross_abi.sh <eabi-out> <eabihf-out>` compares the soft-float (`thumbv7em-none-eabi`) and
hardware-FPU (`thumbv7em-none-eabihf`) smoke runs and fails if any shared `SMOKE_VAL_<name>=` scalar
differs by more than a relative tolerance (`1e-9`). Both ABIs run the same math on the same inputs,
so their headline scalars must agree; a gap beyond tolerance is a soft-float-vs-FPU / `libm`
divergence. Values are printed as f64 in Rust `{:e}` (shortest round-trip decimal). The two outputs
come from separate matrix legs and meet in the `cross-abi` job. Widening the tolerance is a reviewed
change in the same PR — like a budget re-baseline, never loosened to dodge a real divergence.

## Adding a binary or a scalar row

- New smoke crate (v0.9+): add a `[<binary>.<target>]` section per target and add the package name
  to the loop in `.github/workflows/matrix.yml`.
- New scalar row: add `stack_bytes_<scalar>` to the section and make the smoke binary print
  `STACK_HWM_BYTES_<SCALAR>=<n>`. An absent or `0` row is report-only.
