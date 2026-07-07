# embedded-smoke

Runs smoke tests for all five supported platforms for `multicalc`. Must return identical numbers on 
every target it supports: x86_64 and
aarch64 Linux hosts, plus three bare-metal Cortex-M ABIs (Cortex-M4 soft-float,
Cortex-M4 hard-float, and Cortex-M0). Host tests cover the first two. Nothing but
real on-target execution can cover the last three, and that is this crate. It
exercises the `multicalc` code on every Cortex-M ABI and holds each to the same known
answers, so a divergence on any supported architecture fails the build.

It hosts  `no_std` / `no_main` smoke test that runs `multicalc` on bare-metal Cortex-M
ABIs under QEMU. It is a dev-only crate (`publish = false`, not in
`default-members`) and is never built for a host target, and `cortex-m-rt` only
links for the `thumb*` triples.

## Why a separate crate, not tests inside `multicalc`

These smoke tests cannot live next to the code they check. This is a full
program for a bare chip, not a normal test. It brings its own start-up code, its
own crash handler, and a custom link step, and it needs chip-only crates (like
`cortex-m-rt`) that only build for the chip targets. Pulling those into
`multicalc` would break it on a normal computer and drag chip-only dependencies
and build settings into a library that ships to `crates.io`. Keeping the two apart lets `multicalc`
stay small, safe, and portable, while this crate stays dev-only and never ships.

## Checks

`src/checks.rs` holds the on-target cases against the public API. Each asserts a
known answer to a tolerance, so a wrong result panics instead of passing, and
every check stays `no_std`. Golden checks assert against values taken from the
host oracle fixtures (`src/fixtures.rs`, generated from `tools/oracle`); identity
checks assert a mathematical identity that needs no fixture.

The set is tiered by the `full-smoke` feature (on by default): the canary runs on
every target including the Cortex-M0; the full set adds the heavier checks on the
`thumbv7em` ABIs only. `thumbv6m` builds with `--no-default-features`.

| Test | Targets | Details |
|------|---------|---------|
| `portable_path` | all (incl. `thumbv6m`) | Identity: plain `f64` fold, `Σ 1..=4 = 10` — the CAS-free path exercised on M0. |
| `svd_golden` | all (incl. `thumbv6m`) | Golden: singular values of a 3×3 fixture matrix vs the `svd_3x3` oracle golden. Also emits `SMOKE_VAL_svd_s*` for the cross-ABI guard. |
| `error_path_returns_err` | all (incl. `thumbv6m`) | Negative path: a singular matrix's `lu()` and an indefinite matrix's `cholesky()` return a typed `Err`, not a panic. |
| `lm_fit` | `thumbv7em` only | Golden: Levenberg-Marquardt Rosenbrock least-squares minimizer vs the `rosenbrock` oracle golden. |
| `autodiff_derivative` | `thumbv7em` only | Identity: forward-mode autodiff of `x³` at `x = 2`, expects 12. |

## Pass/fail contract

The binary runs under QEMU semihosting:

- Clean finish → `debug::exit(EXIT_SUCCESS)` → QEMU exits 0.
- Panic (failed `assert!`/`expect`) → `panic-semihosting` (with the `exit`
  feature) prints the message and exits QEMU non-zero.

> Note: `cargo run … | tee` masks the exit code unless `pipefail` is set. CI runs
steps under `bash -eo pipefail`, so a panic still fails the job.

## Targets and QEMU machine

| Target                  | Codegen                | QEMU machine   | RAM  |
|-------------------------|------------------------|----------------|------|
| `thumbv7em-none-eabi`   | Cortex-M4, soft-float  | `netduinoplus2`| 64K  |
| `thumbv7em-none-eabihf` | Cortex-M4, hard-float  | `netduinoplus2`| 64K  |
| `thumbv6m-none-eabi`    | Cortex-M0, CAS-free    | `microbit`     | 16K  |

The `thumbv7em` ABIs run on `netduinoplus2` (Cortex-M4, FPU, 64K RAM, flash at
`0x08000000`). `thumbv6m` runs on `microbit` (Cortex-M0, nRF51, 16K RAM, flash
at `0x0`) — a real M0 core, so the run now asserts both RAM-size and ISA
fidelity: an oversized image or an out-of-ISA (ARMv7E-M) instruction faults just
as it would on silicon. `build.rs` picks each target's memory map.

The runners and `rustflags` (`-Tlink.x`, `--nmagic`) live in
`.cargo/config.toml`; the per-target memory map is supplied by `build.rs`.

## Stack high-water mark

`main.rs` measures peak stack by painting free stack below the entry frame with
a sentinel and scanning for the deepest overwritten byte after the checks
return. It emits one line, `STACK_HWM_BYTES=<n>`, read by the size/stack gate.

- `PAINT`: sentinel byte (`0xAA`) written across the watched window.
- `WINDOW`: bytes of free stack to paint and scan (4 KiB). Must exceed the
  deepest check and fit the target RAM. If a printed value equals `WINDOW`, the
  window saturated. Raise it and confirm headroom.
- `GUARD`: bytes just below the current SP left untouched (the live frame), so
  painting never corrupts running code.

This is a fixed-window scan below the SP, so it needs no linker symbol and is
identical across machines. If a target cannot yield a stable number, drop its
`STACK_HWM_BYTES` line and leave that ABI `.text`-gated only.

## Running

```sh
rustup target add thumbv7em-none-eabi thumbv7em-none-eabihf thumbv6m-none-eabi
sudo apt-get install -y qemu-system-arm   # provides qemu-system-arm
cargo install cargo-binutils              # provides cargo size, for the gate

cargo run -p embedded-smoke --release --target thumbv7em-none-eabi
cargo run -p embedded-smoke --release --target thumbv7em-none-eabihf
cargo run -p embedded-smoke --release --target thumbv6m-none-eabi
```

Aliases: `cargo smoke-eabi`, `cargo smoke-eabihf`, `cargo smoke-m0`.

## Size and stack gate

`ci/budgets.toml` holds per-target `.text` and stack budgets with a shared
`tolerance_pct`. `ci/check_budgets.sh` reads `.text` via `cargo size -- -A`,
parses `STACK_HWM_BYTES` from the run output, and fails if either exceeds
`budget × (1 + tolerance_pct/100)`:

```sh
cargo run -p embedded-smoke --release --target thumbv6m-none-eabi | tee smoke.txt
bash ci/check_budgets.sh thumbv6m-none-eabi smoke.txt
```

Intentional growth must bump `ci/budgets.toml` in the same PR so it is visible
in review.

## Lints

This crate does **not** use `[lints] workspace = true`. It needs one small
`unsafe` block for the stack scan, so it sets `unsafe_code = "allow"` locally
and does not inherit the library's panic-free / no-unsafe lints.
