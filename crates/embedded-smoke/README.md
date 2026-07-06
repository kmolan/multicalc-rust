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

`src/checks.rs` runs four cases against the public API. Each asserts a
known answer to a tolerance, so a wrong result panics instead of passing. All checks must also maintain `no-std`

- `lm_fit`: Levenberg-Marquardt fit of `y = a·e^(b·t)`, recovers `a`, `b`.
- `autodiff_derivative`: forward-mode autodiff of `x³` at `x = 2`, expects 12.
- `portable_path`: a plain `f64` fold, the CAS-free path exercised on M0.
- `svd_kabsch`: recovers a known rotation from a 3×3 cross-covariance via SVD.

## Pass/fail contract

The binary runs under QEMU semihosting:

- Clean finish → `debug::exit(EXIT_SUCCESS)` → QEMU exits 0.
- Panic (failed `assert!`/`expect`) → `panic-semihosting` (with the `exit`
  feature) prints the message and exits QEMU non-zero.

> Note: `cargo run … | tee` masks the exit code unless `pipefail` is set. CI runs
steps under `bash -eo pipefail`, so a panic still fails the job.

## Targets and QEMU machine

| Target                  | Codegen                | QEMU machine   |
|-------------------------|------------------------|----------------|
| `thumbv7em-none-eabi`   | Cortex-M4, soft-float  | `netduinoplus2`|
| `thumbv7em-none-eabihf` | Cortex-M4, hard-float  | `netduinoplus2`|
| `thumbv6m-none-eabi`    | Cortex-M0, CAS-free    | `netduinoplus2`|

All three run on `netduinoplus2` (Cortex-M4, FPU, flash aliased at
`0x08000000`). The thumbv6m ELF is ARMv6-M, a strict subset of ARMv7E-M, so it
executes there unchanged. The CAS-free codegen under test lives in the binary,
not the emulated core.

The runners and `rustflags` (`-Tlink.x`, `--nmagic`, `-L .`) live in
`.cargo/config.toml`.

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
