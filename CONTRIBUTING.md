# Contributing to multicalc

## 5-minute quickstart

1. Fork and clone.
2. `cargo test`
3. `cargo clippy --all-targets` — the lint wall (`no unwrap`/`expect`/`panic`/`unsafe`)
   is CI-enforced, so run it before pushing.
4. `cargo fmt`.

That's it. **You do not need QEMU or ARM toolchains** for most contributions — CI runs
the five-target embedded matrix on your PR for you. If you want the bare-metal targets
locally anyway, the setup lives in [ci/README.md](ci/README.md) (optional).

## Workflow

- Fork, branch in your fork, PR against `main`. One focused change
  per PR.
- Every behavior-facing change adds a line under `## [Unreleased]` in
  [CHANGELOG.md](CHANGELOG.md) (grouped `Added`/`Changed`/`Fixed`/`Removed`), in the
  same PR.
- CI must be green: host tests (x86_64 + aarch64), clippy, the three bare-metal QEMU
  smoke runs, and the flash/stack budget gates. If a budget gate trips, see the raise
  protocol in [ci/README.md](ci/README.md) — never raise `tolerance_pct`.

## Picking something to work on

- [`good first issue`](https://github.com/kmolan/multicalc-rust/labels/good%20first%20issue)
  — scoped and mentored; each states **what** to do, **where** in the code, and **how to
  test** it. Comment on the issue to claim it so nobody double-works.
- [`help wanted`](https://github.com/kmolan/multicalc-rust/labels/help%20wanted) — a PR
  here will be accepted if it meets the bar below.

## What makes a PR easy to merge

- **No panics on library paths** — fallible ops return typed per-module errors (`LinalgError`,
  `DiffError`, `IntegrateError`, `SolveError`), never `unwrap`. Clippy enforces this.
- **Stay generic over the scalar** — inside generic code, never call an `f64`-only
  function (it silently drops the autodiff payload). Use the `Numeric` trait surface.
- **Tests**: f64 assertions may use golden values; f32 correctness is asserted via
  mathematical identities (e.g. reconstruction, round-trips), not goldens.
- **Docs**: public APIs get a doc example; behavior notes (NaN policy, iteration
  budgets) live on the item.

## Releasing

Releases are automated from `main`:

1. In a PR, bump `version` in `Cargo.toml` and rename the `## [Unreleased]` heading in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD` (matching the new version), then open a fresh empty `## [Unreleased]` section above it. Refresh the comparison/tag links at the bottom of the changelog (point `[Unreleased]` at `vX.Y.Z...HEAD` and add a `[X.Y.Z]` tag link).
2. When that PR merges, the release workflow publishes to crates.io and creates a `vX.Y.Z` tag and GitHub release whose notes come from the matching changelog section.
3. The `Cargo.toml` version and the top dated changelog heading must match; a bump with no changelog entry fails the release.

Thank you for all the contributions!
