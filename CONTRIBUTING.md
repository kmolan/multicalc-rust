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

## Where does a check go?

The workspace has several test and demo layers. Each has one job; adding a check means picking
the layer that matches and not duplicating another's.

| Layer | Purpose | Must not |
|---|---|---|
| doctests | one minimal runnable demo per public item | become the correctness suite |
| `src/**/test.rs` inline | white-box tests of `pub(crate)` internals only (LU/lmpar) | test public API |
| `tests/suite/` | **the** correctness suite: public API, edge cases, proptests | re-declare problems/helpers inline |
| `demos/examples/basics/` | copy-pasteable, headless, terminating demos; multicalc-only imports | exit 0 without ≥1 sanity `assert!`; touch a sink |
| `demos/examples/showcase/` | live Rerun demos; measured numbers only | panic on edge cases (errors render as demo states); hardcode a perf claim |
| `benches/` | timing; `.md` tables are labeled illustrative snapshots | present tables as verified claims |
| `tools/oracle` | cross-implementation goldens (numpy/mpmath/MINPACK) only | duplicate self-consistency tests |
| `tools/embedded-smoke` | on-target FP-path + stack/text budgets; goldens only via generated `fixtures.rs` | hand-write golden values |

Shared problem definitions and tolerance helpers live in `tools/testkit`, so a problem is
declared once and reused across `tests/suite/`, the oracle, and embedded-smoke.

## Releasing

Releases are automated from `main`:

1. In a PR, bump `version` in `Cargo.toml` and rename the `## [Unreleased]` heading in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD` (matching the new version), then open a fresh empty `## [Unreleased]` section above it. Refresh the comparison/tag links at the bottom of the changelog (point `[Unreleased]` at `vX.Y.Z...HEAD` and add a `[X.Y.Z]` tag link).
2. When that PR merges, the release workflow publishes to crates.io and creates a `vX.Y.Z` tag and GitHub release whose notes come from the matching changelog section.
3. The `Cargo.toml` version and the top dated changelog heading must match; a bump with no changelog entry fails the release.

Thank you for all the contributions!
