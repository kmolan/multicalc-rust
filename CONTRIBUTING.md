# Contributing to multicalc

## Quickstart

1. Fork and clone the repository.
2. `cargo test`
3. `cargo clippy -p multicalc --all-targets --features alloc -- -D warnings`. CI enforces the
   lint wall: no `unwrap`, `expect`, `panic` or `unsafe`.
4. `cargo fmt --all`

You do not need QEMU or a bare-metal toolchain. CI runs the embedded target matrix on each PR. To
run the bare-metal targets locally, see [tools/embedded-smoke/README.md](tools/embedded-smoke/README.md).

## Workflow

- Fork, make a branch in your fork, and open a PR against `main`. Keep one focused change in each PR.
- Each behavior change adds a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md), in the
  same PR. Group the lines under `Added`, `Changed`, `Fixed` or `Removed`.
- CI must be green. The per-PR gates are in [`.github/workflows/ci.yml`](.github/workflows/ci.yml):
  host tests, clippy, docs, QA fixtures, Python bindings, the four bare-metal QEMU smoke runs, the
  `.text` and stack budget gates, and the HILShare TAB build.
- If a budget gate fails, obey the raise protocol in [ci/README.md](ci/README.md). Never raise
  `tolerance_pct`.

## Pick an issue

- [`good first issue`](https://github.com/kmolan/multicalc-rust/labels/good%20first%20issue): each
  issue states what to do, where in the code, and how to test it. Comment on the issue to claim it.
- [`help wanted`](https://github.com/kmolan/multicalc-rust/labels/help%20wanted): we accept a PR here
  if it meets the rules below.

## PR rules

- **No panics on library paths.** A fallible operation returns a typed module error (`LinalgError`,
  `DiffError`, `IntegrateError`, `SolveError`), never `unwrap`. Clippy enforces this.
- **A fallible function has a `try_` prefix** (`try_row`, `try_from_slice`), for `Option` and
  `Result` alike.
- **A `Mut` type mirrors its read-only twin**, plus the write methods. Document each gap.
- **Stay generic over the scalar.** In generic code, never call an `f64`-only function: it drops the
  autodiff payload. Use the `Numeric` trait surface.
- **Descriptive names:** `matrix`, not `m`.
- **Tests:** an `f64` assertion can use a golden value. An `f32` assertion uses a mathematical
  identity (reconstruction, round trip), not a golden.
- **Docs:** each public API has a doc example: one or two sentences, then the example. Do not name a
  method that does not exist. Put behavior notes (NaN policy, iteration budgets) on the item. A new
  API adds a line to its [tutorial page](crates/multicalc/tutorials/), compiled as a doctest.

## Where does a check go?

Each test layer has one job. Put a check in the layer that matches it, and do not duplicate
another layer.

| Layer | Purpose | Must not |
|---|---|---|
| doctests | one minimal runnable demo per public item | become the correctness suite |
| `src/**/test.rs` inline | white-box tests of `pub(crate)` internals only | test public API |
| `crates/multicalc/tests/suite/` | **the** correctness suite: public API, edge cases, proptests | re-declare problems/helpers inline |
| `demos/examples/basics/` | copy-pasteable, headless, terminating demos; multicalc-only imports | exit 0 without ≥1 sanity `assert!`; touch a sink |
| `demos/examples/showcase/` | live Rerun demos; measured numbers only | panic on edge cases (errors render as demo states); hardcode a perf claim |
| `tools/qa` | cross-implementation goldens (numpy/mpmath/MINPACK); `.md` accuracy tables generated from the fixtures | duplicate self-consistency tests; hand-edit the generated tables |
| `tools/embedded-smoke` | on-target FP-path + stack/text budgets; goldens only via generated `fixtures.rs` | hand-write golden values |
| `tools/hilshare` | answers and time budgets on the three HILShare boards, as a Tock app; goldens from the generated `fixtures.rs` of embedded-smoke | write a golden value by hand; move a budget with no board figure |

`tools/testkit` holds the shared problem definitions and tolerance helpers. Declare a problem one
time there, and use it in `tests/suite/`, the QA crate and embedded-smoke.

To regenerate the QA fixtures, run `tools/qa/gen/generate.sh`. It needs Python 3.12. See
[Regeneration](tools/qa/README.md#regeneration).

## Releasing

The release workflow ([`.github/workflows/release.yml`](.github/workflows/release.yml)) runs when a
`Cargo.toml` version bump lands on `main`:

1. It requires a `## [X.Y.Z]` heading in `CHANGELOG.md` that matches the `Cargo.toml` version. If
   the heading is not there, the release fails.
2. It runs the full target matrix.
3. It publishes `multicalc` and then `multicalc-robot-model` to crates.io, at the same version.
   Then it creates the `vX.Y.Z` tag and a GitHub release. The release notes come from the
   matching changelog section.

The bump also changes the `multicalc` requirement in
[`crates/multicalc-robot-model/Cargo.toml`](crates/multicalc-robot-model/Cargo.toml), for
example from `"0.10"` to `"0.11"`. If the requirement is stale, each build fails.

If `multicalc` publishes and `multicalc-robot-model` fails, a re-run fails too, because the
`multicalc` version exists. Then do these steps by hand:

1. Run `cargo publish -p multicalc-robot-model`.
2. Run `gh release create vX.Y.Z --title vX.Y.Z --notes-file <notes>`, with the changelog
   section in `<notes>`.

Before you merge the bump, build the docs as docs.rs does and examine the result. docs.rs rebuilds
only on a publish, so a broken page stays broken until the next release.

```sh
RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc -p multicalc --no-deps --all-features
```

The `--cfg docsrs` flag is nightly-only. It adds the "Available on crate feature `alloc` only"
badges. Make sure that the `alloc`-gated items (`ParticleFilter`, `Jacobian::evaluate_on_heap`) are
present and have the badge.
