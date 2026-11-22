# multicalc-qa

The project's quality-assurance crate. It owns four jobs:

- **Accuracy** — golden-fixture correctness tests that check multicalc's numerics
  against trusted references: numpy/LAPACK for linear algebra, SciPy/MINPACK for
  nonlinear least squares and root finding, and mpmath for quadrature and
  closed-form goldens. Run with `cargo test -p multicalc-qa`; CI-enforced.
- **Accuracy docs** — the `gen_accuracy_tables` bin renders the per-module accuracy
  tables under `benchmarks/` (repo root) from the fixtures, held in sync by a CI
  `git diff --exit-code` guard so the numbers cannot drift from what the fixtures verify.
- **Latency** — the `latency` bench (criterion) runs latency measurements and regenerates `benchmarks/latency.md`. Run with
  `cargo bench -p multicalc-qa`. Timings are machine-dependent, so this is not CI-enforced.
- **Cross-target** — the `gen_smoke_fixtures` bin emits the embedded-smoke goldens
  (`tools/embedded-smoke/src/fixtures.rs`) bit-exact from the committed fixtures.

The reference values live as committed JSON fixtures under `fixtures/`. The Rust
tests load them, run multicalc on the same inputs, and compare within per-fixture
tolerances. Fixtures are produced by the Python generators in `gen/`, which run
against an exact, locked set of library versions.

**Generation is maintainer-run under Linux/WSL. CI never runs Python and never
downloads anything — it only reads the committed fixtures** (`cargo test -p
multicalc-qa`).

## Layout

```
src/        schema, loader, and the named-problem registry
src/bin/    generators for the smoke goldens and the accuracy-doc tables
benches/    the latency bench (criterion), regenerates benchmarks/latency.md
fixtures/   committed goldens, one directory per module
gen/        Python generators, their pinned requirements, and the regeneration script
tests/      one suite per module (calculus, linalg, optimization, root_finding, …)
```

The generated accuracy tables live in `benchmarks/` at the repo root.

## Fixture schema

Every fixture is one JSON object. Floating-point numbers are stored as the hex of
their IEEE-754 bit pattern (`f64::to_bits`), so values round-trip exactly and a
regenerated file is byte-for-byte comparable. Integers and strings are plain JSON.
NaN is not allowed.

An abridged example (`fixtures/quadrature/two_x_legendre_o4.json`):

```json
{
  "schema_version": 2,
  "module": "quadrature",
  "case": "two_x_legendre_o4",
  "metadata": {
    "generator": "quadrature",
    "libraries": { "mpmath": "1.3.0" },
    "seed": 20260706,
    "date": "2026-07-06T00:00:00+00:00",
    "sampling": "closed-form spot checks; mpmath goldens",
    "equation": "∫ 2x dx",
    "operations": ["Gauss-Legendre order 4 on [0,2]"],
    "reference": "mpmath 1.3.0"
  },
  "tolerances": {
    "f64": { "abs": 1e-12, "rel": 1e-12 },
    "f32": { "abs": 1e-4,  "rel": 1e-4 }
  },
  "inputs": {
    "integrand": { "kind": "str",    "v": "two_x" },
    "family":    { "kind": "str",    "v": "gaussian" },
    "method":    { "kind": "str",    "v": "GaussLegendre" },
    "param":     { "kind": "int",    "v": 4 },
    "limits":    { "kind": "vector", "data": ["0x0000000000000000", "0x4000000000000000"] }
  },
  "expected": {
    "integral": { "kind": "scalar", "v": "0x4010000000000000" }
  }
}
```

Fields:

- `schema_version` — the fixture format version; a loader rejects anything else.
- `metadata` — the generator, the exact reference-library versions, the seed, the
  generation date, and a note on how inputs were sampled. `equation`,
  `operations`, and `reference` are the text the accuracy tables show: the problem,
  what was computed from it (one entry per table row), and where the golden came
  from. Writing them here keeps the published description next to the code that
  produced the number.
- `module` / `case` — which suite the fixture belongs to and its unique name.
- `tolerances` — an absolute/relative pair for `f64`, and one for `f32` on the
  fixtures that rerun in single precision. A comparison passes when
  `|got - want| <= abs + rel * max(|got|, |want|)`.
- `inputs` — the exact inputs, as hex where floating-point.
- `expected` — the reference outputs.

Value kinds: `scalar`, `vector`, `matrix` (row-major, with `rows`/`cols`), `int`,
and `str`.

### What is compared

Cross-implementation comparisons use only gauge-free quantities — determinant,
inverse, solve, least-squares solution, residual norm, singular values, the
Moore–Penrose pseudo-inverse, and the (unique) Cholesky factor. Sign- and
rotation-ambiguous factors (raw Q/R/U/V) are never compared against the reference;
they are checked only through multicalc's own reconstruction identities.

The `f64` result carries the golden value. For `f32`, the same input is re-run and
a mathematical identity is asserted (reconstruction, `A·A⁺·A ≈ A`, `L·Lᵀ ≈ A`, and
so on) against the `f32` tolerance; `f32` results are never compared to the
`f64` golden.

## The named-problem registry

Quadrature integrands and least-squares residuals are functions, so they cannot
live in a fixture. A fixture references a problem by a stable string key (e.g.
`"rosenbrock"`, `"x_squared"`). The key is implemented on both sides:
`src/problems.rs` in Rust and `gen/problems.py` in Python, with the identical
formula. Adding a problem means adding the same key to **both** files; an unknown
key panics on both sides.

## Extending to other references

The harness is meant to grow to Pinocchio (`pin`), MuJoCo, FilterPy, and similar.
Adding a reference is additive:

1. Pin the package in `gen/requirements.txt` (commented placeholders are already
   there) so its version is recorded in each fixture's metadata, then rebuild
   `gen/requirements.lock` as described under Regeneration.
2. Add a generator under `gen/generators/` and call it from `gen/generate.py`.
   Compute goldens from the reference and write them with the shared `schema.py`
   builders — no schema change needed for scalars, vectors, or matrices.
3. If the outputs need a shape the value kinds do not cover — a rotation, a
   rigid-body state — add the kind to `gen/schema.py` and `src/schema.rs` together
   and bump `schema_version`. A reader rejects a version it does not know, so the
   break is visible rather than silent.
4. For any function-valued problem (a dynamics model, a filter step), add a stable
   key to both `gen/problems.py` and `src/problems.rs`, then a matching Rust suite
   under `tests/`.

Keep new goldens alongside the existing ones. MuJoCo and other large downloads
stay on the maintainer's generation side — CI must remain fixture-only.

## Regeneration

Generation runs in a virtualenv under `gen/.venv`, installed from a locked set of
library versions so the versions recorded in each fixture match the committed
ones. It needs Python 3.12. From anywhere in the repo:

```bash
tools/qa/gen/generate.sh
```

The script builds the virtualenv on first use, installs `gen/requirements.lock`,
and writes to `tools/qa/fixtures`. It also sets the four environment variables the
goldens depend on: `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` so BLAS adds
its partial sums in the same order every run, `PYTHONHASHSEED=0`, and
`SOURCE_DATE_EPOCH=1783296000` — the instant stamped into every fixture's
`metadata.date`. The seed is fixed at `20260706`. A run with no toolchain change
leaves `git status` clean.

Pass an output directory as the first argument to write somewhere else, and set
`PYTHON=` to choose a different interpreter.

### Relocking

`requirements.txt` holds the packages the generators use directly;
`requirements.lock` adds everything those pull in, with a hash for each, and is
what actually gets installed. After changing a direct pin, rebuild the lock from
`gen/`:

```bash
pip-compile --generate-hashes --allow-unsafe --strip-extras \
  --output-file=requirements.lock requirements.txt
```

Then delete `gen/.venv`, rerun `generate.sh` so the new versions are the ones
stamped into the fixtures, and review the diff.

Byte-stability holds for the same locked set. A different numpy/BLAS build may
differ in the last few bits, which is exactly why comparisons use tolerances.

## Changing a fixture

Fixtures are regenerated in place, and the diff is the review artifact. A change to
any golden value has to be explained in the change that carries it — a new
reference-library version, a corrected input, a new case. Superseded values stay
recoverable through git history.

Two things hold steady independently of that. `schema_version` guards the file
format: the loader rejects a file it does not recognise, so a format change is a
deliberate, visible break. And because each fixture freezes the exact
reference-library versions in its metadata, accuracy claims carry those versions
with them: a fixture states that multicalc matches, for example, numpy 2.1.3 and
mpmath 1.3.0 to the listed tolerance.
