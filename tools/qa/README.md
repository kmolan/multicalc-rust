# multicalc-qa

The quality-assurance crate of the project. It has these jobs:

| Job | Tool | CI |
|---|---|---|
| **Accuracy:** golden-fixture tests against trusted references (numpy/LAPACK for linear algebra, SciPy/MINPACK for least squares and root finding, mpmath for quadrature and closed forms, and FilterPy, MuJoCo, mink and Pinocchio for the robotics modules) | `cargo test -p multicalc-qa` | enforced |
| **Accuracy docs:** the per-module accuracy tables under `benchmarks/` (repo root), rendered from the fixtures | `gen_accuracy_tables` bin | `git diff --exit-code` guard |
| **Latency:** criterion measurements, written to `benchmarks/latency.md` | `cargo bench -p multicalc-qa` | not enforced (machine-dependent) |
| **Cross-target:** the embedded-smoke goldens (`tools/embedded-smoke/src/fixtures.rs`), bit-exact from the committed fixtures | `gen_smoke_fixtures` bin | `git diff --exit-code` guard |
| **Model source:** `tools/embedded-smoke/src/franka_panda_model.rs` from the Menagerie Franka MJCF | `gen_model_source` bin | `git diff --exit-code` guard |

The reference values are committed JSON fixtures under `fixtures/`. The Rust tests load them, run
multicalc on the same inputs, and compare within per-fixture tolerances. The Python generators in
`gen/` produce the fixtures against a locked set of library versions.

**A maintainer runs the generation under Linux/WSL. CI never runs Python and never downloads
anything. It only reads the committed fixtures.**

## Layout

```
src/        schema, loader, and the named-problem registry
src/bin/    generators for the smoke goldens, the model source and the accuracy-doc tables;
            bootstrap_thresholds and spatial_small_angle.py for the spatial::lie small-angle cutoffs
benches/    the latency bench (criterion), regenerates benchmarks/latency.md
fixtures/   committed goldens, one directory per module
gen/        Python generators, their pinned requirements, and the regeneration script
tests/      one suite per module (calculus, linalg, optimization, root_finding, …)
```

## Fixture schema

Each fixture is one JSON object. A floating-point number is the hex of its IEEE-754 bit pattern
(`f64::to_bits`), so values round-trip exactly and a regenerated file is byte-for-byte comparable.
Integers and strings are plain JSON. NaN is not permitted.

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

| Field | Meaning |
|---|---|
| `schema_version` | the fixture format version; the loader rejects all other versions |
| `module` / `case` | the suite of the fixture and its unique name |
| `metadata` | the generator, the exact reference-library versions, the seed, the generation date, and a note on how the inputs were sampled. `equation`, `operations` (one entry per table row) and `reference` are the text of the accuracy tables |
| `tolerances` | an abs/rel pair for `f64`, and one for `f32` on the fixtures that run again in single precision. A comparison passes when `\|got - want\| <= abs + rel * max(\|got\|, \|want\|)` |
| `inputs` | the exact inputs, as hex where floating-point |
| `expected` | the reference outputs |

Value kinds: `scalar`, `vector`, `matrix` (row-major, with `rows`/`cols`), `int`, and `str`.

### What is compared

Cross-implementation comparisons use only gauge-free quantities: determinant, inverse, solve,
least-squares solution, residual norm, singular values, the Moore–Penrose pseudo-inverse, and the
unique Cholesky factor. Sign- and rotation-ambiguous factors (raw Q/R/U/V) are never compared
against the reference. Only the reconstruction identities of multicalc check them.

The `f64` result carries the golden value. For `f32`, the test runs the same input again and asserts
a mathematical identity (reconstruction, `A·A⁺·A ≈ A`, `L·Lᵀ ≈ A`) against the `f32` tolerance. An
`f32` result is never compared to the `f64` golden.

## The named-problem registry

Quadrature integrands and least-squares residuals are functions, so they cannot be in a fixture. A
fixture names a problem by a stable string key (`"rosenbrock"`, `"x_squared"`). Both sides implement
the key with the identical formula:

- Rust: `tools/testkit/src/problems.rs`, re-exported by `src/problems.rs`. The ODE right-hand sides
  are in `src/problems.rs`.
- Python: `gen/problems.py`.

To add a problem, add the same key on **both** sides. An unknown key panics in Rust and raises
`KeyError` in Python.

## Add a reference library

1. Pin the package in `gen/requirements.txt`, so each fixture records its version. Then rebuild
   `gen/requirements.lock` (see [Relocking](#relocking)).
2. Add a generator under `gen/generators/` and call it from `gen/generate.py`. Write the goldens
   with the shared `schema.py` builders. Scalars, vectors and matrices need no schema change.
3. If an output needs a shape that the value kinds do not cover (a rotation, a rigid-body state),
   add the kind to `gen/schema.py` and `src/schema.rs` together, and bump `schema_version`. The
   loader rejects an unknown version, so the break is visible.
4. For a function-valued problem (a dynamics model, a filter step), add a stable key to
   `gen/problems.py` and the Rust registry. Then add a matching Rust suite under `tests/`.

Keep new goldens with the existing ones. Large downloads such as MuJoCo stay on the generation side.
CI must stay fixture-only.

## Regeneration

Generation needs Python 3.12. It runs in a virtualenv under `gen/.venv`, installed from the locked
library versions, so the versions in each fixture match the committed ones. From anywhere in the
repo:

```bash
tools/qa/gen/generate.sh
```

The script builds the virtualenv on first use, installs `gen/requirements.lock`, and writes to
`tools/qa/fixtures`. It sets the environment variables that the goldens depend on:

| Variable | Value | Reason |
|---|---|---|
| `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` | `1` | BLAS adds its partial sums in the same order on each run |
| `PYTHONHASHSEED` | `0` | fixed hash seed |
| `SOURCE_DATE_EPOCH` | `1783296000` | the instant stamped into `metadata.date` of each fixture |

The seed is `20260706`. A run with no toolchain change leaves `git status` clean.

To write somewhere else, give an output directory as the first argument. To use a different
interpreter, set `PYTHON=`.

### Relocking

`requirements.txt` holds the packages that the generators use directly. `requirements.lock` adds
their dependencies with a hash for each, and pip installs only the lock. After you change a direct
pin, rebuild the lock from `gen/`:

```bash
pip-compile --generate-hashes --allow-unsafe --strip-extras \
  --output-file=requirements.lock requirements.txt
```

Then delete `gen/.venv`, run `generate.sh` again so the fixtures record the new versions, and review
the diff.

Byte-stability holds only for the same locked set. A different numpy/BLAS build can differ in the
last bits, so the comparisons use tolerances.

## Changing a fixture

Regenerate fixtures in place. The diff is the review artifact. Explain each golden-value change in
the change that carries it: a new reference-library version, a corrected input, or a new case. Git
history keeps the old values.

`schema_version` guards the file format: the loader rejects an unknown format, so a format change is
a visible break. Each fixture records the exact reference-library versions, so an accuracy claim
carries them: for example, multicalc matches numpy 2.1.3 and mpmath 1.3.0 to the listed tolerance.
