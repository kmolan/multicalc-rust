# multicalc-oracle

Golden-value tests that check multicalc's numerics against trusted references:
numpy/LAPACK for linear algebra, MINPACK (via SciPy) for nonlinear least squares,
and mpmath for quadrature.

The reference values live as committed JSON fixtures under `fixtures/`. The Rust
tests load them, run multicalc on the same inputs, and compare within per-fixture
tolerances. Fixtures are produced by containerized Python generators in `gen/`.

**Generation is maintainer-run under Linux/WSL. CI never runs Python and never
downloads anything — it only reads the committed fixtures** (`cargo test -p
multicalc-oracle`).

## Layout

```
src/        schema, loader, and the named-problem registry
fixtures/   committed goldens, versioned as v1, v2, …
gen/        Python generators and their pinned container
tests/      one suite per module (linalg, optimization, quadrature)
```

## Fixture schema

Every fixture is one JSON object. Floating-point numbers are stored as the hex of
their IEEE-754 bit pattern (`f64::to_bits`), so values round-trip exactly and a
regenerated file is byte-for-byte comparable. Integers, strings, and booleans are
plain JSON. NaN is not allowed.

An abridged example (`fixtures/v1/quadrature/two_x_legendre_o4.json`):

```json
{
  "schema_version": 1,
  "module": "quadrature",
  "case": "two_x_legendre_o4",
  "metadata": {
    "generator": "quadrature",
    "libraries": { "mpmath": "1.3.0" },
    "seed": 20260706,
    "date": "2026-07-06T00:00:00+00:00",
    "sampling": "closed-form spot checks; mpmath goldens"
  },
  "tolerances": {
    "table": {
      "f64/host": { "abs": 1e-12, "rel": 1e-12 },
      "f32/host": { "abs": 1e-4,  "rel": 1e-4 }
    }
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
  generation date, and a note on how inputs were sampled.
- `module` / `case` — which suite the fixture belongs to and its unique name.
- `tolerances.table` — keyed `"<scalar>/<target>"` (e.g. `f64/host`, `f32/host`),
  each an absolute/relative pair. A comparison passes when
  `|got - want| <= abs + rel * max(|got|, |want|)`. The `aarch64` and `thumbv*`
  targets are reserved for later phases.
- `inputs` — the exact inputs, as hex where floating-point.
- `expected` — the reference outputs.

Value kinds: `scalar`, `vector`, `matrix` (row-major, with `rows`/`cols`), `int`,
`str`, `bool`. `quaternion` and `manifold_state` are reserved and unused today.

### What is compared

Cross-implementation comparisons use only gauge-free quantities — determinant,
inverse, solve, least-squares solution, residual norm, singular values, the
Moore–Penrose pseudo-inverse, and the (unique) Cholesky factor. Sign- and
rotation-ambiguous factors (raw Q/R/U/V) are never compared against the reference;
they are checked only through multicalc's own reconstruction identities.

The `f64` result carries the golden value. For `f32`, the same input is re-run and
a mathematical identity is asserted (reconstruction, `A·A⁺·A ≈ A`, `L·Lᵀ ≈ A`, and
so on) against the `f32/host` tolerance; `f32` results are never compared to the
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
   there) so its version is recorded in each fixture's metadata.
2. Add a generator under `gen/generators/` and call it from `gen/generate.py`.
   Compute goldens from the reference and write them with the shared `schema.py`
   builders — no schema change needed for scalars, vectors, or matrices.
3. If the outputs are rigid-body states or rotations, use the reserved
   `quaternion` / `manifold_state` value kinds and the reserved per-target
   tolerance keys (`aarch64`, `thumbv*`) rather than inventing new ones — they
   exist so this stays forward-compatible.
4. For any function-valued problem (a dynamics model, a filter step), add a stable
   key to both `gen/problems.py` and `src/problems.rs`, then a matching Rust suite
   under `tests/`.

Keep new goldens in the current `fixtures/vN/`; only a value-changing regeneration
of existing fixtures needs a new version. MuJoCo and other large downloads stay on
the maintainer's generation side — CI must remain fixture-only.

## Regeneration

Generators are pinned in a container so the recorded library versions match the
committed fixtures. The seed is fixed (`20260706`). To reproduce a fixture set
byte-for-byte, set `SOURCE_DATE_EPOCH` to the instant frozen in the fixtures'
`metadata.date` — v1 was generated with `SOURCE_DATE_EPOCH=1783296000`
(2026-07-06 UTC).

With Docker (run from the repo root):

```bash
docker build -t mc-oracle-gen tools/oracle/gen
docker run --rm --user "$(id -u):$(id -g)" \
  -e SOURCE_DATE_EPOCH=1783296000 \
  -v "$PWD/tools/oracle/fixtures:/out" mc-oracle-gen
```

`--user` keeps the written files owned by you rather than root.

Without Docker, a virtualenv with the pinned requirements works too (also from the
repo root):

```bash
python -m venv .venv
.venv/bin/pip install -r tools/oracle/gen/requirements.txt
SOURCE_DATE_EPOCH=1783296000 .venv/bin/python tools/oracle/gen/generate.py --out tools/oracle/fixtures
```

Byte-stability holds for the same image/build. A different numpy/BLAS build may
differ in the last few bits, which is exactly why comparisons use tolerances and
why such drift creates a new fixture version rather than rewriting an old one.

## Immutability and versioning

`fixtures/vN/` is immutable once committed. Regeneration that changes any value
goes into a new `fixtures/v(N+1)/` in a reviewed change; `v1` is never rewritten.

Because each fixture freezes the exact reference-library versions in its metadata,
accuracy claims are version-qualified: a fixture states that multicalc matches, for
example, numpy 2.1.3 and mpmath 1.3.0 to the listed tolerance.
