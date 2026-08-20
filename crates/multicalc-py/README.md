# multicalc-py

Python bindings for [`multicalc`](../multicalc), built with PyO3 and maturin.

This is a workspace-internal crate (`publish = false`): host-only development bindings,
not published to crates.io or PyPI on their own. Import name is `multicalc_py`.

Requires CPython **3.10+**. On **3.14**, the crate needs PyO3 **0.27+** (already pinned
in `Cargo.toml`).

## Develop

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip maturin pytest
maturin develop
```

`maturin develop` needs an active venv (or a `.venv` in this directory / a parent). Prefer
`python -m pytest` so you do not pick up a system `pytest`.

## Test

```bash
python -m pytest tests/ -v
cargo clippy -p multicalc-py -- -D warnings
```

PR CI runs the same path on Python 3.12 (`maturin develop` + `pytest`) and
`cargo clippy -p multicalc-py -- -D warnings`.

## What is exposed

| Python name | Notes |
| --- | --- |
| `version()` | crate version string |
| `LinalgError` | exception for linear-algebra failures |
| `Vector4` | `.dot` |
| `Matrix2` | `.cholesky` (raises on failure; success is None) |
| `Matrix3` | `.zeros`, `.lu_decompose`, (raises on failure; success is None); `.svd` returns singular values only |
| `Pid` | constructor gains + timestep; `.update` |
| `Polynomial3` | `.real_roots` |
| `Polynomial8` | `.evaluate_with_derivatives` |
| `SO3` | `.exp`, `.act` |

Coverage is a thin slice of `multicalc` for smoke-testing the binding path, not a full
Python API.

## Example

```python
import multicalc_py

print(multicalc_py.version())

a = multicalc_py.Vector4([1.0, 2.0, 3.0, 4.0])
b = multicalc_py.Vector4([4.0, 3.0, 2.0, 1.0])
assert a.dot(b) == 20.0

pid = multicalc_py.Pid(2.0, 1.0, 0.0, 0.01)
assert abs(pid.update(1.0, 0.0) - 2.01) < 1e-12
```
