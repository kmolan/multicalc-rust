"""Linear-algebra goldens from numpy/LAPACK.

Only gauge-free quantities are stored (determinant, inverse, solve, least-squares
solution, residual norm, singular values, pseudo-inverse, and the unique Cholesky
factor). Raw Q/R/U/V are never emitted; the Rust side checks those through its own
reconstruction identities. Inputs are sampled with the passed RNG and written as
exact bits, so the tests never depend on the RNG.
"""

import numpy as np

import schema


def _tolerances(f32=(1e-3, 1e-3)):
    return {"f64/host": schema.tol(1e-11, 1e-10), "f32/host": schema.tol(*f32)}


def _sample_well_conditioned(rng, rows, cols, max_cond=1e4):
    """Uniform entries in [-1, 1], rejecting ill-conditioned draws so every
    decomposition's precondition (non-singular / full column rank) holds."""
    while True:
        a = rng.uniform(-1.0, 1.0, size=(rows, cols))
        s = np.linalg.svd(a, compute_uv=False)
        if s[-1] > 0.0 and s[0] / s[-1] < max_cond:
            return a


def _lu(out, rng, meta):
    for n in (3, 4, 5):
        a = _sample_well_conditioned(rng, n, n)
        b = rng.uniform(-1.0, 1.0, size=n)
        inputs = {
            "decomp": schema.string("lu"),
            "A": schema.matrix(a),
            "b": schema.vector(b),
        }
        expected = {
            "det": schema.scalar(float(np.linalg.det(a))),
            "x": schema.vector(np.linalg.solve(a, b)),
            "inv": schema.matrix(np.linalg.inv(a)),
        }
        schema.write_fixture(out, "linalg", f"lu_{n}x{n}", meta, _tolerances(), inputs, expected)


def _emit_qr(out, meta, a, rows, cols, rng, f32):
    b = rng.uniform(-1.0, 1.0, size=rows)
    x_ls, *_ = np.linalg.lstsq(a, b, rcond=None)
    residual_norm = float(np.linalg.norm(a @ x_ls - b))
    inputs = {
        "decomp": schema.string("qr"),
        "A": schema.matrix(a),
        "b": schema.vector(b),
    }
    expected = {
        "x_ls": schema.vector(x_ls),
        "residual_norm": schema.scalar(residual_norm),
    }
    schema.write_fixture(
        out, "linalg", f"qr_{rows}x{cols}", meta, _tolerances(f32), inputs, expected
    )


def _qr(out, rng, meta):
    for rows, cols in ((3, 2), (4, 3), (3, 3)):
        a = _sample_well_conditioned(rng, rows, cols)
        _emit_qr(out, meta, a, rows, cols, rng, (1e-3, 1e-3))
    # A well-conditioned tall case: a Vandermonde system on evenly spaced nodes.
    nodes = np.linspace(-1.0, 1.0, 20)
    a = np.vander(nodes, 7, increasing=True)
    _emit_qr(out, meta, a, 20, 7, rng, (1e-2, 1e-2))


def _svd(out, rng, meta):
    shapes = ((3, 2), (3, 3), (4, 3), (12, 6), (20, 6))
    f32_for = {(3, 2): (1e-3, 1e-3), (3, 3): (1e-3, 1e-3), (4, 3): (1e-3, 1e-3),
               (12, 6): (1e-2, 1e-2), (20, 6): (1e-2, 1e-2)}
    for rows, cols in shapes:
        a = _sample_well_conditioned(rng, rows, cols)
        b = rng.uniform(-1.0, 1.0, size=rows)
        x_ls, *_ = np.linalg.lstsq(a, b, rcond=None)
        inputs = {
            "decomp": schema.string("svd"),
            "A": schema.matrix(a),
            "b": schema.vector(b),
        }
        expected = {
            "singular_values": schema.vector(np.linalg.svd(a, compute_uv=False)),
            "x_ls": schema.vector(x_ls),
            "pinv": schema.matrix(np.linalg.pinv(a)),
        }
        schema.write_fixture(
            out, "linalg", f"svd_{rows}x{cols}", meta, _tolerances(f32_for[(rows, cols)]), inputs, expected
        )


def _cholesky(out, rng, meta):
    for n in (2, 3, 4):
        m = rng.uniform(-1.0, 1.0, size=(n, n))
        a = m @ m.T + n * np.eye(n)  # symmetric positive definite by construction
        b = rng.uniform(-1.0, 1.0, size=n)
        inputs = {
            "decomp": schema.string("cholesky"),
            "A": schema.matrix(a),
            "b": schema.vector(b),
        }
        expected = {
            "L": schema.matrix(np.linalg.cholesky(a)),
            "det": schema.scalar(float(np.linalg.det(a))),
            "x": schema.vector(np.linalg.solve(a, b)),
        }
        schema.write_fixture(
            out, "linalg", f"cholesky_{n}x{n}", meta, _tolerances(), inputs, expected
        )


def run(out, rng, seed):
    meta = schema.metadata(
        "linalg", seed, "entries uniform in [-1, 1], ill-conditioned draws rejected",
        libraries=("numpy",),
    )
    _lu(out, rng, meta)
    _qr(out, rng, meta)
    _svd(out, rng, meta)
    _cholesky(out, rng, meta)
