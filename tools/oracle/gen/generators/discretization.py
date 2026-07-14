"""Discretization + matrix-exponential goldens from scipy."""

import numpy as np
import scipy.linalg
import scipy.signal

import schema


def _tol():
    return {"f64/host": schema.tol(1e-11, 1e-10), "f32/host": schema.tol(1e-3, 1e-3)}


def _expm(out, rng, meta):
    cases = {f"rand_{n}": rng.uniform(-0.8, 0.8, size=(n, n)) for n in (2, 3, 4, 5)}
    cases["skew_3"] = np.array([[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]])
    cases["nilpotent_3"] = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    cases["stiff_diag_3"] = np.diag([-5.0, -0.5, -0.05])
    for name, a in cases.items():
        inputs = {"kind": schema.string("expm"), "A": schema.matrix(a)}
        expected = {"expm": schema.matrix(scipy.linalg.expm(a))}
        schema.write_fixture(out, "discretization", f"expm_{name}", meta, _tol(), inputs, expected)


def _zoh(out, rng, meta):
    dt = 0.1
    for (n, m) in ((2, 1), (3, 2), (4, 2)):
        a = rng.uniform(-0.8, 0.8, size=(n, n))
        b = rng.uniform(-0.8, 0.8, size=(n, m))
        f, g, *_ = scipy.signal.cont2discrete((a, b, np.eye(n), np.zeros((n, m))), dt, method="zoh")
        inputs = {"kind": schema.string("zoh"), "A": schema.matrix(a),
                  "B": schema.matrix(b), "dt": schema.scalar(dt)}
        expected = {"F": schema.matrix(f), "G": schema.matrix(g)}
        schema.write_fixture(out, "discretization", f"zoh_{n}x{m}", meta, _tol(), inputs, expected)


def _van_loan(out, rng, meta):
    dt = 0.1
    for n in (2, 3):
        a = rng.uniform(-0.8, 0.8, size=(n, n))
        m = rng.uniform(-0.8, 0.8, size=(n, n))
        qc = m @ m.T  # SPD
        blk = np.zeros((2 * n, 2 * n))
        blk[:n, :n] = -a
        blk[:n, n:] = qc
        blk[n:, n:] = a.T
        e = scipy.linalg.expm(blk * dt)
        f = e[n:, n:].T
        qd = f @ e[:n, n:]
        inputs = {"kind": schema.string("van_loan"), "A": schema.matrix(a),
                  "Qc": schema.matrix(qc), "dt": schema.scalar(dt)}
        expected = {"F": schema.matrix(f), "Qd": schema.matrix(qd)}
        schema.write_fixture(out, "discretization", f"van_loan_{n}", meta, _tol(), inputs, expected)


def _qdwn(out, meta):
    dt, var = 0.1, 2.5
    tables = {
        2: var * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]),
        3: var * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                           [dt**3 / 2, dt**2, dt],
                           [dt**2 / 2, dt, 1.0]]),
        4: var * np.array([[dt**6 / 36, dt**5 / 12, dt**4 / 6, dt**3 / 6],
                           [dt**5 / 12, dt**4 / 4, dt**3 / 2, dt**2 / 2],
                           [dt**4 / 6, dt**3 / 2, dt**2, dt],
                           [dt**3 / 6, dt**2 / 2, dt, 1.0]]),
    }
    for dim, q in tables.items():
        inputs = {"kind": schema.string("qdwn"), "dim": schema.integer(dim),
                  "dt": schema.scalar(dt), "variance": schema.scalar(var)}
        expected = {"Q": schema.matrix(q)}
        schema.write_fixture(out, "discretization", f"qdwn_{dim}", meta, _tol(), inputs, expected)


def run(out, rng, seed):
    meta = schema.metadata(
        "discretization", seed,
        "entries uniform in [-0.8, 0.8]; SPD Qc = M Mᵀ", libraries=("numpy", "scipy"),
    )
    _expm(out, rng, meta)
    _zoh(out, rng, meta)
    _van_loan(out, rng, meta)
    _qdwn(out, meta)
