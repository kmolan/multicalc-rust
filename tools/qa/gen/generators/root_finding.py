"""Root-finding goldens: reference roots from scipy.optimize.

Scalar equations are bracketed and solved with brentq; square systems with the
hybr (MINPACK) method from the same start point multicalc uses, so both converge
to the same branch. Each case stores the residual at a probe point (`f_at_probe`)
so a Rust/Python formula divergence fails locally rather than as a loose mismatch.
"""

import numpy as np
from scipy.optimize import brentq, root

import problems
import schema


def _write_scalar(out, meta, case, problem, solver, params, root_val, probe, tol,
                  bracket=None, start=None):
    f = problems.scalar1(problem, **params)
    inputs = {
        "problem": schema.string(problem),
        "solver": schema.string(solver),
        "probe": schema.scalar(probe),
    }
    for key, value in params.items():
        inputs[key] = schema.scalar(value)
    if bracket is not None:
        inputs["bracket"] = schema.vector(bracket)
    if start is not None:
        inputs["start"] = schema.scalar(start)
    expected = {
        "root": schema.scalar(float(root_val)),
        "f_at_probe": schema.scalar(float(f(probe))),
    }
    schema.write_fixture(out, "root_finding", case, meta, {"f64/host": schema.tol(*tol)}, inputs, expected)


def _write_system(out, meta, case, problem, params, start, root_val, tol):
    f = problems.vectorfn(problem, **params)
    inputs = {
        "problem": schema.string(problem),
        "solver": schema.string("newton_system"),
        "start": schema.vector(start),
        "probe": schema.vector(start),
    }
    for key, value in params.items():
        inputs[key] = schema.scalar(value)
    expected = {
        "root": schema.vector([float(x) for x in root_val]),
        "f_at_probe": schema.vector([float(x) for x in f(np.array(start, dtype=float))]),
    }
    schema.write_fixture(out, "root_finding", case, meta, {"f64/host": schema.tol(*tol)}, inputs, expected)


def run(out, seed):
    meta = schema.metadata(
        "root_finding", seed, "reference roots from scipy.optimize",
        libraries=("numpy", "scipy"),
    )

    # Wien's displacement, f = -5 + x + 5 e^{-x}; the nonzero root is near 4.965.
    wien_root = brentq(problems.scalar1("root_wien"), 1.0, 10.0, xtol=1e-14)
    _write_scalar(out, meta, "wien_bisection", "root_wien", "bisection", {},
                  wien_root, probe=1.0, tol=(1e-9, 1e-9), bracket=[1.0, 10.0])
    _write_scalar(out, meta, "wien_newton", "root_wien", "newton", {},
                  wien_root, probe=5.0, tol=(1e-9, 1e-9), start=5.0)

    # Kepler's equation, e = 0.8, m = 1 - 0.8 sin(1); bracketed on [0, pi].
    e, m = 0.8, 1.0 - 0.8 * np.sin(1.0)
    kepler_root = brentq(problems.scalar1("root_kepler", e=e, m=m), 0.0, np.pi, xtol=1e-14)
    _write_scalar(out, meta, "kepler_bisection", "root_kepler", "bisection", {"e": e, "m": m},
                  kepler_root, probe=0.0, tol=(1e-9, 1e-9), bracket=[0.0, float(np.pi)])
    _write_scalar(out, meta, "kepler_newton", "root_kepler", "newton", {"e": e, "m": m},
                  kepler_root, probe=m, tol=(1e-9, 1e-9), start=m)

    # Colebrook-White friction factor, Re = 1e5, rel_roughness = 1e-4.
    reynolds, rel_roughness = 1.0e5, 1.0e-4
    colebrook_root = brentq(
        problems.scalar1("root_colebrook", reynolds=reynolds, rel_roughness=rel_roughness),
        0.005, 0.1, xtol=1e-14,
    )
    _write_scalar(out, meta, "colebrook_newton", "root_colebrook", "newton",
                  {"reynolds": reynolds, "rel_roughness": rel_roughness},
                  colebrook_root, probe=0.02, tol=(1e-9, 1e-9), start=0.02)

    # Sigmoid x / sqrt(1 + x^2); the only root is exactly 0 (closed form).
    _write_scalar(out, meta, "sigmoid_damped_newton", "root_sigmoid", "damped_newton", {},
                  0.0, probe=2.0, tol=(1e-9, 1e-9), start=2.0)

    # Two-link inverse kinematics: target built from joint angles (0.5, 0.8).
    l1, l2, t1, t2 = 1.0, 1.0, 0.5, 0.8
    px = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
    py = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
    tl_params = {"l1": l1, "l2": l2, "px": float(px), "py": float(py)}
    tl_root = root(problems.vectorfn("sys_two_link", **tl_params), [0.4, 0.9], method="hybr").x
    _write_system(out, meta, "two_link_ik", "sys_two_link", tl_params, [0.4, 0.9], tl_root, tol=(1e-9, 1e-9))

    # Circle x^2 + y^2 = 4 intersected with hyperbola xy = 1.
    ch_root = root(problems.vectorfn("sys_circle_hyperbola"), [1.5, 0.8], method="hybr").x
    _write_system(out, meta, "circle_hyperbola", "sys_circle_hyperbola", {}, [1.5, 0.8], ch_root, tol=(1e-9, 1e-9))

    # Chemical equilibrium mass balance (3x3); looser tolerance for the larger system.
    eq_root = root(problems.vectorfn("sys_equilibrium"), [0.5, 0.25, 0.25], method="hybr").x
    _write_system(out, meta, "equilibrium_3x3", "sys_equilibrium", {}, [0.5, 0.25, 0.25], eq_root, tol=(1e-8, 1e-8))
