"""Nonlinear least-squares goldens from MINPACK via scipy.

Each problem is solved with `scipy.optimize.least_squares(method="lm")`. The
comparison quantity is the residual norm, not scipy's `cost` (which carries a 0.5
factor that multicalc does not), so the golden is convention-free and recomputed
identically on the Rust side.
"""

import numpy as np
from scipy.optimize import least_squares

import problems
import schema

# Starting points, identical to the ones the Rust tests use.
X0 = {
    "rosenbrock": [-1.2, 1.0],
    "trigonometric6": [1.0 / 6.0] * 6,
    "circle_fit": [2.4, -0.6, 3.5],
    "gaussian_peaks": [2.2, 3.2, 0.7, 1.3, 6.8, 1.3],
}


def run(out, seed):
    meta = schema.metadata(
        "optimization", seed, "fixed starts; MINPACK Levenberg-Marquardt",
        libraries=("numpy", "scipy"),
    )
    tolerances = {"f64/host": schema.tol(1e-7, 1e-6)}
    for key, x0 in X0.items():
        res = least_squares(problems.residual(key), np.array(x0, dtype=float), method="lm")
        inputs = {
            "problem": schema.string(key),
            "x0": schema.vector(x0),
        }
        expected = {
            "solution": schema.vector(res.x),
            "residual_norm": schema.scalar(float(np.linalg.norm(res.fun))),
        }
        schema.write_fixture(out, "optimization", key, meta, tolerances, inputs, expected)
