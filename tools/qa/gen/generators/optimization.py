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

# Starting point, operation, and equation per problem. The starts are identical to
# the ones the Rust tests use.
CASES = {
    "rosenbrock": (
        [-1.2, 1.0],
        "Rosenbrock least-squares minimizer",
        "min ‖[10(y − x²), 1 − x]‖²",
    ),
    "trigonometric6": (
        [1.0 / 6.0] * 6,
        "Trigonometric least-squares, 6 vars",
        "rᵢ = n − Σⱼcos xⱼ + i(1 − cos xᵢ) − sin xᵢ",
    ),
    "circle_fit": (
        [2.4, -0.6, 3.5],
        "Geometric circle fit, 40 points",
        "rᵢ = √((xᵢ − cₓ)² + (yᵢ − cᵧ)²) − r",
    ),
    "gaussian_peaks": (
        [2.2, 3.2, 0.7, 1.3, 6.8, 1.3],
        "Two Gaussian peaks fit, 50 samples",
        "rᵢ = Σₖ aₖ·e^(−((tᵢ − μₖ)/σₖ)²) − yᵢ",
    ),
}


def run(out, seed):
    meta = schema.metadata(
        "optimization", seed, "fixed starts; MINPACK Levenberg-Marquardt",
        libraries=("numpy", "scipy"),
        reference="SciPy/MINPACK {scipy}",
    )
    tolerances = {"f64": schema.tol(1e-7, 1e-6)}
    for key, (x0, operation, equation) in CASES.items():
        res = least_squares(problems.residual(key), np.array(x0, dtype=float), method="lm")
        inputs = {
            "problem": schema.string(key),
            "x0": schema.vector(x0),
        }
        expected = {
            "solution": schema.vector(res.x),
            "residual_norm": schema.scalar(float(np.linalg.norm(res.fun))),
        }
        schema.write_fixture(out, "optimization", key, meta, tolerances, inputs, expected,
                             equation=equation, operations=[operation])
