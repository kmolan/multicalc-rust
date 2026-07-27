"""Single-variable quadrature goldens from mpmath.

The golden is the true integral computed at high precision, so a rule's result is
compared against the exact value. Gauss-Hermite folds an `e^{-x^2}` weight and
Gauss-Laguerre an `e^{-x}` weight around the integrand; Legendre and the iterative
rules integrate it directly.
"""

import mpmath

import problems
import schema

mpmath.mp.dps = 50

INF = float("inf")

# case, integrand, family, method, param, [lo, hi], f64 tol, f32 tol (or None),
# operation, equation
CASES = [
    ("two_x_legendre_o4", "two_x", "gaussian", "GaussLegendre", 4, [0.0, 2.0], (1e-12, 1e-12), (1e-4, 1e-4),
     "Gauss-Legendre order 4 on [0,2]", "∫ 2x dx"),
    ("quartic_legendre_o4", "quartic", "gaussian", "GaussLegendre", 4, [0.0, 2.0], (1e-12, 1e-12), (1e-4, 1e-4),
     "Gauss-Legendre order 4 on [0,2]", "∫ 4x³ − 3x² dx"),
    ("cube_booles_120", "cube", "iterative", "Booles", 120, [0.0, 2.0], (1e-10, 1e-10), (1e-3, 1e-3),
     "Boole's rule, 120 intervals on [0,2]", "∫ x³ dx"),
    ("inv_1px2_trapezoidal_2p20", "inv_1px2", "iterative", "Trapezoidal", 2**20, [0.0, 1.0], (1e-3, 1e-3), None,
     "Trapezoidal, 2²⁰ intervals on [0,1]", "∫ 1/(1+x²) dx"),
    ("x_squared_hermite_o5", "x_squared", "gaussian", "GaussHermite", 5, [-INF, INF], (1e-10, 1e-10), None,
     "Gauss-Hermite order 5 on ℝ", "∫ x²e^{-x²} dx"),
    ("x_squared_laguerre_o5", "x_squared", "gaussian", "GaussLaguerre", 5, [0.0, INF], (1e-10, 1e-10), None,
     "Gauss-Laguerre order 5 on [0,∞)", "∫ x²e^{-x} dx"),
    ("two_x_booles_120", "two_x", "iterative", "Booles", 120, [0.0, 2.0], (1e-10, 1e-10), (1e-4, 1e-4),
     "Boole's rule, 120 intervals on [0,2]", "∫ 2x dx"),
    ("two_x_simpsons_120", "two_x", "iterative", "Simpsons", 120, [0.0, 2.0], (1e-10, 1e-10), (1e-4, 1e-4),
     "Simpson's rule, 120 intervals on [0,2]", "∫ 2x dx"),
    ("two_x_trapezoidal_120", "two_x", "iterative", "Trapezoidal", 120, [0.0, 2.0], (1e-10, 1e-10), (1e-4, 1e-4),
     "Trapezoidal, 120 intervals on [0,2]", "∫ 2x dx"),
    ("exp_neg_sq_booles_120", "exp_neg_sq", "iterative", "Booles", 120, [-5.0, 5.0], (1e-6, 1e-6), None,
     "Boole's rule, 120 intervals on [-5,5]", "∫ e^{-x²} dx"),
    ("quartic_legendre_o16", "quartic", "gaussian", "GaussLegendre", 16, [0.0, 2.0], (1e-12, 1e-12), (1e-4, 1e-4),
     "Gauss-Legendre order 16 on [0,2]", "∫ 4x³ − 3x² dx"),
]


def _mp_limit(v):
    if v == INF:
        return mpmath.inf
    if v == -INF:
        return -mpmath.inf
    return mpmath.mpf(v)


def _integral(method, f, lo, hi):
    a, b = _mp_limit(lo), _mp_limit(hi)
    if method == "GaussHermite":
        return mpmath.quad(lambda x: f(x) * mpmath.e ** (-x * x), [a, b])
    if method == "GaussLaguerre":
        return mpmath.quad(lambda x: f(x) * mpmath.e ** (-x), [a, b])
    return mpmath.quad(f, [a, b])


def run(out, seed):
    meta = schema.metadata(
        "quadrature", seed, "closed-form spot checks; mpmath goldens",
        libraries=("mpmath",),
        reference="mpmath {mpmath}",
    )
    for case, integrand, family, method, param, limits, f64tol, f32tol, operation, equation in CASES:
        f = problems.integrand(integrand)
        value = _integral(method, f, limits[0], limits[1])

        tolerances = {"f64": schema.tol(*f64tol)}
        if f32tol is not None:
            tolerances["f32"] = schema.tol(*f32tol)

        inputs = {
            "integrand": schema.string(integrand),
            "family": schema.string(family),
            "method": schema.string(method),
            "param": schema.integer(param),
            "limits": schema.vector(limits),
        }
        expected = {"integral": schema.scalar(float(value))}
        schema.write_fixture(out, "quadrature", case, meta, tolerances, inputs, expected,
                             equation=equation, operations=[operation])
