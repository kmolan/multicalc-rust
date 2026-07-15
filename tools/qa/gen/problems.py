"""Named-problem registry mirroring ../src/problems.rs.

Every key here has an identical formula on the Rust side. Quadrature integrands
are the unweighted `f(x)`; the Gauss-Hermite/-Laguerre weight kernels are folded
in by the quadrature generator, not here.
"""

import mpmath
import numpy as np

# --- quadrature integrands ---


def integrand(key):
    table = {
        "two_x": lambda x: 2 * x,
        "quartic": lambda x: 4 * x**3 - 3 * x**2,
        "cube": lambda x: x**3,
        "x_squared": lambda x: x**2,
        "inv_1px2": lambda x: 1 / (1 + x**2),
        "exp_neg": lambda x: mpmath.e ** (-x),
        "exp_neg_sq": lambda x: mpmath.e ** (-(x**2)),
    }
    if key not in table:
        raise KeyError(f"unknown integrand key {key!r}")
    return table[key]


# --- differentiation, root-finding, and system problems ---
#
# Each key mirrors a struct in ../../testkit/src/problems.rs with the identical
# formula. Parametrized problems (Kepler, Colebrook, the two-link arm) read their
# parameters from the fixture inputs, so the numbers are shared, not duplicated.
# The calculus keys use mpmath (high-precision analytic derivatives); the root
# keys use numpy (evaluated by scipy solvers).


def scalar1(key, **params):
    """A single-variable `f(x)`. Raises KeyError on an unknown key."""
    table = {
        "cube": lambda x: x**3,
        "root_wien": lambda x: -5.0 + x + 5.0 * np.exp(-x),
        "root_sigmoid": lambda x: x / np.sqrt(1.0 + x * x),
        "root_kepler": lambda x: x
        - params["e"] * np.sin(x)
        - params["m"],
        "root_colebrook": lambda f: 1.0 / np.sqrt(f)
        + 2.0
        * np.log10(
            params["rel_roughness"] / 3.7
            + 2.51 / (params["reynolds"] * np.sqrt(f))
        ),
    }
    if key not in table:
        raise KeyError(f"unknown scalar1 key {key!r}")
    return table[key]


def scalarn(key):
    """A multi-variable scalar `f(v)` over an mpmath vector. Raises KeyError."""
    table = {
        "g_transcendental": lambda v: v[1] * mpmath.sin(v[0])
        + v[0] * mpmath.cos(v[1])
        + v[0] * v[1] * mpmath.e ** v[2],
        "hessian_target": lambda v: v[1] * mpmath.sin(v[0])
        + 2 * v[0] * mpmath.e ** v[1]
        + v[2] ** 2,
        "approx_target": lambda v: v[0] + v[1] ** 2 + v[2] ** 3,
    }
    if key not in table:
        raise KeyError(f"unknown scalarn key {key!r}")
    return table[key]


def vectorfn(key, **params):
    """A vector-to-vector `f(v)`. Raises KeyError on an unknown key."""
    table = {
        "jac_23": lambda v: [v[0] * v[1] * v[2], v[0] ** 2 + v[1] ** 2],
        "jac_66": lambda v: [
            v[0] * v[1] + v[2],
            v[1] * v[2] + v[3],
            v[2] * v[3] + v[4],
            v[3] * v[4] + v[5],
            v[4] * v[5] + v[0],
            v[5] * v[0] + v[1],
        ],
        "vfield_3d": lambda v: [v[1], -v[0], 2 * v[2]],
        "sys_two_link": lambda v: [
            params["l1"] * np.cos(v[0])
            + params["l2"] * np.cos(v[0] + v[1])
            - params["px"],
            params["l1"] * np.sin(v[0])
            + params["l2"] * np.sin(v[0] + v[1])
            - params["py"],
        ],
        "sys_circle_hyperbola": lambda v: [
            v[0] ** 2 + v[1] ** 2 - 4.0,
            v[0] * v[1] - 1.0,
        ],
        "sys_equilibrium": lambda v: [
            v[0] + v[1] + v[2] - 1.0,
            v[1] - 1.25 * v[0] ** 2,
            v[2] - 5.0 * v[0] * v[1],
        ],
    }
    if key not in table:
        raise KeyError(f"unknown vectorfn key {key!r}")
    return table[key]


# --- least-squares residuals ---

CIRCLE_POINTS = 40
GAUSS_POINTS = 50
GAUSS_TRUTH = [2.0, 3.0, 0.8, 1.5, 7.0, 1.2]


def _rosenbrock(p):
    return np.array([10.0 * (p[1] - p[0] ** 2), 1.0 - p[0]])


def _trigonometric6(p):
    n = 6
    cos_sum = np.sum(np.cos(p))
    return np.array(
        [n - cos_sum + (i + 1) * (1 - np.cos(p[i])) - np.sin(p[i]) for i in range(n)]
    )


def _circle_fit(p):
    cx, cy, r = p[0], p[1], p[2]
    i = np.arange(CIRCLE_POINTS)
    angle = 2 * np.pi * i / CIRCLE_POINTS
    px = 2.0 + 3.0 * np.cos(angle)
    py = -1.0 + 3.0 * np.sin(angle)
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2) - r


def _gaussian_peaks(p):
    t = np.arange(GAUSS_POINTS) * 10.0 / (GAUSS_POINTS - 1)
    model = np.zeros(GAUSS_POINTS)
    for k in range(2):
        a, mu, sigma = p[3 * k], p[3 * k + 1], p[3 * k + 2]
        z = (t - mu) / sigma
        model += a * np.exp(-(z**2))
    y = np.zeros(GAUSS_POINTS)
    for k in range(2):
        a, mu, sigma = GAUSS_TRUTH[3 * k], GAUSS_TRUTH[3 * k + 1], GAUSS_TRUTH[3 * k + 2]
        z = (t - mu) / sigma
        y += a * np.exp(-(z**2))
    return model - y


def residual(key):
    table = {
        "rosenbrock": _rosenbrock,
        "trigonometric6": _trigonometric6,
        "circle_fit": _circle_fit,
        "gaussian_peaks": _gaussian_peaks,
    }
    if key not in table:
        raise KeyError(f"unknown problem key {key!r}")
    return table[key]


# --- ODE right-hand sides y' = f(t, y), mirroring ../src/problems.rs ---


def ode_rhs(key):
    def exp_decay(t, y):
        return np.array([-y[0]])

    def harmonic(t, y):
        return np.array([y[1], -y[0]])

    def two_body(t, y):
        r = np.hypot(y[0], y[1])
        r3 = r**3
        return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3])

    def van_der_pol_mild(t, y):
        mu = 1.0
        return np.array([y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]])

    table = {
        "exp_decay": exp_decay,
        "harmonic": harmonic,
        "two_body": two_body,
        "van_der_pol_mild": van_der_pol_mild,
    }
    if key not in table:
        raise KeyError(f"unknown ode key {key!r}")
    return table[key]
