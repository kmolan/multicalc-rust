"""Nonstiff ODE goldens from scipy.integrate.solve_ivp (reference RK45)."""

import numpy as np
from scipy.integrate import solve_ivp

import problems
import schema

# case, key, y0, t0, t_end, n_samples, f64 tol (abs, rel)
CASES = [
    ("exp_decay",        "exp_decay",        [1.0],                0.0, 5.0,     11, (1e-8, 1e-8)),
    ("harmonic",         "harmonic",         [1.0, 0.0],           0.0, 10.0,    11, (1e-8, 1e-8)),
    ("two_body",         "two_body",         [1.0, 0.0, 0.0, 1.0], 0.0, 2*np.pi, 11, (1e-7, 1e-7)),
    ("van_der_pol_mild", "van_der_pol_mild", [2.0, 0.0],           0.0, 20.0,    21, (1e-7, 1e-7)),
]


def run(out, seed):
    meta = schema.metadata("ode", seed, "scipy solve_ivp goldens, RK45 rtol=1e-12",
                           libraries=("scipy", "numpy"))
    for case, key, y0, t0, t_end, n, f64tol in CASES:
        rhs = problems.ode_rhs(key)
        times = list(np.linspace(t0, t_end, n))
        sol = solve_ivp(rhs, (t0, t_end), np.array(y0, float), t_eval=times,
                        method="RK45", rtol=1e-12, atol=1e-14, dense_output=False)
        assert sol.success, f"{case}: {sol.message}"
        states = sol.y.T  # shape (n, N)
        inputs = {
            "problem": schema.string(key),
            "y0": schema.vector(y0),
            "t0": schema.scalar(t0),
            "times": schema.vector(times),
        }
        expected = {"states": schema.matrix([list(row) for row in states])}
        tolerances = {"f64/host": schema.tol(*f64tol)}
        schema.write_fixture(out, "ode", case, meta, tolerances, inputs, expected)
