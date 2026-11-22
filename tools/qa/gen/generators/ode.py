"""Nonstiff ODE and attitude goldens from scipy.integrate.solve_ivp (reference RK45)."""

import numpy as np
from scipy.integrate import solve_ivp

import problems
import schema

# case, key, y0, t0, t_end, n_samples, f64 tol (abs, rel), operation, equation
CASES = [
    ("exp_decay",        "exp_decay",        [1.0],                0.0, 5.0,     11, (1e-8, 1e-8),
     "Exponential decay, RK45", "y' = -y"),
    ("harmonic",         "harmonic",         [1.0, 0.0],           0.0, 10.0,    11, (1e-8, 1e-8),
     "Harmonic oscillator, RK45", "y' = [y₁, -y₀]"),
    ("two_body",         "two_body",         [1.0, 0.0, 0.0, 1.0], 0.0, 2*np.pi, 11, (1e-7, 1e-7),
     "Two-body orbit, RK45", "y' = [vₓ, vᵧ, -x/r³, -y/r³]"),
    ("van_der_pol_mild", "van_der_pol_mild", [2.0, 0.0],           0.0, 20.0,    21, (1e-7, 1e-7),
     "Van der Pol (μ=1), RK45", "y' = [y₁, (1 - y₀²)·y₁ - y₀]"),
]

# The same layout with the fixed step the Rust side is to walk the samples at, appended. These
# cases drive ExponentialMap and RigidBody::stepped rather than an adaptive solver, so the step
# belongs in the fixture instead of being hardcoded in the test.
ATTITUDE_CASES = [
    ("prescribed_rate_attitude_first_order",  "prescribed_rate_attitude",
     [1.0, 0.0, 0.0, 0.0], 0.0, 4.0, 9, (5e-4, 5e-4),
     "Attitude under a set turn rate, exp-map first order (step 1e-4)",
     "q̇ = ½ q ⊗ [0, ω(t)]", 1e-4),
    ("prescribed_rate_attitude_second_order", "prescribed_rate_attitude",
     [1.0, 0.0, 0.0, 0.0], 0.0, 4.0, 9, (2.5e-6, 2.5e-6),
     "Attitude under a set turn rate, exp-map midpoint (step 1e-3)",
     "q̇ = ½ q ⊗ [0, ω(t)]", 1e-3),
    ("tumbling_free_body",                    "tumbling_free_body",
     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 3.0],
     0.0, 2.0, 11, (2e-5, 2e-5),
     "Free body under gravity and a steady push, exp-map step (step 1e-3)",
     "ṗ = v, q̇ = ½ q ⊗ [0, ω], v̇ = R·f/m + g, ω̇ = I⁻¹(τ − ω × Iω)", 1e-3),
]


def run(out, seed):
    meta = schema.metadata("ode", seed, "scipy solve_ivp goldens, RK45 rtol=1e-12",
                           libraries=("scipy", "numpy"),
                           reference="SciPy solve_ivp {scipy}")
    for case, key, y0, t0, t_end, n, f64tol, operation, equation in CASES:
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
        tolerances = {"f64": schema.tol(*f64tol)}
        schema.write_fixture(out, "ode", case, meta, tolerances, inputs, expected,
                             equation=equation, operations=[operation])

    attitude_meta = schema.metadata("ode", seed, "scipy solve_ivp attitude goldens, RK45 rtol=1e-12",
                                    libraries=("scipy", "numpy"),
                                    reference="SciPy solve_ivp {scipy}")
    for case, key, y0, t0, t_end, n, f64tol, operation, equation, timestep in ATTITUDE_CASES:
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
            "timestep": schema.scalar(timestep),
        }
        expected = {"states": schema.matrix([list(row) for row in states])}
        tolerances = {"f64": schema.tol(*f64tol)}
        schema.write_fixture(out, "ode", case, attitude_meta, tolerances, inputs, expected,
                             equation=equation, operations=[operation])
