"""Control goldens from scipy: the two matrix equations behind an optimal linear feedback law,
and the geometric attitude law evaluated on rotations built by scipy.

The Riccati and Lyapunov rows are true external checks — scipy solves both equations by a
different method than multicalc does. The attitude rows re-evaluate the published control law in
numpy, so what they pin is the rotation and sign conventions rather than the formula itself; the
rotations they use come from scipy's own rotation type.
"""

import numpy as np
import scipy.linalg
import scipy.signal
from scipy.spatial.transform import Rotation

import schema

ATTITUDE_GAIN = 6.0
RATE_GAIN = 1.2


def _tol():
    return {"f64": schema.tol(1e-9, 1e-9)}


def _attitude_tol():
    return {"f64": schema.tol(1e-12, 1e-12)}


def _gain(a, b, q, r):
    """The feedback gain that goes with a Riccati solution."""
    p = scipy.linalg.solve_discrete_are(a, b, q, r)
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return p, k


def _quadrotor_hover_system():
    """Position and speed in three directions at hover, the outer loop of a flying machine."""
    dt = 0.02
    a = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
    b = np.vstack([0.5 * dt * dt * np.eye(3), dt * np.eye(3)])
    q = np.diag([4.0, 4.0, 4.0, 1.0, 1.0, 1.0])
    r = 0.5 * np.eye(3)
    return a, b, q, r


def _riccati(out, meta):
    dt = 0.1
    double_integrator = (
        np.array([[1.0, dt], [0.0, 1.0]]),
        np.array([[0.5 * dt * dt], [dt]]),
        np.eye(2),
        np.array([[1.0]]),
        "double integrator",
    )

    # A cart carrying a pole, linearized about the pole standing up, then discretized.
    cart_mass, pole_mass, pole_half_length, gravity = 1.0, 0.1, 0.5, 9.81
    continuous_state = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, pole_mass * gravity / cart_mass, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (cart_mass + pole_mass) * gravity / (cart_mass * pole_half_length), 0.0],
        ]
    )
    continuous_input = np.array(
        [[0.0], [1.0 / cart_mass], [0.0], [-1.0 / (cart_mass * pole_half_length)]]
    )
    cart_dt = 0.02
    a_cart, b_cart, *_ = scipy.signal.cont2discrete(
        (continuous_state, continuous_input, np.eye(4), np.zeros((4, 1))), cart_dt, method="zoh"
    )
    cart_pole = (
        a_cart,
        b_cart,
        np.diag([10.0, 1.0, 10.0, 1.0]),
        np.array([[0.1]]),
        "cart and pole",
    )

    a_hover, b_hover, q_hover, r_hover = _quadrotor_hover_system()
    hover = (a_hover, b_hover, q_hover, r_hover, "quadrotor hover")

    rng = np.random.default_rng(meta["seed"])
    a_rand = rng.uniform(-0.6, 0.6, size=(5, 5))
    b_rand = rng.uniform(-1.0, 1.0, size=(5, 2))
    m = rng.uniform(-1.0, 1.0, size=(5, 5))
    q_rand = m @ m.T + np.eye(5)
    mixed = (a_rand, b_rand, q_rand, np.eye(2), "mixed 5×2 system")

    cases = {
        "riccati_double_integrator": double_integrator,
        "riccati_cart_pole": cart_pole,
        "riccati_quadrotor_hover": hover,
        "riccati_mixed_5x2": mixed,
    }
    for name, (a, b, q, r, description) in cases.items():
        p, k = _gain(a, b, q, r)
        inputs = {
            "A": schema.matrix(a),
            "B": schema.matrix(b),
            "Q": schema.matrix(q),
            "R": schema.matrix(r),
        }
        expected = {"P": schema.matrix(p), "K": schema.matrix(k)}
        schema.write_fixture(
            out,
            "control",
            name,
            meta,
            _tol(),
            inputs,
            expected,
            equation="P = AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q",
            operations=[f"Riccati solve, {description}"],
        )


def _lyapunov(out, meta):
    rng = np.random.default_rng(meta["seed"] + 1)

    def stable(n, radius=0.9):
        """A random matrix rescaled so repeated application shrinks every direction."""
        m = rng.uniform(-1.0, 1.0, size=(n, n))
        return m * (radius / np.max(np.abs(np.linalg.eigvals(m))))

    def spd(n):
        m = rng.uniform(-1.0, 1.0, size=(n, n))
        return m @ m.T + np.eye(n)

    cases = {
        "lyapunov_stable_3x3": (stable(3), spd(3), "3×3"),
        "lyapunov_stable_5x5": (stable(5), spd(5), "5×5"),
    }

    # The closed loop of the double-integrator design above, which is the certificate the LQR runs.
    dt = 0.1
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.5 * dt * dt], [dt]])
    q = np.eye(2)
    _, k = _gain(a, b, q, np.array([[1.0]]))
    cases["lyapunov_closed_loop_double_integrator"] = (a - b @ k, q, "closed double integrator")

    # The closed loop of the hover design above: the certificate that a flying machine's position
    # loop settles, which is the check that ships beside the controller.
    a_hover, b_hover, q_hover, r_hover = _quadrotor_hover_system()
    _, k_hover = _gain(a_hover, b_hover, q_hover, r_hover)
    cases["lyapunov_closed_loop_quadrotor_hover"] = (
        a_hover - b_hover @ k_hover,
        q_hover,
        "closed quadrotor hover",
    )

    for name, (a_case, q_case, description) in cases.items():
        # scipy solves A·X·Aᵀ − X + Q = 0; multicalc solves AᵀPA − P + Q = 0, so the
        # transpose goes in.
        p = scipy.linalg.solve_discrete_lyapunov(a_case.T, q_case)
        inputs = {"A": schema.matrix(a_case), "Q": schema.matrix(q_case)}
        expected = {"P": schema.matrix(p)}
        schema.write_fixture(
            out,
            "control",
            name,
            meta,
            _tol(),
            inputs,
            expected,
            equation="AᵀPA − P + Q = 0",
            operations=[f"Lyapunov certificate, {description}"],
        )


def _geometric_attitude(out, meta):
    def law(rotation, rate, desired_rotation, desired_rate, desired_rate_change, inertia):
        relative = rotation.T @ desired_rotation
        error = 0.5 * np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        # vee of (relativeᵀ − relative) is the negative of vee of (relative − relativeᵀ).
        error = -error
        carried = relative @ desired_rate
        rate_error = rate - carried
        carried_change = relative @ desired_rate_change
        spin = np.cross(rate, inertia @ rate)
        following = inertia @ (np.cross(rate, carried) - carried_change)
        torque = -ATTITUDE_GAIN * error - RATE_GAIN * rate_error + spin - following
        return error, torque

    cases = {
        "geometric_attitude_general": (
            Rotation.from_rotvec([0.4, -0.3, 0.8]).as_matrix(),
            np.array([0.9, -0.5, 0.35]),
            Rotation.from_rotvec([-0.2, 0.15, 0.05]).as_matrix(),
            np.array([0.1, 0.2, -0.3]),
            np.array([0.05, -0.02, 0.01]),
            np.array([[0.021, 0.002, 0.001], [0.002, 0.019, 0.003], [0.001, 0.003, 0.038]]),
            "tilted and turning",
        ),
        "geometric_attitude_near_target": (
            Rotation.from_rotvec([0.01, -0.02, 0.005]).as_matrix(),
            np.array([0.02, 0.01, -0.005]),
            np.eye(3),
            np.zeros(3),
            np.zeros(3),
            np.diag([0.02, 0.02, 0.04]),
            "close to the target",
        ),
    }
    for name, (r, w, r_d, w_d, w_d_dot, inertia, description) in cases.items():
        error, torque = law(r, w, r_d, w_d, w_d_dot, inertia)
        inputs = {
            "R": schema.matrix(r),
            "omega": schema.vector(w),
            "R_desired": schema.matrix(r_d),
            "omega_desired": schema.vector(w_d),
            "omega_desired_derivative": schema.vector(w_d_dot),
            "inertia": schema.matrix(inertia),
            "attitude_gain": schema.scalar(ATTITUDE_GAIN),
            "rate_gain": schema.scalar(RATE_GAIN),
        }
        expected = {"attitude_error": schema.vector(error), "torque": schema.vector(torque)}
        schema.write_fixture(
            out,
            "control",
            name,
            meta,
            _attitude_tol(),
            inputs,
            expected,
            equation="τ = −k_R·e_R − k_ω·e_ω + ω×(Jω) − J(ω×RᵀR_d·ω_d − RᵀR_d·ω̇_d)",
            operations=[f"Geometric attitude, {description}"],
        )


def run(out, seed):
    meta = schema.metadata(
        "control",
        seed,
        "fixed textbook systems plus entries uniform in [-1, 1] from a generator-local stream; "
        "stable matrices rescaled to a largest growth factor of 0.9; SPD costs are M·Mᵀ + I",
        libraries=("numpy", "scipy"),
        reference="SciPy {scipy}",
    )
    _riccati(out, meta)
    _lyapunov(out, meta)
    _geometric_attitude(out, meta)
