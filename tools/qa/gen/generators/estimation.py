"""Linear Kalman filter goldens from filterpy."""

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

import schema


def _tol():
    # A filter run compounds per-step error; start here and tighten once the suite is green.
    return {"f64/host": schema.tol(1e-10, 1e-9), "f32/host": schema.tol(1e-3, 1e-3)}


def _run_filter(kf, measurements, controls=None):
    """Steps the filter over the measurement sequence, returning its final quantities."""
    for index, z in enumerate(measurements):
        if controls is None:
            kf.predict()
        else:
            kf.predict(u=controls[index].reshape(-1, 1))
        kf.update(z.reshape(-1, 1))
    return kf


def _expected(kf):
    return {
        "state": schema.vector(kf.x.flatten()),
        "covariance": schema.matrix(kf.P),
        "innovation": schema.vector(np.atleast_1d(kf.y).flatten()),
        "innovation_covariance": schema.matrix(kf.S),
    }


def _constant_velocity_one_dimensional(out, rng, meta):
    """State [position, velocity] over a 1 s step; position is measured."""
    dt, steps = 1.0, 8
    f = np.array([[1.0, dt], [0.0, 1.0]])
    h = np.array([[1.0, 0.0]])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
    r = np.array([[0.5]])
    x0 = np.array([0.0, 1.0])
    p0 = np.eye(2)

    truth = np.arange(1, steps + 1) * dt
    zs = (truth + rng.normal(0.0, 0.5, size=steps)).reshape(steps, 1)

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = x0.reshape(2, 1)
    kf.P = p0.copy()
    kf.F, kf.H, kf.Q, kf.R = f, h, q, r
    _run_filter(kf, zs)

    inputs = {
        "kind": schema.string("kalman_filter"),
        "case": schema.string("constant_velocity_one_dimensional"),
        "state_transition": schema.matrix(f),
        "measurement_model": schema.matrix(h),
        "process_noise": schema.matrix(q),
        "measurement_noise": schema.matrix(r),
        "initial_state": schema.vector(x0),
        "initial_covariance": schema.matrix(p0),
        "measurements": schema.matrix(zs),
    }
    schema.write_fixture(
        out, "estimation", "kalman_filter_constant_velocity_one_dimensional",
        meta, _tol(), inputs, _expected(kf),
    )


def _constant_velocity_two_dimensional(out, rng, meta):
    """State [x, vx, y, vy]; x and y are measured. Process noise from Q_discrete_white_noise."""
    dt, steps = 0.5, 10
    block = np.array([[1.0, dt], [0.0, 1.0]])
    f = np.zeros((4, 4))
    f[:2, :2] = block
    f[2:, 2:] = block
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.02, block_size=2)
    r = np.diag([0.4, 0.6])
    x0 = np.array([0.0, 1.0, 0.0, -0.5])
    p0 = np.diag([1.0, 0.5, 1.0, 0.5])

    times = np.arange(1, steps + 1) * dt
    truth = np.column_stack((times, -0.5 * times))
    zs = truth + rng.normal(0.0, 0.5, size=(steps, 2))

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = x0.reshape(4, 1)
    kf.P = p0.copy()
    kf.F, kf.H, kf.Q, kf.R = f, h, q, r
    _run_filter(kf, zs)

    inputs = {
        "kind": schema.string("kalman_filter"),
        "case": schema.string("constant_velocity_two_dimensional"),
        "state_transition": schema.matrix(f),
        "measurement_model": schema.matrix(h),
        "process_noise": schema.matrix(q),
        "measurement_noise": schema.matrix(r),
        "initial_state": schema.vector(x0),
        "initial_covariance": schema.matrix(p0),
        "measurements": schema.matrix(zs),
    }
    schema.write_fixture(
        out, "estimation", "kalman_filter_constant_velocity_two_dimensional",
        meta, _tol(), inputs, _expected(kf),
    )


def _with_control_input(out, rng, meta):
    """A driven constant-velocity tracker: acceleration enters through the control model."""
    dt, steps = 1.0, 8
    f = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.5 * dt * dt], [dt]])
    h = np.array([[1.0, 0.0]])
    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
    r = np.array([[0.5]])
    x0 = np.array([0.0, 0.0])
    p0 = np.eye(2)

    us = rng.uniform(-1.0, 1.0, size=(steps, 1))
    truth = np.cumsum(np.cumsum(us.flatten()) * dt) * dt
    zs = (truth + rng.normal(0.0, 0.5, size=steps)).reshape(steps, 1)

    kf = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
    kf.x = x0.reshape(2, 1)
    kf.P = p0.copy()
    kf.F, kf.H, kf.Q, kf.R, kf.B = f, h, q, r, b
    _run_filter(kf, zs, controls=us)

    inputs = {
        "kind": schema.string("kalman_filter_with_control"),
        "case": schema.string("with_control_input"),
        "state_transition": schema.matrix(f),
        "measurement_model": schema.matrix(h),
        "process_noise": schema.matrix(q),
        "measurement_noise": schema.matrix(r),
        "initial_state": schema.vector(x0),
        "initial_covariance": schema.matrix(p0),
        "measurements": schema.matrix(zs),
        "control_model": schema.matrix(b),
        "control_inputs": schema.matrix(us),
    }
    schema.write_fixture(
        out, "estimation", "kalman_filter_with_control_input",
        meta, _tol(), inputs, _expected(kf),
    )


def run(out, rng, seed):
    meta = schema.metadata(
        "estimation", seed,
        "measurements are a constant-velocity truth track plus N(0, 0.5) noise; "
        "controls uniform in [-1, 1]",
        libraries=("numpy", "filterpy"),
    )
    _constant_velocity_one_dimensional(out, rng, meta)
    _constant_velocity_two_dimensional(out, rng, meta)
    _with_control_input(out, rng, meta)
