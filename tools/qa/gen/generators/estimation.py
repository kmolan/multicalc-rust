"""Linear Kalman filter goldens from filterpy."""

import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter

import schema


def _tol():
    # A filter run compounds per-step error; start here and tighten once the suite is green.
    return {"f64": schema.tol(1e-10, 1e-9), "f32": schema.tol(1e-3, 1e-3)}


def _unscented_tol(default_spread):
    # At the shipped spread of alpha = 1e-3 the middle point carries a weight near -10^6, so the
    # covariance is a difference of large nearly-equal numbers and a few digits go with it. A wider
    # spread keeps every weight ordinary and holds the same bound the other filters do.
    if default_spread:
        return {"f64": schema.tol(1e-8, 1e-7), "f32": schema.tol(1e-2, 1e-2)}
    return {"f64": schema.tol(1e-10, 1e-9), "f32": schema.tol(1e-3, 1e-3)}


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
        equation="F = [[1, 1], [0, 1]], H = [1, 0]",
        operations=["Linear Kalman filter, 8 steps, state 2 / measurement 1"],
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
        equation="F = blkdiag([[1, 0.5], [0, 1]], [[1, 0.5], [0, 1]]), "
                 "H = [[1, 0, 0, 0], [0, 0, 1, 0]]",
        operations=["Linear Kalman filter, 10 steps, state 4 / measurement 2"],
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
        equation="x⁻ = [[1, 1], [0, 1]]·x + [[0.5], [1]]·u, H = [1, 0]",
        operations=["Linear Kalman filter with control, 8 steps, control 1"],
    )


def _coordinated_turn_step(state, timestep):
    """One tick along the turning arc. Mirrors CoordinatedTurn in
    tools/testkit/src/problems.rs and CoordinatedTurnModel in
    demos/src/sim/kalman_filter_models.rs; the three must stay in step."""
    x, y, heading, speed, turn_rate = state
    next_heading = heading + turn_rate * timestep
    radius = speed / turn_rate
    return np.array([
        x + radius * (np.sin(next_heading) - np.sin(heading)),
        y + radius * (np.cos(heading) - np.cos(next_heading)),
        next_heading,
        speed,
        turn_rate,
    ])


def _landmark_range_and_bearing(out, rng, meta):
    """A stationary pose observed by range and bearing to one known landmark: nonlinear h, linear f."""
    from filterpy.kalman import ExtendedKalmanFilter

    landmark = np.array([3.0, 4.0])
    steps = 8
    q = np.eye(3) * 0.001
    r = np.diag([0.1, 0.05])
    x0 = np.array([0.2, -0.1, 0.05])
    p0 = np.eye(3) * 0.5
    truth = np.array([0.0, 0.0, 0.0])

    def hx(x):
        d = landmark - x[:2, 0]
        return np.array([[np.hypot(d[0], d[1])], [np.arctan2(d[1], d[0]) - x[2, 0]]])

    def h_jacobian(x):
        d = landmark - x[:2, 0]
        squared = d @ d
        distance = np.sqrt(squared)
        return np.array([
            [-d[0] / distance, -d[1] / distance, 0.0],
            [d[1] / squared, -d[0] / squared, -1.0],
        ])

    d = landmark - truth[:2]
    exact = np.array([np.hypot(d[0], d[1]), np.arctan2(d[1], d[0]) - truth[2]])
    zs = exact + rng.normal(0.0, [0.1, 0.05], size=(steps, 2))

    kf = ExtendedKalmanFilter(dim_x=3, dim_z=2)
    kf.x = x0.reshape(3, 1)
    kf.P = p0.copy()
    kf.Q, kf.R = q, r
    kf.F = np.eye(3)          # stationary pose; the Rust side's model is the identity too
    for z in zs:
        kf.predict()
        kf.update(z.reshape(2, 1), h_jacobian, hx)

    inputs = {
        "kind": schema.string("extended_kalman_filter"),
        "case": schema.string("landmark_range_and_bearing"),
        "landmark": schema.vector(landmark),
        "process_noise": schema.matrix(q),
        "measurement_noise": schema.matrix(r),
        "initial_state": schema.vector(x0),
        "initial_covariance": schema.matrix(p0),
        "measurements": schema.matrix(zs),
    }
    schema.write_fixture(
        out, "estimation", "extended_kalman_filter_landmark_range_and_bearing",
        meta, _tol(), inputs, _expected(kf),
        equation="h = [√((3−x)²+(4−y)²), atan2(4−y, 3−x)−θ], F = I",
        operations=["Extended Kalman filter, 8 steps, state 3 / measurement 2"],
    )


def _coordinated_turn_fusion(out, rng, meta):
    """A turning ground vehicle tracked from position fixes: nonlinear f, linear h.

    The process model is the showcase's coordinated turn over
    [x, y, heading, speed, turn_rate]; the measurement is a position fix. filterpy
    propagates with the matrix F, so the state is advanced through the nonlinear
    model by hand and F is set to its Jacobian at each step.
    """
    from filterpy.kalman import ExtendedKalmanFilter

    timestep, steps = 0.1, 8
    process_noise = np.diag([1e-7, 1e-7, 1e-7, 4e-4, 4e-4])
    measurement_noise = np.diag([0.09, 0.09])
    initial_state = np.array([0.2, -0.1, 0.05, 0.9, 0.25])
    initial_covariance = np.diag([0.5, 0.5, 0.2, 0.2, 0.2])
    true_initial_state = np.array([0.0, 0.0, 0.0, 1.0, 0.3])

    def advance_state(state):
        return _coordinated_turn_step(state, timestep)

    def transition_jacobian(state):
        """How each output of advance_state changes with each input."""
        _, _, heading, speed, turn_rate = state
        next_heading = heading + turn_rate * timestep
        radius = speed / turn_rate
        sine_difference = np.sin(next_heading) - np.sin(heading)
        cosine_difference = np.cos(heading) - np.cos(next_heading)
        jacobian = np.eye(5)
        jacobian[0, 2] = radius * (np.cos(next_heading) - np.cos(heading))
        jacobian[0, 3] = sine_difference / turn_rate
        jacobian[0, 4] = (-speed * sine_difference / turn_rate**2
                          + speed * np.cos(next_heading) * timestep / turn_rate)
        jacobian[1, 2] = radius * sine_difference
        jacobian[1, 3] = cosine_difference / turn_rate
        jacobian[1, 4] = (-speed * cosine_difference / turn_rate**2
                          + speed * np.sin(next_heading) * timestep / turn_rate)
        jacobian[2, 4] = timestep
        return jacobian

    def measurement_function(state):
        """The sensor sees position only."""
        return np.array([[state[0, 0]], [state[1, 0]]])

    def measurement_jacobian(_state):
        return np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0, 0.0]])

    # Measurements: the true track's position, plus position-sensor noise.
    true_state = true_initial_state.copy()
    measurements = np.zeros((steps, 2))
    for step in range(steps):
        true_state = advance_state(true_state)
        measurements[step] = true_state[:2] + rng.normal(0.0, 0.3, size=2)

    estimator = ExtendedKalmanFilter(dim_x=5, dim_z=2)
    estimator.x = initial_state.reshape(5, 1)
    estimator.P = initial_covariance.copy()
    estimator.Q, estimator.R = process_noise, measurement_noise
    for measurement in measurements:
        # Advance through the nonlinear model, then hand filterpy the matching
        # linearization so its covariance predict uses the same transition.
        estimator.F = transition_jacobian(estimator.x.flatten())
        predicted = advance_state(estimator.x.flatten())
        estimator.P = estimator.F @ estimator.P @ estimator.F.T + estimator.Q
        estimator.x = predicted.reshape(5, 1)
        estimator.update(measurement.reshape(2, 1),
                         measurement_jacobian, measurement_function)

    inputs = {
        "kind": schema.string("extended_kalman_filter"),
        "case": schema.string("coordinated_turn_fusion"),
        "timestep": schema.scalar(timestep),
        "process_noise": schema.matrix(process_noise),
        "measurement_noise": schema.matrix(measurement_noise),
        "initial_state": schema.vector(initial_state),
        "initial_covariance": schema.matrix(initial_covariance),
        "measurements": schema.matrix(measurements),
    }
    schema.write_fixture(
        out, "estimation", "extended_kalman_filter_coordinated_turn_fusion",
        meta, _tol(), inputs, _expected(estimator),
        equation="f = coordinated turn on [x, y, θ, v, ω], h = [x, y]",
        operations=["Extended Kalman filter, 8 steps, state 5 / measurement 2"],
    )


def _unscented_coordinated_turn_fusion(out, rng, meta):
    """The turning-vehicle track again, sampled at a spread of points rather than linearized."""
    from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

    timestep, steps = 0.1, 8
    alpha, beta, kappa = 1e-3, 2.0, 0.0
    process_noise = np.diag([1e-7, 1e-7, 1e-7, 4e-4, 4e-4])
    measurement_noise = np.diag([0.09, 0.09])
    initial_state = np.array([0.2, -0.1, 0.05, 0.9, 0.25])
    initial_covariance = np.diag([0.5, 0.5, 0.2, 0.2, 0.2])
    true_initial_state = np.array([0.0, 0.0, 0.0, 1.0, 0.3])

    true_state = true_initial_state.copy()
    measurements = np.zeros((steps, 2))
    for step in range(steps):
        true_state = _coordinated_turn_step(true_state, timestep)
        measurements[step] = true_state[:2] + rng.normal(0.0, 0.3, size=2)

    points = MerweScaledSigmaPoints(n=5, alpha=alpha, beta=beta, kappa=kappa)
    estimator = UnscentedKalmanFilter(
        dim_x=5, dim_z=2, dt=timestep,
        fx=lambda state, _dt: _coordinated_turn_step(state, timestep),
        hx=lambda state: state[:2].copy(),
        points=points,
    )
    estimator.x = initial_state.copy()
    estimator.P = initial_covariance.copy()
    estimator.Q, estimator.R = process_noise, measurement_noise
    for measurement in measurements:
        estimator.predict()
        estimator.update(measurement)

    inputs = {
        "kind": schema.string("unscented_kalman_filter"),
        "case": schema.string("coordinated_turn_fusion"),
        "timestep": schema.scalar(timestep),
        "alpha": schema.scalar(alpha),
        "beta": schema.scalar(beta),
        "kappa": schema.scalar(kappa),
        "process_noise": schema.matrix(process_noise),
        "measurement_noise": schema.matrix(measurement_noise),
        "initial_state": schema.vector(initial_state),
        "initial_covariance": schema.matrix(initial_covariance),
        "measurements": schema.matrix(measurements),
    }
    schema.write_fixture(
        out, "estimation", "unscented_kalman_filter_coordinated_turn_fusion",
        meta, _unscented_tol(True), inputs, _expected(estimator),
        equation="f = coordinated turn on [x, y, θ, v, ω], h = [x, y], α = 1e-3",
        operations=["Unscented Kalman filter, 8 steps, state 5 / measurement 2, spread 1e-3"],
    )


def _unscented_landmark_range_and_bearing(out, rng, meta):
    """The landmark sighting again, with the points spread wide enough to sample the curve."""
    from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

    landmark = np.array([3.0, 4.0])
    steps = 8
    alpha, beta, kappa = 0.3, 2.0, 0.0
    process_noise = np.eye(3) * 0.001
    measurement_noise = np.diag([0.1, 0.05])
    initial_state = np.array([0.2, -0.1, 0.05])
    initial_covariance = np.eye(3) * 0.5
    truth = np.array([0.0, 0.0, 0.0])

    def measure(state):
        offset = landmark - state[:2]
        return np.array([np.hypot(offset[0], offset[1]),
                         np.arctan2(offset[1], offset[0]) - state[2]])

    exact = measure(truth)
    zs = exact + rng.normal(0.0, [0.1, 0.05], size=(steps, 2))

    points = MerweScaledSigmaPoints(n=3, alpha=alpha, beta=beta, kappa=kappa)
    estimator = UnscentedKalmanFilter(
        dim_x=3, dim_z=2, dt=1.0,
        fx=lambda state, _dt: state.copy(),
        hx=measure,
        points=points,
    )
    estimator.x = initial_state.copy()
    estimator.P = initial_covariance.copy()
    estimator.Q, estimator.R = process_noise, measurement_noise
    for z in zs:
        estimator.predict()
        estimator.update(z)

    inputs = {
        "kind": schema.string("unscented_kalman_filter"),
        "case": schema.string("landmark_range_and_bearing"),
        "landmark": schema.vector(landmark),
        "alpha": schema.scalar(alpha),
        "beta": schema.scalar(beta),
        "kappa": schema.scalar(kappa),
        "process_noise": schema.matrix(process_noise),
        "measurement_noise": schema.matrix(measurement_noise),
        "initial_state": schema.vector(initial_state),
        "initial_covariance": schema.matrix(initial_covariance),
        "measurements": schema.matrix(zs),
    }
    schema.write_fixture(
        out, "estimation", "unscented_kalman_filter_landmark_range_and_bearing",
        meta, _unscented_tol(False), inputs, _expected(estimator),
        equation="h = [√((3−x)²+(4−y)²), atan2(4−y, 3−x)−θ], f = I, α = 0.3",
        operations=["Unscented Kalman filter, 8 steps, state 3 / measurement 2, spread 0.3"],
    )


def run(out, rng, seed):
    meta = schema.metadata(
        "estimation", seed,
        "measurements are a constant-velocity truth track plus N(0, 0.5) noise; "
        "controls uniform in [-1, 1]",
        libraries=("numpy", "filterpy"),
        reference="FilterPy {filterpy}",
    )
    _constant_velocity_one_dimensional(out, rng, meta)
    _constant_velocity_two_dimensional(out, rng, meta)
    _with_control_input(out, rng, meta)
    _landmark_range_and_bearing(out, rng, meta)
    _coordinated_turn_fusion(out, rng, meta)
    _unscented_coordinated_turn_fusion(out, rng, meta)
    _unscented_landmark_range_and_bearing(out, rng, meta)

    # These two go last so they cannot move the random stream the cases above draw from. Their
    # goldens come from somewhere else, so they carry their own provenance rather than inheriting
    # FilterPy's.
    error_state_meta = schema.metadata(
        "estimation", seed,
        "IMU readings are a constant-turn, slow-sway truth plus noise at the densities the filter "
        "is given; position fixes and heading aids are the truth plus their own noise",
        libraries=("numpy", "scipy"),
        reference="in-house numpy transcription of Sola's error-state equations",
    )
    _error_state_kalman_filter_imu_trajectory(out, rng, error_state_meta)

    triad_meta = schema.metadata(
        "estimation", seed,
        "two exactly-rotated direction pairs, noiseless",
        libraries=("numpy", "scipy"),
        reference="SciPy {scipy}",
    )
    _triad_attitude_from_two_directions(out, rng, triad_meta)


def _error_state_tol():
    # A 200-step run with 30 corrections compounds, so this is looser than a single-operation
    # bound and looser than the other filter rows, which run 8 to 10 steps.
    return {"f64": schema.tol(1e-9, 1e-8), "f32": schema.tol(1e-2, 1e-2)}


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _scalar_first(rotation):
    """Scipy's quaternion, reordered scalar-first and turned so the scalar part is not negative."""
    x, y, z, w = rotation.as_quat()
    q = np.array([w, x, y, z])
    return -q if q[0] < 0.0 else q


def _error_state_kalman_filter_imu_trajectory(out, rng, meta):
    """A body turning and swaying for two seconds, tracked from its own IMU plus a position fix
    and a heading aid.

    Reference implementation of Sola's error-state equations, written here rather than taken from
    a library -- FilterPy has no error-state filter. It pins the crate against an independent
    transcription of the same paper, not against a trusted third party. If both transcriptions
    read the paper the same wrong way this fixture still agrees, which is why the crate also
    checks its closed-form transition against an autodiff Jacobian of its own propagation.
    """
    from scipy.spatial.transform import Rotation

    dt, steps = 0.01, 200
    gravity = np.array([0.0, 0.0, -9.81])

    turn_rate = np.array([0.3, -0.2, 0.5])
    gyroscope_bias = np.array([0.02, -0.015, 0.01])
    accelerometer_bias = np.array([0.15, -0.10, 0.05])

    gyroscope_noise_density = 0.02
    accelerometer_noise_density = 0.05
    gyroscope_bias_random_walk = 1e-4
    accelerometer_bias_random_walk = 1e-3

    position_fix_period, heading_aid_period = 10, 20
    position_fix_sigma, heading_aid_sigma = 0.03, np.deg2rad(2.0)
    position_fix_noise = np.eye(3) * position_fix_sigma**2
    heading_aid_noise = np.array([[heading_aid_sigma**2]])

    # The truth: a constant turn with a slow sway along the world x axis, and the readings a real
    # IMU would have produced from it.
    truth_position = np.zeros(3)
    truth_velocity = np.zeros(3)
    truth_orientation = Rotation.identity()
    gyroscope_readings = np.zeros((steps, 3))
    accelerometer_readings = np.zeros((steps, 3))
    truth_positions = np.zeros((steps, 3))
    truth_orientations = []
    for step in range(steps):
        time = step * dt
        world_push = np.array([0.5 * np.sin(2.0 * np.pi * 0.5 * time), 0.0, 0.0])
        proper_push = truth_orientation.inv().apply(world_push - gravity)

        gyroscope_readings[step] = (
            turn_rate + gyroscope_bias + rng.normal(0.0, gyroscope_noise_density, 3)
        )
        accelerometer_readings[step] = (
            proper_push + accelerometer_bias + rng.normal(0.0, accelerometer_noise_density, 3)
        )

        truth_position = truth_position + truth_velocity * dt + 0.5 * world_push * dt * dt
        truth_velocity = truth_velocity + world_push * dt
        truth_orientation = truth_orientation * Rotation.from_rotvec(turn_rate * dt)
        truth_positions[step] = truth_position
        truth_orientations.append(truth_orientation)

    position_fixes = np.array(
        [
            truth_positions[step] + rng.normal(0.0, position_fix_sigma, 3)
            for step in range(steps)
            if (step + 1) % position_fix_period == 0
        ]
    )
    heading_aids = np.array(
        [
            [
                truth_orientations[step].as_euler("ZYX")[0]
                + rng.normal(0.0, heading_aid_sigma)
            ]
            for step in range(steps)
            if (step + 1) % heading_aid_period == 0
        ]
    )

    # The filter, from a starting guess that is tilted a little and knows nothing of the offsets.
    initial_orientation = Rotation.from_rotvec([0.05, -0.03, 0.02])
    position = np.zeros(3)
    velocity = np.zeros(3)
    orientation = initial_orientation
    gyroscope_offset = np.zeros(3)
    accelerometer_offset = np.zeros(3)
    initial_covariance = np.diag([0.1] * 3 + [0.1] * 3 + [0.05] * 3 + [0.01] * 3 + [0.05] * 3)
    covariance = initial_covariance.copy()

    def fold_in(jacobian, residual, noise):
        """One correction: the gain, the Joseph covariance, the injection, and the reset."""
        nonlocal position, velocity, orientation, gyroscope_offset, accelerometer_offset
        nonlocal covariance

        innovation_covariance = jacobian @ covariance @ jacobian.T + noise
        gain = covariance @ jacobian.T @ np.linalg.inv(innovation_covariance)
        error = gain @ residual

        transfer = np.eye(15) - gain @ jacobian
        covariance = transfer @ covariance @ transfer.T + gain @ noise @ gain.T

        position = position + error[0:3]
        velocity = velocity + error[3:6]
        orientation = orientation * Rotation.from_rotvec(error[6:9])
        gyroscope_offset = gyroscope_offset + error[9:12]
        accelerometer_offset = accelerometer_offset + error[12:15]

        reset = np.eye(15)
        reset[6:9, 6:9] = np.eye(3) - 0.5 * _skew(error[6:9])
        covariance = reset @ covariance @ reset.T
        covariance = 0.5 * (covariance + covariance.T)

    position_fix_index, heading_aid_index = 0, 0
    for step in range(steps):
        corrected_turn = gyroscope_readings[step] - gyroscope_offset
        corrected_push = accelerometer_readings[step] - accelerometer_offset
        rotation_matrix = orientation.as_matrix()
        world_push = rotation_matrix @ corrected_push + gravity

        transition = np.eye(15)
        transition[0:3, 3:6] = np.eye(3) * dt
        transition[3:6, 6:9] = -rotation_matrix @ _skew(corrected_push) * dt
        transition[3:6, 12:15] = -rotation_matrix * dt
        transition[6:9, 6:9] = Rotation.from_rotvec(-corrected_turn * dt).as_matrix()
        transition[6:9, 9:12] = -np.eye(3) * dt

        position = position + velocity * dt + 0.5 * world_push * dt * dt
        velocity = velocity + world_push * dt
        orientation = orientation * Rotation.from_rotvec(corrected_turn * dt)

        covariance = transition @ covariance @ transition.T
        covariance[3:6, 3:6] += np.eye(3) * (accelerometer_noise_density * dt) ** 2
        covariance[6:9, 6:9] += np.eye(3) * (gyroscope_noise_density * dt) ** 2
        covariance[9:12, 9:12] += np.eye(3) * gyroscope_bias_random_walk**2 * dt
        covariance[12:15, 12:15] += np.eye(3) * accelerometer_bias_random_walk**2 * dt
        covariance = 0.5 * (covariance + covariance.T)

        if (step + 1) % position_fix_period == 0:
            jacobian = np.zeros((3, 15))
            jacobian[0:3, 0:3] = np.eye(3)
            residual = position_fixes[position_fix_index] - position
            position_fix_index += 1
            fold_in(jacobian, residual, position_fix_noise)

        if (step + 1) % heading_aid_period == 0:
            # The heading's sensitivity to a small turn, by a central difference on the
            # orientation itself -- the crate takes the same derivative by autodiff. The capital
            # letters in "ZYX" matter: they ask for turns about the body's own axes as it goes,
            # which is what the crate's to_euler_zyx reports. Lowercase would fix the axes in the
            # world instead and give a different first angle.
            jacobian = np.zeros((1, 15))
            nudge = 1e-7
            for axis in range(3):
                step_vector = np.zeros(3)
                step_vector[axis] = nudge
                ahead = (orientation * Rotation.from_rotvec(step_vector)).as_euler("ZYX")[0]
                behind = (orientation * Rotation.from_rotvec(-step_vector)).as_euler("ZYX")[0]
                jacobian[0, 6 + axis] = _wrap_to_pi(ahead - behind) / (2.0 * nudge)
            predicted = orientation.as_euler("ZYX")[0]
            residual = np.array([_wrap_to_pi(heading_aids[heading_aid_index][0] - predicted)])
            heading_aid_index += 1
            fold_in(jacobian, residual, heading_aid_noise)

    inputs = {
        "kind": schema.string("error_state_kalman_filter"),
        "case": schema.string("imu_trajectory"),
        "timestep": schema.scalar(dt),
        "gravity": schema.vector(gravity),
        "initial_position": schema.vector(np.zeros(3)),
        "initial_velocity": schema.vector(np.zeros(3)),
        "initial_orientation": schema.vector(_scalar_first(initial_orientation)),
        "initial_gyroscope_bias": schema.vector(np.zeros(3)),
        "initial_accelerometer_bias": schema.vector(np.zeros(3)),
        "initial_covariance": schema.matrix(initial_covariance),
        "gyroscope_noise_density": schema.scalar(gyroscope_noise_density),
        "accelerometer_noise_density": schema.scalar(accelerometer_noise_density),
        "gyroscope_bias_random_walk": schema.scalar(gyroscope_bias_random_walk),
        "accelerometer_bias_random_walk": schema.scalar(accelerometer_bias_random_walk),
        "gyroscope_readings": schema.matrix(gyroscope_readings),
        "accelerometer_readings": schema.matrix(accelerometer_readings),
        "position_fix_period": schema.integer(position_fix_period),
        "position_fixes": schema.matrix(position_fixes),
        "position_fix_noise": schema.matrix(position_fix_noise),
        "heading_aid_period": schema.integer(heading_aid_period),
        "heading_aids": schema.matrix(heading_aids),
        "heading_aid_noise": schema.matrix(heading_aid_noise),
    }
    expected = {
        "position": schema.vector(position),
        "velocity": schema.vector(velocity),
        "orientation": schema.vector(_scalar_first(orientation)),
        "gyroscope_bias": schema.vector(gyroscope_offset),
        "accelerometer_bias": schema.vector(accelerometer_offset),
        "covariance": schema.matrix(covariance),
    }
    schema.write_fixture(
        out, "estimation", "error_state_kalman_filter_imu_trajectory",
        meta, _error_state_tol(), inputs, expected,
        equation="p- = p + v*dt + 0.5*(R*a + g)*dt^2, q- = q x exp(w*dt), reset G = I - 0.5*[dtheta]x",
        operations=["Error-state Kalman filter, 200 IMU steps + 20 position fixes + 10 heading "
                    "aids, error state 15 / measurement 3 and 1"],
    )


def _wrap_to_pi(angle):
    return angle - 2.0 * np.pi * np.round(angle / (2.0 * np.pi))


def _triad_attitude_from_two_directions(out, rng, meta):
    """An orientation read straight off two directions a still body can see.

    With two noiseless, consistent direction pairs the orientation is unique, so the
    two-direction construction and the least-squares fit scipy solves give the same answer
    exactly. Under noise they would differ -- this fixture is deliberately noiseless.
    """
    from scipy.spatial.transform import Rotation

    truth = Rotation.from_rotvec([0.4, -0.25, 0.9])
    down_in_world = np.array([0.0, 0.0, -1.0])
    north_in_world = np.array([1.0, 0.0, 0.0])
    down_in_body = truth.inv().apply(down_in_world)
    north_in_body = truth.inv().apply(north_in_world)

    fitted = Rotation.align_vectors(
        np.vstack([down_in_world, north_in_world]),
        np.vstack([down_in_body, north_in_body]),
    )[0]

    inputs = {
        "kind": schema.string("triad"),
        "case": schema.string("attitude_from_two_directions"),
        "primary_observed": schema.vector(down_in_body),
        "secondary_observed": schema.vector(north_in_body),
        "primary_reference": schema.vector(down_in_world),
        "secondary_reference": schema.vector(north_in_world),
    }
    expected = {
        "orientation": schema.vector(_scalar_first(fitted)),
        "rotation_matrix": schema.matrix(fitted.as_matrix()),
    }
    schema.write_fixture(
        out, "estimation", "triad_attitude_from_two_directions",
        meta, {"f64": schema.tol(1e-13, 1e-12), "f32": schema.tol(1e-5, 1e-5)}, inputs, expected,
        equation="R = [r1 r2 r3] * [o1 o2 o3]^T from two direction pairs",
        operations=["Attitude from two direction pairs (down + north), noiseless"],
    )
