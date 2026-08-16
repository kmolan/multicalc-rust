"""Minimum-snap trajectory goldens: an independent constrained solve in numpy.

multicalc works out which values are already known, substitutes them out, and
solves only for the ones left. This generator does the opposite: it keeps every
coefficient as an unknown, writes every requirement as an explicit equation, and
solves the whole constrained system with its multipliers in one go.

    minimise   cᵀHc      subject to   C·c = b
    solving    [[2H, Cᵀ], [C, 0]] · [c; λ] = [0; b]

The two routes share only the definition of the problem, so matching
coefficients mean two genuinely different sets of equations agree.

Each segment runs on its own 0-to-1 clock, matching how multicalc stores them.
Requirements written in real time therefore divide by the segment's duration
once per order, which is where the durations enter.

The motion profiles below follow the same principle. multicalc works out the
phase lengths by case analysis — algebra that says outright which limits a move
reaches. This generator never assumes the answer is fastest: it hands the three
free lengths to a constrained minimizer, asks for the smallest total time that
still covers the distance without breaking a limit, and takes whatever comes
back. The states are then produced by integrating the resulting jerk schedule
numerically, not by evaluating a formula. So a match means an algebraic answer
and a searched one agree.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize

import schema

COEFFICIENTS_PER_SEGMENT = 8
DIMENSION = 3
PROFILE_PHASE_COUNT = 7

# The full system is larger and worse conditioned than the reduced one multicalc
# solves, so the coefficients agree to rather less than the states they produce.
COEFFICIENT_TOL = (1e-7, 1e-7)
STATE_TOL = (1e-9, 1e-9)

# The phase lengths come back from a numerical optimizer and the states from an
# integrator, so the two families of value are compared at different scales.
PROFILE_DURATION_TOL = (1e-7, 1e-7)
PROFILE_STATE_TOL = (1e-9, 1e-9)


def _snap_cost():
    """The 8×8 taking a segment's coefficients to its total snap, on its own clock."""
    cost = np.zeros((COEFFICIENTS_PER_SEGMENT, COEFFICIENTS_PER_SEGMENT))
    for j in range(4, COEFFICIENTS_PER_SEGMENT):
        for k in range(4, COEFFICIENTS_PER_SEGMENT):
            ways_j = j * (j - 1) * (j - 2) * (j - 3)
            ways_k = k * (k - 1) * (k - 2) * (k - 3)
            cost[j, k] = ways_j * ways_k / (j + k - 7)
    return cost


def _derivative_row(order, at_end, duration):
    """How each coefficient contributes to one order at one end of a segment, in real time."""
    row = np.zeros(COEFFICIENTS_PER_SEGMENT)
    for power in range(COEFFICIENTS_PER_SEGMENT):
        if power < order:
            continue
        ways = 1.0
        for step in range(order):
            ways *= power - step
        # On the piece's own clock the position is 0 at the start and 1 at the end.
        clock = 1.0 if at_end else (1.0 if power == order else 0.0)
        if not at_end and power != order:
            clock = 0.0
        row[power] = ways * clock / duration**order
    return row


def _plan(waypoints, durations, start, end):
    """Every coefficient of every segment, from the full constrained system."""
    segments = len(durations)
    unknowns = segments * COEFFICIENTS_PER_SEGMENT
    cost = _snap_cost()

    objective = np.zeros((unknowns, unknowns))
    for segment, duration in enumerate(durations):
        base = segment * COEFFICIENTS_PER_SEGMENT
        block = slice(base, base + COEFFICIENTS_PER_SEGMENT)
        objective[block, block] = cost / duration**7

    coefficients = np.zeros((segments, DIMENSION, COEFFICIENTS_PER_SEGMENT))
    for axis in range(DIMENSION):
        rows = []
        values = []

        # Where the path goes: both ends of every segment.
        for segment, duration in enumerate(durations):
            base = segment * COEFFICIENTS_PER_SEGMENT
            for at_end in (False, True):
                row = np.zeros(unknowns)
                row[base : base + COEFFICIENTS_PER_SEGMENT] = _derivative_row(0, at_end, duration)
                rows.append(row)
                values.append(waypoints[segment + 1 if at_end else segment][axis])

        # How it is moving where it starts and finishes.
        for order, boundary in ((1, start), (2, start), (3, start)):
            row = np.zeros(unknowns)
            row[0:COEFFICIENTS_PER_SEGMENT] = _derivative_row(order, False, durations[0])
            rows.append(row)
            values.append(boundary[order - 1][axis])
        last = (segments - 1) * COEFFICIENTS_PER_SEGMENT
        for order, boundary in ((1, end), (2, end), (3, end)):
            row = np.zeros(unknowns)
            row[last : last + COEFFICIENTS_PER_SEGMENT] = _derivative_row(
                order, True, durations[-1]
            )
            rows.append(row)
            values.append(boundary[order - 1][axis])

        # Matched across every interior joint, as a difference equal to zero.
        for joint in range(segments - 1):
            before = joint * COEFFICIENTS_PER_SEGMENT
            after = (joint + 1) * COEFFICIENTS_PER_SEGMENT
            for order in (1, 2, 3):
                row = np.zeros(unknowns)
                row[before : before + COEFFICIENTS_PER_SEGMENT] = _derivative_row(
                    order, True, durations[joint]
                )
                row[after : after + COEFFICIENTS_PER_SEGMENT] -= _derivative_row(
                    order, False, durations[joint + 1]
                )
                rows.append(row)
                values.append(0.0)

        constraints = np.array(rows)
        wanted = np.array(values)
        count = constraints.shape[0]

        system = np.zeros((unknowns + count, unknowns + count))
        system[:unknowns, :unknowns] = 2.0 * objective
        system[:unknowns, unknowns:] = constraints.T
        system[unknowns:, :unknowns] = constraints
        right = np.concatenate([np.zeros(unknowns), wanted])

        solved = np.linalg.solve(system, right)[:unknowns]
        for segment in range(segments):
            base = segment * COEFFICIENTS_PER_SEGMENT
            coefficients[segment, axis] = solved[base : base + COEFFICIENTS_PER_SEGMENT]
    return coefficients


def _sample(coefficients, durations, times):
    """Position, velocity and acceleration at each time, in real time."""
    boundaries = np.concatenate([[0.0], np.cumsum(durations)])
    rows = []
    for time in times:
        clamped = min(max(time, 0.0), boundaries[-1])
        segment = int(np.searchsorted(boundaries, clamped, side="right") - 1)
        segment = min(max(segment, 0), len(durations) - 1)
        duration = durations[segment]
        along = (clamped - boundaries[segment]) / duration
        for order in range(3):
            row = []
            for axis in range(DIMENSION):
                total = 0.0
                for power in range(COEFFICIENTS_PER_SEGMENT):
                    if power < order:
                        continue
                    ways = 1.0
                    for step in range(order):
                        ways *= power - step
                    total += coefficients[segment, axis, power] * ways * along ** (power - order)
                row.append(total / duration**order)
            rows.append(row)
    return rows


# --- motion profiles ---


def _profile_distance(jerk_time, hold_time, cruise_time, jerk_limit, acceleration_limit):
    """How far a profile with these phase lengths travels.

    This is the shared definition of the problem, and the only thing the two
    routes have in common."""
    if jerk_limit is None:
        peak_speed = acceleration_limit * hold_time
        return peak_speed * (hold_time + cruise_time)
    peak_acceleration = jerk_limit * jerk_time
    peak_speed = peak_acceleration * (jerk_time + hold_time)
    ramp_time = 2.0 * jerk_time + hold_time
    return peak_speed * (ramp_time + cruise_time)


def _profile_peaks(jerk_time, hold_time, jerk_limit, acceleration_limit):
    """The fastest and hardest the move gets, from its phase lengths."""
    if jerk_limit is None:
        return acceleration_limit * hold_time, acceleration_limit
    peak_acceleration = jerk_limit * jerk_time
    return peak_acceleration * (jerk_time + hold_time), peak_acceleration


def _ramp_lengths(peak_speed, acceleration_limit, jerk_limit):
    """Jerk-phase and constant-acceleration lengths to reach `peak_speed` from rest."""
    if jerk_limit is None:
        return 0.0, peak_speed / acceleration_limit
    if peak_speed * jerk_limit >= acceleration_limit**2:
        jerk_time = acceleration_limit / jerk_limit
        return jerk_time, max(peak_speed / acceleration_limit - jerk_time, 0.0)
    return np.sqrt(peak_speed / jerk_limit), 0.0


def _ramp_distance(peak_speed, acceleration_limit, jerk_limit):
    """How far speeding up to `peak_speed` and back down to rest covers."""
    jerk_time, hold_time = _ramp_lengths(peak_speed, acceleration_limit, jerk_limit)
    return peak_speed * (2.0 * jerk_time + hold_time)


def _solve_profile(distance, speed_limit, acceleration_limit, jerk_limit):
    """The three free phase lengths, as the solution of a minimum-time problem.

    Minimises the total move time subject to covering the distance exactly and
    staying under every limit. Nothing here knows which limits the move reaches;
    that is what the minimizer is asked to work out."""
    trapezoidal = jerk_limit is None

    def total_time(free):
        jerk_time, hold_time, cruise_time = free
        if trapezoidal:
            return 2.0 * hold_time + cruise_time
        return 4.0 * jerk_time + 2.0 * hold_time + cruise_time

    def covers_the_distance(free):
        return _profile_distance(*free, jerk_limit, acceleration_limit) - distance

    def speed_headroom(free):
        peak_speed, _ = _profile_peaks(free[0], free[1], jerk_limit, acceleration_limit)
        return speed_limit - peak_speed

    def acceleration_headroom(free):
        _, peak_acceleration = _profile_peaks(free[0], free[1], jerk_limit, acceleration_limit)
        return acceleration_limit - peak_acceleration

    constraints = [
        {"type": "eq", "fun": covers_the_distance},
        {"type": "ineq", "fun": speed_headroom},
    ]
    if not trapezoidal:
        constraints.append({"type": "ineq", "fun": acceleration_headroom})

    # Pinning the jerk phases to zero length is what makes the trapezoid the
    # same problem without a jerk limit.
    bounds = [(0.0, 0.0) if trapezoidal else (0.0, None), (0.0, None), (0.0, None)]

    # A cruise-heavy start, a ramp-heavy one, and the closed-form answer, so a
    # single bad basin cannot decide the result.
    reachable = min(speed_limit, _closed_form_peak(distance, acceleration_limit, jerk_limit))
    guess_jerk, guess_hold = _ramp_lengths(reachable, acceleration_limit, jerk_limit)
    guess_ramp = 2.0 * guess_jerk + guess_hold
    guess_cruise = max((distance - reachable * guess_ramp) / reachable, 0.0) if reachable else 0.0
    starts = [
        [guess_jerk, guess_hold, guess_cruise],
        [0.0 if trapezoidal else 0.1, 0.1, max(distance / max(speed_limit, 1e-9), 0.1)],
        [0.0 if trapezoidal else 1.0, 1.0, 0.0],
    ]

    best = None
    for start in starts:
        found = minimize(
            total_time,
            np.array(start, dtype=float),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 2000},
        )
        if not found.success:
            continue
        free = [max(value, 0.0) for value in found.x]
        if abs(covers_the_distance(free)) > 1e-9:
            continue
        if speed_headroom(free) < -1e-9:
            continue
        if not trapezoidal and acceleration_headroom(free) < -1e-9:
            continue
        if best is None or total_time(free) < total_time(best) - 1e-12:
            best = free
    assert best is not None, f"no feasible profile for a distance of {distance}"
    return tuple(best)


def _closed_form_peak(distance, acceleration_limit, jerk_limit):
    """A starting guess at the fastest the distance allows, for the minimizer to improve on."""
    if distance <= 0.0:
        return 0.0
    if jerk_limit is None:
        return np.sqrt(distance * acceleration_limit)
    triangular = np.cbrt(distance**2 * jerk_limit / 4.0)
    if triangular * jerk_limit < acceleration_limit**2:
        return triangular
    ratio = acceleration_limit**2 / jerk_limit
    return (-ratio + np.sqrt(ratio**2 + 4.0 * acceleration_limit * distance)) / 2.0


def _cross_check(free, distance, speed_limit, acceleration_limit, jerk_limit):
    """Guards the minimizer against stopping early or in a feasible-but-slow corner.

    Where the move never cruises, the peak speed is pinned by the distance alone,
    so a root find recovers it without reference to the minimizer's answer. Not a
    golden — a generation run that fails this raises rather than shipping."""
    jerk_time, hold_time, cruise_time = free
    if distance <= 0.0 or cruise_time > 1e-9:
        return
    peak_speed, _ = _profile_peaks(jerk_time, hold_time, jerk_limit, acceleration_limit)
    if _ramp_distance(speed_limit, acceleration_limit, jerk_limit) <= distance:
        return
    rooted = brentq(
        lambda speed: _ramp_distance(speed, acceleration_limit, jerk_limit) - distance,
        1e-12,
        speed_limit,
        xtol=1e-14,
        rtol=8.9e-16,
    )
    assert abs(peak_speed - rooted) < 1e-9, (
        f"minimizer peaked at {peak_speed}, a root find says {rooted}"
    )


def _phase_schedule(free, jerk_limit, acceleration_limit):
    """The seven phases as `(duration, jerk, acceleration to start at)`.

    A phase starting at `None` carries on from wherever the last one left off, so
    the S-curve's accelerations come out of the integration rather than going in.
    Without a jerk limit nothing changes on its own, so the steps are given."""
    jerk_time, hold_time, cruise_time = free
    if jerk_limit is None:
        step = acceleration_limit
        return [
            (0.0, 0.0, 0.0),
            (hold_time, 0.0, step),
            (0.0, 0.0, None),
            (cruise_time, 0.0, 0.0),
            (0.0, 0.0, None),
            (hold_time, 0.0, -step),
            (0.0, 0.0, None),
        ]
    return [
        (jerk_time, jerk_limit, 0.0),
        (hold_time, 0.0, None),
        (jerk_time, -jerk_limit, None),
        (cruise_time, 0.0, None),
        (jerk_time, -jerk_limit, None),
        (hold_time, 0.0, None),
        (jerk_time, jerk_limit, None),
    ]


def _sample_profile(free, jerk_limit, acceleration_limit, sign, times):
    """Position, velocity and acceleration at each time, by integrating the jerk schedule."""
    schedule = _phase_schedule(free, jerk_limit, acceleration_limit)
    total = sum(duration for duration, _, _ in schedule)
    if total <= 0.0:
        # An axis with nowhere to go stays where it is for as long as it is asked.
        return [[0.0, 0.0, 0.0] for _ in times]
    rows = [None] * len(times)

    state = np.array([0.0, 0.0, 0.0])
    phase_start = 0.0
    for duration, jerk, forced in schedule:
        if forced is not None:
            state[2] = forced
        phase_end = phase_start + duration
        if duration <= 0.0:
            phase_start = phase_end
            continue

        solved = solve_ivp(
            lambda _t, y, jerk=jerk: [y[1], y[2], jerk],
            (0.0, duration),
            state,
            method="DOP853",
            rtol=1e-13,
            atol=1e-15,
            dense_output=True,
        )
        assert solved.success, "the jerk schedule could not be integrated"

        for index, time in enumerate(times):
            clamped = min(max(time, 0.0), total)
            # Each time belongs to the phase it falls inside, the last one taking
            # the finish itself.
            inside = phase_start <= clamped < phase_end
            if inside or (clamped >= total and phase_end >= total):
                rows[index] = [value * sign for value in solved.sol(clamped - phase_start)]

        state = solved.sol(duration)
        phase_start = phase_end

    for index, row in enumerate(rows):
        assert row is not None, f"time {times[index]} fell outside every phase"

    # The move is over at the finish, so nothing is commanded there. Without a
    # jerk limit the deceleration is still on the instant before, and the two
    # differ; the position and velocity are still whatever the integration made
    # them.
    for index, time in enumerate(times):
        if time >= total:
            rows[index][2] = 0.0
    return rows


def _profile_durations(free):
    """The three free lengths as the seven phases multicalc stores."""
    jerk_time, hold_time, cruise_time = free
    return [jerk_time, hold_time, jerk_time, cruise_time, jerk_time, hold_time, jerk_time]


def _write_profile(
    out, meta, case, distance, speed_limit, acceleration_limit, jerk_limit, equation
):
    magnitude = abs(distance)
    sign = -1.0 if distance < 0.0 else 1.0
    free = _solve_profile(magnitude, speed_limit, acceleration_limit, jerk_limit)
    _cross_check(free, magnitude, speed_limit, acceleration_limit, jerk_limit)

    durations = _profile_durations(free)
    total = float(sum(durations))
    times = [total * fraction / 8.0 for fraction in range(9)]
    rows = _sample_profile(free, jerk_limit, acceleration_limit, sign, times)

    schema.write_fixture(
        out,
        "motion",
        case,
        meta,
        {"f64": schema.tol(*PROFILE_DURATION_TOL)},
        {
            "distance": schema.scalar(distance),
            "speed_limit": schema.scalar(speed_limit),
            "acceleration_limit": schema.scalar(acceleration_limit),
            "jerk_limit": schema.scalar(jerk_limit if jerk_limit is not None else 0.0),
            "has_jerk_limit": schema.integer(1 if jerk_limit is not None else 0),
            "sample_times": schema.vector(times),
        },
        {
            "phase_durations": schema.vector(durations),
            "total_duration": schema.scalar(total),
            "sampled_states": schema.matrix(rows),
            "state_tolerance": schema.vector([PROFILE_STATE_TOL[0], PROFILE_STATE_TOL[1]]),
        },
        equation=equation,
        operations=["phase durations", "p, v, a at 9 samples"],
    )


def _write_synchronized(out, meta, case, displacements, limits, equation):
    """Every axis solved on its own, then time-scaled onto the slowest one's clock."""
    solved = []
    for displacement, (speed_limit, acceleration_limit, jerk_limit) in zip(displacements, limits):
        magnitude = abs(displacement)
        free = _solve_profile(magnitude, speed_limit, acceleration_limit, jerk_limit)
        _cross_check(free, magnitude, speed_limit, acceleration_limit, jerk_limit)
        solved.append(free)

    totals = [sum(_profile_durations(free)) for free in solved]
    total = float(max(totals))
    times = [total * fraction / 8.0 for fraction in range(9)]

    rows = []
    per_axis = []
    for displacement, free, alone, (_, acceleration_limit, jerk_limit) in zip(
        displacements, solved, totals, limits
    ):
        sign = -1.0 if displacement < 0.0 else 1.0
        if alone > 0.0:
            # Time-scaling: lengths stretch, and the jerk and acceleration that
            # drive them shrink by the cube and the square of the same factor.
            scale = total / alone
            stretched = tuple(length * scale for length in free)
            scaled_jerk = None if jerk_limit is None else jerk_limit / scale**3
            scaled_acceleration = acceleration_limit / scale**2
        else:
            stretched = free
            scaled_jerk = jerk_limit
            scaled_acceleration = acceleration_limit
        per_axis.append(_sample_profile(stretched, scaled_jerk, scaled_acceleration, sign, times))

    # One row per time and axis, so the fixture reads in the order a caller asks.
    for index in range(len(times)):
        for axis in per_axis:
            rows.append(axis[index])

    schema.write_fixture(
        out,
        "motion",
        case,
        meta,
        {"f64": schema.tol(*PROFILE_DURATION_TOL)},
        {
            "displacements": schema.vector(displacements),
            "speed_limits": schema.vector([limit[0] for limit in limits]),
            "acceleration_limits": schema.vector([limit[1] for limit in limits]),
            "jerk_limits": schema.vector([limit[2] for limit in limits]),
            "axis_count": schema.integer(len(displacements)),
            "sample_times": schema.vector(times),
        },
        {
            "total_duration": schema.scalar(total),
            "sampled_states": schema.matrix(rows),
            "state_tolerance": schema.vector([PROFILE_STATE_TOL[0], PROFILE_STATE_TOL[1]]),
        },
        equation=equation,
        operations=["synchronized duration", "p, v, a at 9 samples"],
    )


def _write(out, meta, case, waypoints, durations, start, end, times, equation):
    coefficients = _plan(waypoints, durations, start, end)
    segments = len(durations)
    flat = [
        list(coefficients[segment, axis])
        for segment in range(segments)
        for axis in range(DIMENSION)
    ]
    schema.write_fixture(
        out,
        "motion",
        case,
        meta,
        {"f64": schema.tol(*COEFFICIENT_TOL)},
        {
            "waypoints": schema.matrix(waypoints),
            "durations": schema.vector(durations),
            "segment_count": schema.integer(segments),
            "start_velocity": schema.vector(start[0]),
            "start_acceleration": schema.vector(start[1]),
            "start_jerk": schema.vector(start[2]),
            "end_velocity": schema.vector(end[0]),
            "end_acceleration": schema.vector(end[1]),
            "end_jerk": schema.vector(end[2]),
            "sample_times": schema.vector(times),
        },
        {
            "coefficients": schema.matrix(flat),
            "sampled_states": schema.matrix(_sample(coefficients, durations, times)),
            "state_tolerance": schema.vector([STATE_TOL[0], STATE_TOL[1]]),
        },
        equation=equation,
        operations=["segment coefficients", "p, v, a at 9 samples"],
    )


def run(out, seed):
    meta = schema.metadata(
        "motion",
        seed,
        "hand-chosen waypoints with durations either given or from an average speed",
        libraries=("numpy", "scipy"),
        reference="NumPy {numpy} / SciPy {scipy}",
    )
    at_rest = [[0.0] * DIMENSION] * 3

    _write(
        out, meta, "minimum_snap_single_segment_3d",
        [[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]],
        [2.0], at_rest, at_rest,
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        equation="2 waypoints, 3D, rest to rest",
    )

    three = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.5], [3.0, 1.0, 1.5], [4.0, 3.0, 1.0]]
    three_durations = [1.0, 1.5, 1.2]
    three_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.4, 3.7]
    _write(
        out, meta, "minimum_snap_three_segments_3d",
        three, three_durations, at_rest, at_rest, three_times,
        equation="4 waypoints, 3D, rest to rest",
    )

    moving_start = [[1.0, 0.0, 0.0], [0.0] * DIMENSION, [0.0] * DIMENSION]
    moving_end = [[0.0, -0.5, 0.0], [0.0] * DIMENSION, [0.0] * DIMENSION]
    _write(
        out, meta, "minimum_snap_three_segments_moving_ends_3d",
        three, three_durations, moving_start, moving_end, three_times,
        equation="4 waypoints, 3D, moving endpoints",
    )

    box = [
        [0.0, 0.0, 0.0], [7.0, 0.0, 1.0], [7.0, 7.0, 2.5], [0.0, 7.0, 1.5],
        [0.0, 0.0, 2.0], [3.5, 3.5, 2.5], [7.0, 3.5, 1.0], [3.5, 0.0, 0.5],
    ]
    speed = 3.0
    box_durations = []
    for first, second in zip(box, box[1:]):
        distance = float(np.linalg.norm(np.array(second) - np.array(first)))
        box_durations.append(distance / speed)
    total = float(sum(box_durations))
    _write(
        out, meta, "minimum_snap_seven_segments_3d",
        box, box_durations, at_rest, at_rest,
        [total * fraction / 8.0 for fraction in range(9)],
        equation="8 waypoints, 3D, 7 × 7 × 2.5 m box at 3 m/s average",
    )

    _write_profile(
        out, meta, "profile_trapezoid_cruise_1d",
        10.0, 2.0, 1.0, None,
        equation="trapezoidal, d = 10 m, v_max = 2 m/s, a_max = 1 m/s²; velocity-limited",
    )
    _write_profile(
        out, meta, "profile_trapezoid_triangular_1d",
        1.0, 2.0, 1.0, None,
        equation="trapezoidal, d = 1 m, v_max = 2 m/s, a_max = 1 m/s²; triangular, no cruise",
    )
    _write_profile(
        out, meta, "profile_jerk_limited_cruise_1d",
        10.0, 2.0, 1.0, 2.0,
        equation="S-curve, d = 10 m, v_max = 2 m/s, a_max = 1 m/s², j_max = 2 m/s³; all 7 phases",
    )
    _write_profile(
        out, meta, "profile_jerk_limited_no_cruise_1d",
        1.5, 2.0, 1.0, 2.0,
        equation="S-curve, d = 1.5 m, v_max = 2 m/s, a_max = 1 m/s², j_max = 2 m/s³; no cruise",
    )
    _write_profile(
        out, meta, "profile_jerk_limited_short_1d",
        0.25, 2.0, 1.0, 2.0,
        equation="S-curve, d = 0.25 m, v_max = 2 m/s, a_max = 1 m/s², j_max = 2 m/s³; a below a_max",
    )
    _write_synchronized(
        out, meta, "profile_synchronized_three_axes",
        [1.0, -0.5, 0.0],
        # The second joint is the slower of the two that move, so the first is
        # time-scaled onto its clock rather than the two coinciding.
        [(1.0, 2.0, 10.0), (0.4, 1.0, 5.0), (2.0, 2.0, 10.0)],
        equation="3 axes, d = [1, −0.5, 0] m, per-axis limits, common finish time",
    )
