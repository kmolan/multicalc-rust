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
"""

import numpy as np

import schema

COEFFICIENTS_PER_SEGMENT = 8
DIMENSION = 3

# The full system is larger and worse conditioned than the reduced one multicalc
# solves, so the coefficients agree to rather less than the states they produce.
COEFFICIENT_TOL = (1e-7, 1e-7)
STATE_TOL = (1e-9, 1e-9)


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
        operations=["segment coefficients", "position, velocity and acceleration at nine times"],
    )


def run(out, seed):
    meta = schema.metadata(
        "motion",
        seed,
        "hand-chosen waypoints with durations either given or from an average speed",
        libraries=("numpy",),
        reference="NumPy {numpy}",
    )
    at_rest = [[0.0] * DIMENSION] * 3

    _write(
        out, meta, "minimum_snap_single_segment_3d",
        [[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]],
        [2.0], at_rest, at_rest,
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        equation="the smoothest path between two points in three dimensions",
    )

    three = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.5], [3.0, 1.0, 1.5], [4.0, 3.0, 1.0]]
    three_durations = [1.0, 1.5, 1.2]
    three_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.4, 3.7]
    _write(
        out, meta, "minimum_snap_three_segments_3d",
        three, three_durations, at_rest, at_rest, three_times,
        equation="the smoothest path through four waypoints, from a standstill to a standstill",
    )

    moving_start = [[1.0, 0.0, 0.0], [0.0] * DIMENSION, [0.0] * DIMENSION]
    moving_end = [[0.0, -0.5, 0.0], [0.0] * DIMENSION, [0.0] * DIMENSION]
    _write(
        out, meta, "minimum_snap_three_segments_moving_ends_3d",
        three, three_durations, moving_start, moving_end, three_times,
        equation="the same four waypoints, entered and left while already moving",
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
        equation="eight waypoints around a 7 × 7 × 2.5 m box, timed for a 3 m/s average",
    )
