"""Mapping goldens: the exact Euclidean distance transform, costmap inflation, and the
likelihood-field measurement model.

The distance field is the load-bearing one. multicalc computes it by Felzenszwalb-Huttenlocher:
two separable one-dimensional passes taking the lower envelope of parabolas, in squared cell
units, rooted at the end. `scipy.ndimage.distance_transform_edt` computes the same quantity by a
different exact method. Both are exact, so a disagreement beyond floating-point noise means one of
them is wrong rather than that one is an approximation.

The costmap and the likelihood field are closed forms in numpy, evaluated straight from their
definitions rather than by reimplementing anything.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

import schema

# Both sides compute an exact transform, so the only slack is floating-point noise.
DISTANCE_TOL = (1e-12, 1e-12)
# Both sides are u8 counts, compared exactly.
COST_TOL = (0.0, 0.0)
# A total log-weight sums 32 logarithms, so it is compared a little looser than one lookup.
SCORE_TOL = (1e-10, 1e-10)

LETHAL = 255
HIGHEST_PASSABLE = 254


def _occupancy_matrix(blocked):
    """A boolean map as a matrix of 0.0 and 1.0, which is what the fixture schema carries."""
    return schema.matrix([[1.0 if cell else 0.0 for cell in row] for row in blocked])


def _distance_field(blocked, resolution):
    """Each cell's distance in metres to the nearest blocked cell."""
    free = np.logical_not(blocked)
    return distance_transform_edt(free) * resolution


def _single_obstacle(rows, columns, row, column):
    blocked = np.zeros((rows, columns), dtype=bool)
    blocked[row, column] = True
    return blocked


def _straight_wall(rows, columns, row):
    blocked = np.zeros((rows, columns), dtype=bool)
    blocked[row, :] = True
    return blocked


def _two_obstacles(rows, columns, first, second):
    blocked = np.zeros((rows, columns), dtype=bool)
    blocked[first] = True
    blocked[second] = True
    return blocked


def _random_map(rows, columns, density, seed):
    """A scattered map at the given density, from its own generator so the shared stream is left
    alone."""
    rng = np.random.default_rng(seed)
    blocked = rng.random((rows, columns)) < density
    # A map with nothing on it has no distance to report, so one cell is forced.
    if not blocked.any():
        blocked[rows // 2, columns // 2] = True
    return blocked


def _write_distance_case(out, meta, case, blocked, resolution, shape):
    rows, columns = blocked.shape
    field = _distance_field(blocked, resolution)

    inputs = {
        "kind": schema.string("distance_transform"),
        "rows": schema.integer(rows),
        "columns": schema.integer(columns),
        "resolution": schema.scalar(resolution),
        "occupancy": _occupancy_matrix(blocked),
    }
    expected = {"distance": schema.matrix(field)}
    schema.write_fixture(
        out,
        "mapping",
        case,
        meta,
        {"f64": schema.tol(*DISTANCE_TOL)},
        inputs,
        expected,
        equation="d(cell) = min over blocked cells of the Euclidean distance, in metres",
        operations=[
            f"Exact Euclidean distance transform: {shape}, {rows}x{columns} at "
            f"{resolution} m cells"
        ],
    )
    return field


def _inflate(field, inscribed_radius, inflation_radius, cost_scaling_factor):
    """The nav2 inflation of a distance field, from its definition."""
    decayed = HIGHEST_PASSABLE * np.exp(
        -cost_scaling_factor * (field - inscribed_radius)
    )
    cost = np.where(
        field <= inscribed_radius,
        float(LETHAL),
        np.where(field <= inflation_radius, np.floor(decayed), 0.0),
    )
    return cost


def _write_costmap_case(out, meta, case, blocked, resolution, parameter_sets, shape,
                        numpy_version):
    rows, columns = blocked.shape
    field = _distance_field(blocked, resolution)

    inputs = {
        "kind": schema.string("costmap"),
        "rows": schema.integer(rows),
        "columns": schema.integer(columns),
        "resolution": schema.scalar(resolution),
        "occupancy": _occupancy_matrix(blocked),
        "inscribed_radius": schema.vector([row[0] for row in parameter_sets]),
        "inflation_radius": schema.vector([row[1] for row in parameter_sets]),
        "cost_scaling_factor": schema.vector([row[2] for row in parameter_sets]),
    }
    expected = {}
    for index, (inscribed, inflation, scaling) in enumerate(parameter_sets):
        expected[f"cost_{index}"] = schema.matrix(
            _inflate(field, inscribed, inflation, scaling)
        )
    # The inflation is a closed form evaluated in NumPy, not a SciPy routine, so the accuracy
    # table credits the library that actually produced it.
    closed_form = {**meta, "reference": f"NumPy {numpy_version}"}
    schema.write_fixture(
        out,
        "mapping",
        case,
        closed_form,
        {"f64": schema.tol(*COST_TOL)},
        inputs,
        expected,
        equation=(
            "cost = 255 where d <= inscribed; floor(254·exp(-k·(d − inscribed))) where "
            "d <= inflation; 0 beyond"
        ),
        operations=[
            f"Costmap inflation: {shape}, {rows}x{columns}, {len(parameter_sets)} parameter sets"
        ],
    )


def _bilinear(field, resolution, origin, point):
    """The field at a world point, blended over the four surrounding cell centres.

    Returns `None` where any of the four falls outside, which is what multicalc reports there.
    """
    rows, columns = field.shape
    column_axis = (point[0] - origin[0]) / resolution - 0.5
    row_axis = (point[1] - origin[1]) / resolution - 0.5
    column_floor = np.floor(column_axis)
    row_floor = np.floor(row_axis)
    if column_floor < 0 or row_floor < 0:
        return None
    row = int(row_floor)
    column = int(column_floor)
    if row + 1 >= rows or column + 1 >= columns:
        return None
    row_fraction = row_axis - row_floor
    column_fraction = column_axis - column_floor

    lower = field[row, column] + (field[row, column + 1] - field[row, column]) * column_fraction
    upper = (
        field[row + 1, column]
        + (field[row + 1, column + 1] - field[row + 1, column]) * column_fraction
    )
    return lower + (upper - lower) * row_fraction


def _write_likelihood_case(
    out,
    meta,
    case,
    blocked,
    resolution,
    poses,
    ranges,
    field_of_view,
    maximum_range,
    deviation,
    random_weight,
    numpy_version,
):
    rows, columns = blocked.shape
    field = _distance_field(blocked, resolution)
    origin = (0.0, 0.0)
    num_beams = len(ranges)

    # Beam angles, measured from straight ahead and growing to the left.
    offsets = [
        -field_of_view * 0.5 + field_of_view * index / (num_beams - 1)
        for index in range(num_beams)
    ]

    endpoint_distances = []
    log_weights = []
    for pose in poses:
        score = 0.0
        for beam, measured in enumerate(ranges):
            bearing = pose[2] + offsets[beam]
            endpoint = (
                pose[0] + measured * np.cos(bearing),
                pose[1] + measured * np.sin(bearing),
            )
            distance = _bilinear(field, resolution, origin, endpoint)
            # An endpoint off the field is infinitely far from any obstacle, so its Gaussian term
            # vanishes and only the noise floor is left.
            hit = 0.0 if distance is None else np.exp(
                -(distance * distance) / (2.0 * deviation * deviation)
            )
            endpoint_distances.append(-1.0 if distance is None else float(distance))
            score += float(
                np.log((1.0 - random_weight) * hit + random_weight / maximum_range)
            )
        log_weights.append(score)

    inputs = {
        "kind": schema.string("likelihood_field"),
        "rows": schema.integer(rows),
        "columns": schema.integer(columns),
        "resolution": schema.scalar(resolution),
        "occupancy": _occupancy_matrix(blocked),
        "field_of_view": schema.scalar(field_of_view),
        "maximum_range": schema.scalar(maximum_range),
        "measurement_deviation": schema.scalar(deviation),
        "random_measurement_weight": schema.scalar(random_weight),
        "ranges": schema.vector(ranges),
        "pose_x": schema.vector([pose[0] for pose in poses]),
        "pose_y": schema.vector([pose[1] for pose in poses]),
        "pose_heading": schema.vector([pose[2] for pose in poses]),
    }
    expected = {
        # One entry per pose per beam, in that order. A distance of -1 marks an endpoint that fell
        # off the field.
        "endpoint_distance": schema.vector(endpoint_distances),
        "log_weight": schema.vector(log_weights),
    }
    # The score is a closed form in NumPy; only the field underneath it came from SciPy.
    closed_form = {**meta, "reference": f"NumPy {numpy_version}"}
    schema.write_fixture(
        out,
        "mapping",
        case,
        closed_form,
        {"f64": schema.tol(*SCORE_TOL)},
        inputs,
        expected,
        equation=(
            "log w = sum over beams of ln((1 − p)·exp(−d²/2σ²) + p/r_max), for d the endpoint's "
            "distance to the nearest obstacle"
        ),
        operations=[
            f"Likelihood field: {num_beams}-beam scan scored from {len(poses)} poses"
        ],
    )


def run(out, seed):
    meta = schema.metadata(
        "mapping",
        seed,
        "five fixed occupancy maps: a single obstacle, a straight wall, two obstacles, and two "
        "scattered maps drawn at fixed densities from their own seeded generators",
        libraries=("numpy", "scipy"),
        reference="SciPy {scipy}",
    )

    single = _single_obstacle(9, 9, 4, 4)
    wall = _straight_wall(16, 16, 5)
    pair = _two_obstacles(16, 16, (2, 3), (11, 12))
    scattered_small = _random_map(32, 32, 0.2, 0)
    scattered_large = _random_map(64, 64, 0.35, 1)

    _write_distance_case(
        out, meta, "distance_transform_single_obstacle_9x9", single, 0.2, "one obstacle"
    )
    _write_distance_case(
        out, meta, "distance_transform_straight_wall_16x16", wall, 0.1, "a straight wall"
    )
    _write_distance_case(
        out, meta, "distance_transform_two_obstacles_16x16", pair, 0.1, "two obstacles"
    )
    _write_distance_case(
        out, meta, "distance_transform_random_32x32", scattered_small, 0.25,
        "scattered at density 0.2",
    )
    _write_distance_case(
        out, meta, "distance_transform_random_64x64", scattered_large, 0.05,
        "scattered at density 0.35",
    )

    parameter_sets = [(0.2, 1.0, 3.0), (0.0, 0.5, 10.0), (0.5, 0.5, 1.0)]
    _write_costmap_case(
        out, meta, "costmap_straight_wall_16x16", wall, 0.1, parameter_sets,
        "a straight wall", np.__version__,
    )
    _write_costmap_case(
        out, meta, "costmap_random_32x32", scattered_small, 0.25, parameter_sets,
        "scattered at density 0.2", np.__version__,
    )

    # A 32-beam scan reading the same range on every beam, scored from three poses well inside the
    # map so most endpoints land on the field and a few run off it.
    _write_likelihood_case(
        out,
        meta,
        "likelihood_field_wall_16x16",
        wall,
        0.1,
        poses=[(0.75, 1.05, 0.0), (0.85, 0.95, 0.4), (0.55, 1.25, -0.9)],
        ranges=[0.4] * 32,
        field_of_view=np.pi,
        maximum_range=2.0,
        deviation=0.2,
        random_weight=0.05,
        numpy_version=np.__version__,
    )
