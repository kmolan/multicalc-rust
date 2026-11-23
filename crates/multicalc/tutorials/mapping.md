# Mapping

Occupancy maps, ray casts across them, and the distance and cost fields a planner uses. Everything
is generic over `Numeric` and works without `std`.

- `OccupancyMap`: the trait every map implements. It gives the size, the world placement, the
  occupancy of a cell, and the distance a beam travels before it hits an occupied cell.
- `MutableOccupancyMap`: rasterizes world geometry into cells: points, polylines and circles.
- `OccupancyGrid`: a bit-packed grid sized at compile time, one bit per cell, with no heap.
- `DynamicOccupancyGrid` (`alloc` only): a grid sized at run time, on the heap.
- `LogOddsGrid`: a map that the robot builds from range scans, by addition in log-odds.
- `CellState`: free, occupied or unknown. A planner must not route through unknown space.
- `GridGeometry` and `RayWalk`: the world placement and index arithmetic of a grid, and the cells a
  beam crosses, in order, as an iterator.
- `DistanceField` and `CostGrid`: the distance from each cell to the nearest obstacle, and the cost
  to enter each cell.
- `ScanGeometry`: the beam directions and range limits of a forward-facing range scan.

## Grids and ray casts

A map is a rectangle of square cells, each free or occupied. Cells are indexed row first, as they
are stored. Cell `(row, column)` covers the world square from
`origin + [column · resolution, row · resolution]`, so `origin` is the lowest corner of cell
`(0, 0)`. A point exactly on the edge between two cells belongs to the higher cell.

A ray cast walks the beam one cell at a time (DDA) and returns the point where it first crosses into
an occupied cell. The distance is to that crossing, not to the cell centre, so a wall on a cell edge
reads exactly. A beam that starts in an occupied cell reads zero. A beam that starts off the map is
walked from the point where it enters the map, and the distance is still from its start point.

The rasterizers draw outlines, not fills: `occupy_polyline` around a closed shape leaves the inside
free. They sample at less than one cell width, so a drawn wall has no gap for a beam. A drawn line is
one cell thick, so the face that a beam hits can be one cell to either side of the line. That is the
accuracy of a scan against a map.

`ScanGeometry` describes the shape of a scan, not a sensor: the direction of each beam and the
maximum range, with no noise model or rate. Beam `0` is at the right edge of the arc and angles
increase to the left, in the same order as the ranges of a
[ROS `LaserScan`](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html).
[`FollowTheGap`](control.md) numbers its beams with the same formula, so a scan and the steering from
it agree beam for beam.

```rust
use multicalc::mapping::{DynamicOccupancyGrid, MutableOccupancyMap, OccupancyMap, ScanGeometry};
use multicalc::{SE2, SO2, Vector2D};

// A 4 m by 3 m room of 10 cm cells, with its lowest corner at the world origin.
let cell_size = 0.1_f64;
let columns = 40;
let rows = 30;
let lowest_corner = [0.0, 0.0];
let mut room = DynamicOccupancyGrid::try_new(columns, rows, cell_size, lowest_corner)?;

// A wall across the far end, and a pillar in the middle of the floor.
let wall = [[3.0, 0.0], [3.0, 3.0]];
let open_ended = false;
room.occupy_polyline(&wall, open_ended);
let pillar_centre = [2.0, 1.5];
let pillar_radius = 0.2;
room.occupy_circle(pillar_centre, pillar_radius);

// A robot 1 m in, facing along its row, hits the pillar before the wall.
let robot = [1.0, 1.5];
let along_the_row = 0.0;
let maximum_range = 5.0;
let pillar_face = pillar_centre[0] - pillar_radius - robot[0];
let distance = room.cast_ray(robot, along_the_row, maximum_range);
assert!(distance.is_some_and(|met| (met - pillar_face).abs() <= 2.0 * cell_size));

// Behind it the room is open to the map edge, so the beam hits nothing.
let back_along_the_row = core::f64::consts::PI;
assert!(room.cast_ray(robot, back_along_the_row, maximum_range).is_none());

// A whole scan gives the ranges of a perfect sensor. A beam that hits nothing reads the maximum
// range of the scan.
const NUM_BEAMS: usize = 5;
let quarter_turn = core::f64::consts::FRAC_PI_2;
let scan: ScanGeometry<NUM_BEAMS> = ScanGeometry::try_new(quarter_turn, maximum_range)?;
let pose = SE2::from_parts(SO2::from_angle(along_the_row), Vector2D::new(robot));
let ranges = room.cast_scan(pose, &scan);

// The middle beam hits the pillar square on, so it reads the shortest of the five.
assert!(ranges.iter().all(|&range| range >= ranges[2]));
# Ok::<(), multicalc::CalcError>(())
```

`with_minimum_range` sets the blind zone of a sensor. `range_is_valid` tells if a reading is a
finite distance in `[minimum_range, maximum_range]`. Infinity, the reading of a beam that hit
nothing, is not valid.

```rust
use multicalc::mapping::ScanGeometry;

let quarter_turn = core::f64::consts::FRAC_PI_2;
let maximum_range = 4.0;
let blind_spot = 0.12;
let scan: ScanGeometry<5> =
    ScanGeometry::try_new(quarter_turn, maximum_range)?.with_minimum_range(blind_spot)?;

let a_wall_two_metres_off = 2.0;
assert!(scan.range_is_valid(a_wall_two_metres_off));
assert!(!scan.range_is_valid(f64::INFINITY));

// A direction maps back to the beam nearest to it.
let straight_ahead = 0.0;
assert_eq!(scan.nearest_beam_index(straight_ahead), Some(2));
# Ok::<(), multicalc::CalcError>(())
```

## Fixed-size grid

`DynamicOccupancyGrid` stores one byte per cell on the heap, so a 20 m by 15 m floor at 5 cm cells
is 120 000 cells and approximately 120 KB. `OccupancyGrid` stores one bit per cell in `u32` words,
sized at compile time, with no heap.

`WORDS_PER_ROW` must be at least `COLUMNS.div_ceil(32)`. It is a type parameter, not a computed
value, because stable Rust has no `generic_const_exprs`. `try_new` checks it and returns
`MappingError::WordsPerRowTooSmall` if it is too small. Memory is `ROWS · WORDS_PER_ROW · 4` bytes.

```rust
use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid, OccupancyMap};

// A 6.4 m square at 5 cm cells: 128 by 128, and 128 / 32 = 4 words to a row.
let cell_size = 0.05_f64;
let mut room: OccupancyGrid<128, 128, 4> = OccupancyGrid::try_new(cell_size, [0.0, 0.0])?;

// 128 rows of 4 words of 4 bytes: 2 KB of cells, against 16 KB for a byte-per-cell grid.
assert_eq!(128 * room.words_per_row() * 4, 2048);

let wall = [[3.2, 1.0], [3.2, 5.0]];
let open_ended = false;
room.occupy_polyline(&wall, open_ended);

let standing_at = [1.0, 3.0];
let east = 0.0;
let maximum_range = 6.0;
let distance = room.cast_ray(standing_at, east, maximum_range);
assert!(distance.is_some_and(|met| (met - 2.2).abs() <= cell_size));
# Ok::<(), multicalc::CalcError>(())
```

A map does not have to be one of these types. Implement the five required methods of `OccupancyMap`
(`columns`, `rows`, `resolution`, `origin`, `is_occupied`) over any storage, for example a fixed array
or cells in flash. The ray casts, the rasterizers and the scan helpers are provided methods.

## Map from scans

`LogOddsGrid` builds a map from sensor data. Each cell holds its log-odds of occupancy,
`l = log(p / (1 − p))`, in `i8` fixed point, so a 128 by 128 map is 16 KB. The Bayesian update is
addition in log-odds. `l = 0` is exactly "unknown", so a new grid needs no initialization.

Each cell that a beam crosses moves toward free, and the cell where it stops moves toward occupied.
**The clamp `[clamp_low, clamp_high]` lets a cell recover**: a person who walks through and leaves
does not become a permanent wall. The defaults are `free_update = -2`, `occupied_update = 5`, clamps
`±40` and thresholds `±10`. `try_with_updates`, `try_with_clamps` and `try_with_thresholds` change
them, and return `MappingError::InvalidBeliefSettings` unless the values are ordered as required.

```rust
use multicalc::mapping::{CellState, LogOddsGrid, OccupancyMap, ScanGeometry};
use multicalc::{SE2, SO2, Vector2D};

// A 4 m square at 10 cm cells, all unknown at the start.
let mut belief: LogOddsGrid<40, 40> = LogOddsGrid::try_new(0.1, [0.0, 0.0])?;
assert_eq!(belief.cell_state(20, 20), CellState::Unknown);

// A narrow scan facing east from the middle, with a wall 1 m away.
let scan: ScanGeometry<3> = ScanGeometry::try_new(0.02, 4.0)?;
let pose = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([2.05, 2.05]));
for _ in 0..4 {
    belief.integrate_scan(pose, &scan, &[1.0; 3]);
}

// The cell where the beams stopped is occupied; the cells they crossed are free; the rest is
// unknown.
assert_eq!(belief.cell_state(20, 30), CellState::Occupied);
assert_eq!(belief.cell_state(20, 25), CellState::Free);
assert_eq!(belief.cell_state(35, 35), CellState::Unknown);

// The obstacle leaves, and the beams now pass through its cell. After twenty scans, the cell
// recovers.
for _ in 0..20 {
    belief.integrate_scan(pose, &scan, &[3.0; 3]);
}
assert_eq!(belief.cell_state(20, 30), CellState::Free);
# Ok::<(), multicalc::CalcError>(())
```

`cell_state` is a provided method on `OccupancyMap`. The default derives from `is_occupied` and
never returns `Unknown`. `LogOddsGrid` overrides it, and a cell off the grid reads `Unknown`, where a
plain occupancy map reads free. This stops a planner from a route through space that no sensor saw.

The work per scan is bounded by `NUM_BEAMS · maximum_range / resolution` cells.

## Scan casts

`cast_scan` gives the range of each beam of a scan from an `SE2` pose, and `scan_endpoints` gives the
world point where each beam ends. A beam that hits nothing reads the maximum range of the scan.

```rust
use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid, OccupancyMap, ScanGeometry};
use multicalc::{SE2, SO2, Vector2D};

let mut room: OccupancyGrid<40, 40, 2> = OccupancyGrid::try_new(0.1, [0.0, 0.0])?;
let wall = [[3.0, 0.5], [3.0, 3.5]];
room.occupy_polyline(&wall, false);

let scan: ScanGeometry<3> = ScanGeometry::try_new(core::f64::consts::FRAC_PI_2, 4.0)?;
let facing_east = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([2.0, 2.0]));

// The middle beam hits the wall 1 m away; the endpoints are where the beams stop.
let ranges = room.cast_scan(facing_east, &scan);
assert!((ranges[1] - 1.0).abs() <= 0.1);

let endpoints = room.scan_endpoints(facing_east, &scan);
assert!((endpoints[1][0] - 3.0).abs() <= 0.1);
# Ok::<(), multicalc::CalcError>(())
```

`cast_scan` gives the maximum range, not infinity, for a beam that hits nothing. To tell "no return"
from "a return at the maximum range", cast each beam with `cast_ray`, which returns `None` for no
return.

## Distance and cost

`DistanceField` holds the distance from each cell to the nearest occupied cell. It is the exact
Euclidean distance transform of Felzenszwalb–Huttenlocher: two separable 1-D lower-envelope passes
over parabolas, O(cells), with no priority queue and no approximation. It is much cheaper than the
equivalent set of ray casts. Build it at design time or at a low rate, not on each tick; the queries
are the per-tick calls.

`distance_at` is bilinear over the four nearest cell centres, so it is smooth in the position and
**exactly differentiable when `T` is an autodiff scalar**. Thus a gradient-based trajectory
optimizer can put an obstacle cost into `LevenbergMarquardt` with no hand-derived Jacobian.
`distance_at` returns `None` when one of the four centres is off the grid. `gradient_at` uses
central differences over one cell.

```rust
use multicalc::mapping::{
    CostGrid, DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid,
};

// A 2 m square at 10 cm cells with a wall along one row.
let mut room: OccupancyGrid<20, 20, 1> = OccupancyGrid::try_new(0.1, [0.0, 0.0])?;
for column in 0..20 {
    room.set_cell(2, column, true);
}

// The caller owns the workspace; it must hold the longest span plus one.
let mut workspace: DistanceTransformWorkspace<21> = DistanceTransformWorkspace::new();
let field: DistanceField<20, 20> = DistanceField::try_build(&room, &mut workspace)?;

// Zero on the wall, and one cell more for each row away from it.
assert_eq!(field.distance_of(2, 10), Some(0.0));
assert!(field.distance_of(7, 10).is_some_and(|d| (d - 0.5).abs() < 1e-12));

// Above the wall, the gradient is the wall normal: straight up, unit length.
let above = field.geometry().center_of(10, 10).expect("cell");
let gradient = field.gradient_at(above).expect("inside");
assert!(gradient[0].abs() < 1e-6 && (gradient[1] - 1.0).abs() < 1e-6);
# Ok::<(), multicalc::CalcError>(())
```

`CostGrid` converts the field into a per-cell traversal cost, with the nav2 inflation formula. For a
cell at distance `d` from the nearest obstacle:

```text
d <= inscribed_radius                    -> LETHAL (255)
inscribed_radius < d <= inflation_radius -> 254 · exp(−cost_scaling_factor · (d − inscribed_radius))
d > inflation_radius                     -> 0
```

`inscribed_radius` is the radius of the largest circle inside the robot footprint, so no part of the
robot can enter a cell within it.

```rust
# use multicalc::mapping::{
#     CostGrid, DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid,
# };
# let mut room: OccupancyGrid<20, 20, 1> = OccupancyGrid::try_new(0.1, [0.0, 0.0])?;
# for column in 0..20 { room.set_cell(2, column, true); }
# let mut workspace: DistanceTransformWorkspace<21> = DistanceTransformWorkspace::new();
# let field: DistanceField<20, 20> = DistanceField::try_build(&room, &mut workspace)?;
let inscribed_radius = 0.2;
let inflation_radius = 0.8;
let cost_scaling_factor = 3.0;
let costmap: CostGrid<20, 20> =
    CostGrid::try_build(&field, inscribed_radius, inflation_radius, cost_scaling_factor)?;

// The wall and the band around it are lethal; far away the cost is zero.
assert_eq!(costmap.cost_of(2, 10), Some(CostGrid::<20, 20>::LETHAL));
assert_eq!(costmap.cost_of(19, 10), Some(0));

// The cost decreases monotonically away from the wall.
let near = costmap.cost_of(5, 10).expect("cell");
let far = costmap.cost_of(9, 10).expect("cell");
assert!(near > far);
# Ok::<(), multicalc::CalcError>(())
```

The planner adapter for a costmap, `CostmapCost`, is in [`planning`](planning.md), which owns the
traversal-cost trait. If it were here, the two modules would depend on each other.

## Localizing with a likelihood field

With a distance field, a particle filter can score a scan without one ray cast per beam per
particle. `LikelihoodFieldModel` (`alloc` only, in `estimation`) projects the endpoint of each beam
from the particle pose and looks up its distance to the nearest obstacle. An endpoint on a wall
scores high, and an endpoint in open space scores low. One interpolated lookup replaces one DDA
walk. The score is also smoother in the pose than the beam model, whose likelihood is jagged because
it depends on map resolution.

The model ignores occlusion, so a pose can score high when its beams go through a wall. Where that
matters, use `BeamModel` through `MonteCarloLocalizer::update`. `update_against_field` returns
`EstimationError::InvalidTuning` for a non-finite or non-positive `measurement_deviation`, or a
`random_measurement_weight` outside `[0, 1]`.

```rust
use multicalc::estimation::{
    BeamModel, InitialParticleCloud, LikelihoodFieldModel, MonteCarloLocalizer,
};
use multicalc::mapping::{
    DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid, OccupancyMap,
    ScanGeometry,
};
use multicalc::{SE2, SO2, Vector2D};

// A 6 m square room at 20 cm cells, with walls all round.
let mut room: OccupancyGrid<30, 30, 1> = OccupancyGrid::try_new(0.2, [0.0, 0.0])?;
room.occupy_polyline(&[[0.5, 0.5], [5.5, 0.5], [5.5, 5.5], [0.5, 5.5]], true);

let mut workspace: DistanceTransformWorkspace<31> = DistanceTransformWorkspace::new();
let field: DistanceField<30, 30> = DistanceField::try_build(&room, &mut workspace)?;

// The true pose; the localizer gets only an approximate hint.
const NUM_BEAMS: usize = 16;
let scan: ScanGeometry<NUM_BEAMS> = ScanGeometry::try_new(core::f64::consts::TAU * 0.9, 8.0)?;
let truth = [2.0, 3.0, 0.4];
let reading = room.cast_scan(
    SE2::from_parts(SO2::from_angle(truth[2]), Vector2D::new([truth[0], truth[1]])),
    &scan,
);

let cloud = InitialParticleCloud {
    particle_count: 400,
    position_variance: 0.05,
    heading_variance: 0.01,
};
let mut localizer: MonteCarloLocalizer<NUM_BEAMS> =
    MonteCarloLocalizer::new([2.3, 2.7, 0.4], cloud, BeamModel::default(), 20260830)?;

for _ in 0..8 {
    localizer.update_against_field(&field, &scan, &reading, LikelihoodFieldModel::default())?;
}

let (pose, _spread) = localizer.estimate();
assert!((pose[0] - truth[0]).abs() < 0.5);
assert!((pose[1] - truth[1]).abs() < 0.5);
# Ok::<(), multicalc::CalcError>(())
```

## Errors

Errors are [`MappingError`](error-handling.md):

- `NonFinite`, `NonPositiveResolution`, `EmptyGrid`, `GridTooLarge`: grid placement and size.
- `WordsPerRowTooSmall`: `WORDS_PER_ROW` below `COLUMNS.div_ceil(32)`.
- `TooFewBeams`, `InvalidFieldOfView` (outside `(0, 2π]`), `NonPositiveRange`, `InvalidRangeLimits`:
  scan geometry.
- `GridShapeMismatch`, `WorkspaceTooSmall`: a field built from a map of a different shape, or a
  transform workspace shorter than the longest grid span plus one.
- `InvalidBeliefSettings`: `LogOddsGrid` updates, clamps or thresholds out of order.
- `RadiiNotOrdered`, `NonPositiveScaling`: `CostGrid` with `inscribed_radius > inflation_radius`, or
  `cost_scaling_factor ≤ 0`.

Full demos:
[path_planning.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/path_planning.rs)
(distance field and costmap) and
[2d_localization_obstacle_avoidance.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/2d_localization_obstacle_avoidance.rs).

## Next

[Planning](planning.md) uses all of this: a search over the map, with costs from the costmap, that
returns a `PolylinePath` for a smoother and a path follower.

---

[Back to the tutorial index](README.md)
