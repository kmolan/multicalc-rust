# Mapping

The map a robot measures against, and how far a beam travels across it.

- `OccupancyMap`: what any map must answer — its size, where it sits in the world, whether a cell is
  blocked, and how far a beam travels before it meets something.
- `MutableOccupancyMap`: marking cells from world geometry — single points, joined-up lines, and
  circles.
- `OccupancyGrid`: a bit-packed grid sized at compile time, one bit per cell — what a small board
  uses.
- `DynamicOccupancyGrid` (`alloc` only): a grid of cells sized when the program runs.
- `LogOddsGrid`: a map the robot builds itself, integrating range scans by addition in log-odds.
- `CellState`: free, blocked, or not yet observed — the distinction a planner needs.
- `GridGeometry` and `RayWalk`: where a grid sits in the world, and the cells a beam passes through.
- `DistanceField` and `CostGrid`: how far every cell is from an obstacle, and what that costs to
  enter.
- `ScanGeometry`: the directions the beams of a forward-facing range scan point.

A map is a rectangle of square cells, each free or blocked. Cells are named row first, matching the
way they are stored: cell `(row, column)` covers the world square starting at
`origin + [column · resolution, row · resolution]`, so `origin` is the lowest corner of cell
`(0, 0)`. A point sitting exactly on the edge between two cells belongs to the higher one.

Casting a beam walks it one cell at a time and reports where it first crosses into a blocked cell.
That last part matters: the distance is the crossing point, not the middle of the cell, so a wall
lining up with a cell edge reads exactly rather than rounded. A beam starting on a blocked cell
reads zero, and one starting off the map is walked from where it first enters, with the distance
still measured from where it started.

The rasterizing helpers draw outlines, not fills — `occupy_polyline` around a closed shape leaves
the space inside it free. They sample well inside a cell width, so a wall they draw has no gap a
beam can slip through. Because anything drawn is a cell thick, the face a beam meets can sit a cell
either side of the line that drew it; that is the accuracy a scan against a map can be trusted to.

`ScanGeometry` is the shape of a scan rather than a sensor: where each beam points and how far it
can see, with nothing about noise or how often it reads. Beam `0` sits at the right edge of the arc
and angles grow to the left, matching the order a
[ROS `LaserScan`](https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html) carries
its ranges in. [`FollowTheGap`](control.md) numbers its beams through the same formula, so a scan
and the steering worked out from it always agree beam for beam.

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

// A robot a metre in, facing along the row it sits on, meets the pillar before the wall.
let robot = [1.0, 1.5];
let along_the_row = 0.0;
let maximum_range = 5.0;
let pillar_face = pillar_centre[0] - pillar_radius - robot[0];
let distance = room.cast_ray(robot, along_the_row, maximum_range);
assert!(distance.is_some_and(|met| (met - pillar_face).abs() <= 2.0 * cell_size));

// Behind it the room is open all the way to the map edge, so the beam meets nothing.
let back_along_the_row = core::f64::consts::PI;
assert!(room.cast_ray(robot, back_along_the_row, maximum_range).is_none());

// Casting a whole scan gives the ranges a perfect sensor would report. A beam that meets nothing
// reads the scan's maximum range.
const NUM_BEAMS: usize = 5;
let quarter_turn = core::f64::consts::FRAC_PI_2;
let scan: ScanGeometry<NUM_BEAMS> = ScanGeometry::try_new(quarter_turn, maximum_range)?;
let pose = SE2::from_parts(SO2::from_angle(along_the_row), Vector2D::new(robot));
let ranges = room.cast_scan(pose, &scan);

// The middle beam meets the pillar square on, so it reads the shortest of the five.
assert!(ranges.iter().all(|&range| range >= ranges[2]));
# Ok::<(), multicalc::CalcError>(())
```

A sensor with a blind spot right in front of it carries that as its minimum range, and
`range_is_valid` says whether a reading is one it could actually have taken — a real distance,
neither closer than the blind spot nor further than it can see. A beam that met nothing reads as
infinity, which is not a distance and so is not valid.

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

// And a direction leads back to the beam looking that way.
let straight_ahead = 0.0;
assert_eq!(scan.nearest_beam_index(straight_ahead), Some(2));
# Ok::<(), multicalc::CalcError>(())
```

## A map that fits on a small board

`DynamicOccupancyGrid` holds its cells on the heap, one byte each, so a 20 m by 15 m floor at 5 cm
cells is 120 000 cells and about 120 KB. `OccupancyGrid` packs one *bit* per cell into `u32` words
sized at compile time, and needs no heap at all.

`WORDS_PER_ROW` must be at least `COLUMNS.div_ceil(32)`. It appears in every user's type signature
rather than being computed, because stable Rust has no `generic_const_exprs`; `try_new` checks it
and reports `MappingError::WordsPerRowTooSmall` if it is short. Memory is
`ROWS · WORDS_PER_ROW · 4` bytes.

```rust
use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid, OccupancyMap};

// A 6.4 m square at 5 cm cells: 128 by 128, and 128 / 32 = 4 words to a row.
let cell_size = 0.05_f64;
let mut room: OccupancyGrid<128, 128, 4> = OccupancyGrid::try_new(cell_size, [0.0, 0.0])?;

// 128 rows of 4 words of 4 bytes: 2 KB of cells, against the 16 KB a byte-per-cell grid needs.
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

Because `OccupancyMap` is a trait, a map still does not have to be one of these: implement the five
questions over whatever storage suits — a fixed array, or cells read from flash — and the ray
casting, the rasterizing helpers and the scan helpers all come with it.

## Building a map from scans

`LogOddsGrid` is the other direction: rather than reading a map you already have, it builds one from
what a sensor sees. Each cell holds its log-odds of being blocked, `l = log(p / (1 − p))`, in `i8`
fixed point. The Bayesian update collapses to addition in log-odds, which is why it is the standard
representation, and `l = 0` is exactly "unknown", so a new grid costs nothing to initialise.

Every cell a beam crosses moves toward free and the cell it stops in moves toward blocked, each held
inside a clamp. **The clamp is what lets a cell recover**: a person who walks through and leaves does
not become a permanent wall.

```rust
use multicalc::mapping::{CellState, LogOddsGrid, OccupancyMap, ScanGeometry};
use multicalc::{SE2, SO2, Vector2D};

// A 4 m square at 10 cm cells, entirely unobserved to begin with.
let mut belief: LogOddsGrid<40, 40> = LogOddsGrid::try_new(0.1, [0.0, 0.0])?;
assert_eq!(belief.cell_state(20, 20), CellState::Unknown);

// A narrow scan facing east from the middle, reading a wall one metre off.
let scan: ScanGeometry<3> = ScanGeometry::try_new(0.02, 4.0)?;
let pose = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([2.05, 2.05]));
for _ in 0..4 {
    belief.integrate_scan(pose, &scan, &[1.0; 3]);
}

// The cell the beams stopped in is blocked; the ones they crossed are free; the rest is still
// unobserved.
assert_eq!(belief.cell_state(20, 30), CellState::Occupied);
assert_eq!(belief.cell_state(20, 25), CellState::Free);
assert_eq!(belief.cell_state(35, 35), CellState::Unknown);

// The obstacle leaves, and the beams now pass through where it stood. Twenty scans later the cell
// has recovered.
for _ in 0..20 {
    belief.integrate_scan(pose, &scan, &[3.0; 3]);
}
assert_eq!(belief.cell_state(20, 30), CellState::Free);
# Ok::<(), multicalc::CalcError>(())
```

`cell_state` is a *provided* method on `OccupancyMap`, so nothing that already implements the trait
had to change: the default derives from `is_occupied` and never reports `Unknown`. A belief map
overrides it, and a cell off a belief grid reads `Unknown` where a plain occupancy map reads free.
That difference is what stops a planner routing confidently through space no sensor has seen.

Work per scan is bounded by `NUM_BEAMS · maximum_range / resolution` cells.

## Casting a scan

`cast_scan` gives the range each beam of a scan would read from an `SE2` pose, and `scan_endpoints`
gives where each one ends in world coordinates. A beam that meets nothing reads the scan's maximum
range.

```rust
use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid, OccupancyMap, ScanGeometry};
use multicalc::{SE2, SO2, Vector2D};

let mut room: OccupancyGrid<40, 40, 2> = OccupancyGrid::try_new(0.1, [0.0, 0.0])?;
let wall = [[3.0, 0.5], [3.0, 3.5]];
room.occupy_polyline(&wall, false);

let scan: ScanGeometry<3> = ScanGeometry::try_new(core::f64::consts::FRAC_PI_2, 4.0)?;
let facing_east = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([2.0, 2.0]));

// The middle beam meets the wall a metre away; the endpoints are where the beams stop.
let ranges = room.cast_scan(facing_east, &scan);
assert!((ranges[1] - 1.0).abs() <= 0.1);

let endpoints = room.scan_endpoints(facing_east, &scan);
assert!((endpoints[1][0] - 3.0).abs() <= 0.1);
# Ok::<(), multicalc::CalcError>(())
```

A reading that met nothing is not a distance, and `range_is_valid` is what separates the two. Note
that `cast_scan` reports the maximum range there rather than an infinity, so a caller that has to
tell "saw nothing" from "saw something at the limit" should keep casting beam by beam with
`cast_ray`, which answers `None`.

## Distance and cost

`DistanceField` holds every cell's distance to the nearest blocked cell. It is the exact Euclidean
transform, by Felzenszwalb–Huttenlocher: two separable one-dimensional passes taking the lower
envelope of parabolas. O(cells), no priority queue, no approximation — and far cheaper than the fan
of ray casts it replaces. Building it is a design-time or low-rate operation, never per-tick work;
the queries are what a loop calls.

`distance_at` blends bilinearly over the four surrounding cell centres, so it is smooth in the
position and **exactly differentiable by swapping `T` for an autodiff scalar** — which is what would
let a gradient-based trajectory refinement drop an obstacle cost into `LevenbergMarquardt` with no
hand-derived Jacobian.

```rust
use multicalc::mapping::{
    CostGrid, DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid,
};

// A 2 m square at 10 cm cells with a wall along one row.
let mut room: OccupancyGrid<20, 20, 1> = OccupancyGrid::try_new(0.1, [0.0, 0.0])?;
for column in 0..20 {
    room.set_cell(2, column, true);
}

// The workspace is caller-owned, and must hold the longest span plus one.
let mut workspace: DistanceTransformWorkspace<21> = DistanceTransformWorkspace::new();
let field: DistanceField<20, 20> = DistanceField::try_build(&room, &mut workspace)?;

// Zero on the wall, and rising by a cell a row away from it.
assert_eq!(field.distance_of(2, 10), Some(0.0));
assert!(field.distance_of(7, 10).is_some_and(|d| (d - 0.5).abs() < 1e-12));

// The gradient well above the wall is its normal: straight up, unit length.
let above = field.geometry().center_of(10, 10).expect("cell");
let gradient = field.gradient_at(above).expect("inside");
assert!(gradient[0].abs() < 1e-6 && (gradient[1] - 1.0).abs() < 1e-6);
# Ok::<(), multicalc::CalcError>(())
```

`CostGrid` turns that field into what a planner charges for entering each cell, by the nav2
inflation formulation. `inscribed_radius` is the radius of the largest circle inside the robot's
footprint, so anything within it is lethal; `inflation_radius` sets how far cost spreads beyond that
and `cost_scaling_factor` how steeply it decays.

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

// No part of the robot may enter the wall or the band around it; far away it costs nothing.
assert_eq!(costmap.cost_of(2, 10), Some(CostGrid::<20, 20>::LETHAL));
assert_eq!(costmap.cost_of(19, 10), Some(0));

// Cost falls away from the wall and never rises.
let near = costmap.cost_of(5, 10).expect("cell");
let far = costmap.cost_of(9, 10).expect("cell");
assert!(near > far);
# Ok::<(), multicalc::CalcError>(())
```

The adapter that lets a planner *read* a costmap — `CostmapCost` — lives in
[`planning`](planning.md), which owns the traversal-cost trait. Putting it beside `CostGrid` here
would make the two modules mutually recursive.

## Localizing with a likelihood field

Given a distance field, a particle filter can score a scan without casting a ray per beam per
particle. `LikelihoodFieldModel` projects each beam's endpoint from the guessed pose and looks up
how far it landed from the nearest obstacle: an endpoint on a wall scores well, one in open space
badly. One interpolated lookup replaces one DDA walk, and the score is smoother in the pose than a
beam model's, whose likelihood is jagged because it depends on map resolution.

It ignores occlusion, so a pose can score highly by seeing through a wall. Keep `BeamModel` where
that matters.

```rust
use multicalc::estimation::{
    BeamModel, InitialParticleCloud, LikelihoodFieldModel, MonteCarloLocalizer,
};
use multicalc::mapping::{
    DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid, OccupancyMap,
    ScanGeometry,
};
use multicalc::{SE2, SO2, Vector2D};

// A 6 m square room at 20 cm cells, walled all the way round.
let mut room: OccupancyGrid<30, 30, 1> = OccupancyGrid::try_new(0.2, [0.0, 0.0])?;
room.occupy_polyline(&[[0.5, 0.5], [5.5, 0.5], [5.5, 5.5], [0.5, 5.5]], true);

let mut workspace: DistanceTransformWorkspace<31> = DistanceTransformWorkspace::new();
let field: DistanceField<30, 30> = DistanceField::try_build(&room, &mut workspace)?;

// The robot is really here; the localizer is only told roughly where to look.
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

Errors are [`MappingError`](error-handling.md): a grid with no cells, a cell size that is zero or
negative, a words-per-row capacity below what the columns need, a transform workspace shorter than
the grid's longest span, belief settings that are not ordered as required, an inscribed radius above
the inflation radius, a scan with fewer than two beams, a field of view outside `(0, 2π]`, and so on.

## Next

[Planning](planning.md) is what consumes all of this: a search over the map, charged by the costmap,
producing a `PolylinePath` for a smoother and a path follower.
