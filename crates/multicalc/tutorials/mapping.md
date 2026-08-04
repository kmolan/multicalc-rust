# Mapping

The map a robot measures against, and how far a beam travels across it.

- `OccupancyMap`: what any map must answer — its size, where it sits in the world, whether a cell is
  blocked, and how far a beam travels before it meets something.
- `MutableOccupancyMap`: marking cells from world geometry — single points, joined-up lines, and
  circles.
- `OccupancyGrid` (`alloc` only, as `DynamicOccupancyGrid`): a grid of cells sized when the program
  runs.
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

// Casting a whole scan gives the ranges a perfect sensor would report.
const NUM_BEAMS: usize = 5;
let quarter_turn = core::f64::consts::FRAC_PI_2;
let scan: ScanGeometry<NUM_BEAMS> = ScanGeometry::try_new(quarter_turn, maximum_range)?;
let ranges: [f64; NUM_BEAMS] = core::array::from_fn(|beam| {
    match scan.beam_angle(beam) {
        Some(offset) => room
            .cast_ray(robot, along_the_row + offset, scan.maximum_range())
            .unwrap_or(f64::INFINITY),
        None => f64::INFINITY,
    }
});

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

## Fitting on a small board

`DynamicOccupancyGrid` holds its cells on the heap, one byte each, so a 20 m by 15 m floor at 5 cm
cells is 120 000 cells and about 120 KB. That is more than a small board has to spare. Because
`OccupancyMap` is a trait, a map does not have to be one of these: implement the five questions over
whatever storage suits — a fixed array, bit-packed words, or cells read from flash — and the ray
casting and the rasterizing helpers come with it.

Errors are [`MappingError`](error-handling.md): a grid with no cells, a cell size that is zero or
negative, a scan with fewer than two beams, a field of view outside `(0, 2π]`, and so on.
