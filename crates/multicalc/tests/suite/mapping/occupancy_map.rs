//! The map traits themselves: what any map answers, and the ray casting and rasterizing every map
//! inherits. Run against a plain fixed-size map so they hold without a heap.

use multicalc::Numeric;
use multicalc::mapping::{CellState, MutableOccupancyMap, OccupancyMap, ScanGeometry};
use multicalc::{SE2, SO2, Vector2D};

/// A map of `COLUMNS` by `ROWS` cells, laid out row by row.
struct TestMap<const COLUMNS: usize, const ROWS: usize> {
    resolution: f64,
    origin: [f64; 2],
    cells: [[bool; COLUMNS]; ROWS],
}

impl<const COLUMNS: usize, const ROWS: usize> TestMap<COLUMNS, ROWS> {
    fn new(resolution: f64, origin: [f64; 2]) -> Self {
        TestMap {
            resolution,
            origin,
            cells: [[false; COLUMNS]; ROWS],
        }
    }
}

impl<const COLUMNS: usize, const ROWS: usize> OccupancyMap for TestMap<COLUMNS, ROWS> {
    fn columns(&self) -> usize {
        COLUMNS
    }
    fn rows(&self) -> usize {
        ROWS
    }
    fn resolution(&self) -> f64 {
        self.resolution
    }
    fn origin(&self) -> [f64; 2] {
        self.origin
    }
    fn is_occupied(&self, row: usize, column: usize) -> bool {
        self.cells
            .get(row)
            .and_then(|row| row.get(column))
            .copied()
            .unwrap_or(false)
    }
}

impl<const COLUMNS: usize, const ROWS: usize> MutableOccupancyMap for TestMap<COLUMNS, ROWS> {
    fn set_cell(&mut self, row: usize, column: usize, occupied: bool) {
        if let Some(cell) = self.cells.get_mut(row).and_then(|row| row.get_mut(column)) {
            *cell = occupied;
        }
    }
    fn clear(&mut self) {
        self.cells = [[false; COLUMNS]; ROWS];
    }
}

/// A 15 by 12 map of one-metre cells, the shape the demo grid tests used.
fn wide_map() -> TestMap<15, 12> {
    TestMap::new(1.0, [-5.0, -5.25])
}

#[test]
fn cell_of_finds_the_cell_holding_a_point() {
    let map = wide_map();
    assert_eq!(map.cell_of([-4.5, -4.5]), Some((0, 0)));
    // Row first: half a cell up is the next row, half a cell across is the next column.
    assert_eq!(map.cell_of([-4.5, -4.0]), Some((1, 0)));
    assert_eq!(map.cell_of([-4.0, -4.5]), Some((0, 1)));
}

#[test]
fn cell_of_rejects_points_off_the_map() {
    let map = wide_map();
    assert_eq!(map.cell_of([-5.5, 0.0]), None, "left of the first column");
    assert_eq!(map.cell_of([0.0, -6.0]), None, "below the first row");
    assert_eq!(map.cell_of([10.1, 0.0]), None, "past the last column");
    assert_eq!(map.cell_of([0.0, 6.8]), None, "past the last row");
}

#[test]
fn a_point_on_a_cell_edge_belongs_to_the_higher_cell() {
    let map: TestMap<4, 4> = TestMap::new(0.5, [0.0, 0.0]);
    // The edge between cells (0, 0) and (0, 1) sits at x = 0.5.
    assert_eq!(map.cell_of([0.49, 0.25]), Some((0, 0)));
    assert_eq!(map.cell_of([0.5, 0.25]), Some((0, 1)));
}

#[test]
fn a_clear_map_stops_no_beam() {
    let map = wide_map();
    assert_eq!(map.cast_ray([0.0, 0.0], 0.0, 20.0), None);
}

#[test]
fn a_beam_reads_the_exact_distance_to_a_wall() {
    let mut map = wide_map();
    let blocked_column = 10;
    for row in 0..12 {
        map.set_cell(row, blocked_column, true);
    }
    // Standing in the middle of cell (6, 0), firing along the row.
    let start = [-4.5, 1.25];
    let distance = map.cast_ray(start, 0.0, 20.0);
    // The wall's near face is where its column begins.
    let expected = -5.0 + blocked_column as f64 - start[0];
    assert!(
        distance.is_some_and(|met| (met - expected).abs() < 1e-12),
        "{distance:?}, expected {expected}"
    );
}

#[test]
fn a_beam_starting_off_the_map_is_walked_from_where_it_enters() {
    let mut map = wide_map();
    let blocked_column = 10;
    for row in 0..12 {
        map.set_cell(row, blocked_column, true);
    }
    // Two metres left of the map, aimed into it along the same row as above.
    let start = [-7.0, 1.25];
    let distance = map.cast_ray(start, 0.0, 20.0);
    let expected = -5.0 + blocked_column as f64 - start[0];
    assert!(
        distance.is_some_and(|met| (met - expected).abs() < 1e-12),
        "the distance is measured from where the beam started, not where it entered"
    );
}

#[test]
fn a_beam_aimed_away_from_the_map_meets_nothing() {
    let map = wide_map();
    assert_eq!(
        map.cast_ray([-7.0, 1.25], core::f64::consts::PI, 20.0),
        None
    );
}

#[test]
fn a_beam_stops_at_its_maximum_range() {
    let mut map = wide_map();
    for row in 0..12 {
        map.set_cell(row, 10, true);
    }
    let start = [-4.5, 1.25];
    assert!(map.cast_ray(start, 0.0, 20.0).is_some());
    assert_eq!(map.cast_ray(start, 0.0, 1.0), None, "too short to reach it");
}

#[test]
fn a_beam_running_parallel_to_an_axis_crosses_no_edges_on_it() {
    let mut map = wide_map();
    map.set_cell(5, 5, true);
    // Straight up a column that holds nothing: the sideways stride is infinite.
    let clear_column_start = [-4.5, -4.5];
    assert_eq!(
        map.cast_ray(clear_column_start, core::f64::consts::FRAC_PI_2, 20.0),
        None
    );
}

#[test]
fn a_beam_starting_on_a_blocked_cell_reads_zero() {
    let mut map = wide_map();
    map.set_cell(0, 0, true);
    assert_eq!(map.cast_ray([-4.5, -4.5], 0.0, 20.0), Some(0.0));
}

#[test]
fn a_map_with_no_cells_stops_no_beam() {
    let empty: TestMap<0, 0> = TestMap::new(1.0, [0.0, 0.0]);
    assert_eq!(empty.cast_ray([0.0, 0.0], 0.0, 10.0), None);
    assert_eq!(empty.cell_of([0.0, 0.0]), None);
}

#[test]
fn a_closed_polyline_leaves_no_gap_a_beam_can_slip_through() {
    let mut map: TestMap<40, 40> = TestMap::new(0.1, [0.0, 0.0]);
    let corners = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
    map.occupy_polyline(&corners, true);
    let middle = [2.0, 2.0];
    for step in 0..32 {
        let bearing = core::f64::consts::TAU * step as f64 / 32.0;
        assert!(
            map.cast_ray(middle, bearing, 4.0).is_some(),
            "beam {step} slipped out of a closed box"
        );
    }
    // It is an outline, not a fill.
    assert!(
        map.cell_of(middle)
            .is_some_and(|(row, column)| !map.is_occupied(row, column))
    );
}

#[test]
fn an_open_polyline_leaves_its_last_edge_undrawn() {
    let mut map: TestMap<40, 40> = TestMap::new(0.1, [0.0, 0.0]);
    let corners = [[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
    map.occupy_polyline(&corners, false);
    // The edge from the last corner back to the first is the left wall, so a beam escapes west.
    let middle = [2.0, 2.0];
    assert_eq!(map.cast_ray(middle, core::f64::consts::PI, 4.0), None);
    assert!(
        map.cast_ray(middle, 0.0, 4.0).is_some(),
        "the right wall is still there"
    );
}

#[test]
fn a_single_point_polyline_marks_one_cell() {
    let mut map: TestMap<40, 40> = TestMap::new(0.1, [0.0, 0.0]);
    map.occupy_polyline(&[[2.0, 2.0]], true);
    assert!(map.is_occupied(20, 20));
}

#[test]
fn a_circle_marks_its_rim_and_not_its_middle() {
    let mut map: TestMap<40, 40> = TestMap::new(0.1, [0.0, 0.0]);
    let centre = [2.0, 2.0];
    let radius = 1.0;
    map.occupy_circle(centre, radius);
    assert!(!map.is_occupied(20, 20), "the middle should stay free");
    for step in 0..32 {
        let bearing = core::f64::consts::TAU * step as f64 / 32.0;
        let distance = map.cast_ray(centre, bearing, 4.0);
        assert!(
            distance.is_some_and(|met| (met - radius).abs() <= 2.0 * map.resolution()),
            "beam {step} read {distance:?} instead of about {radius}"
        );
    }
}

#[test]
fn a_large_circle_blocks_beams_through_its_east_rim() {
    let resolution = 1.0;
    let radius = 2_000.0;
    let centre = [0.5, 0.5];
    // Keep the test map small by placing a window near the circle's eastern rim in world
    // coordinates. Rasterization and ray casting still exercise the large-circle geometry
    // without allocating a map large enough to contain the whole circle.
    let mut map: TestMap<10, 32> = TestMap::new(resolution, [1_996.0, -2.0]);

    map.occupy_circle(centre, radius);

    const BEARING_DIVISIONS: usize = 8_192;
    for beam in 0..16 {
        // Offset from angle zero, which is also the circle rasterizer's first sample, so these
        // rays probe the gaps between sampled rim points rather than that convenient sample.
        let bearing = core::f64::consts::TAU * (beam as f64 + 0.5) / BEARING_DIVISIONS as f64;
        let distance = map.cast_ray(centre, bearing, radius + 2.0);
        assert!(
            distance.is_some_and(|met| (met - radius).abs() <= 2.0 * map.resolution()),
            "beam {beam} read {distance:?} instead of about {radius}"
        );
    }
}

#[test]
fn a_non_finite_circle_radius_leaves_the_map_unchanged() {
    let mut map: TestMap<40, 40> = TestMap::new(0.1, [0.0, 0.0]);
    for radius in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        map.occupy_circle([2.0, 2.0], radius);
    }
    for row in 0..40 {
        for column in 0..40 {
            assert!(!map.is_occupied(row, column));
        }
    }
}

#[test]
fn clearing_frees_every_cell() {
    let mut map: TestMap<40, 40> = TestMap::new(0.1, [0.0, 0.0]);
    map.occupy_circle([2.0, 2.0], 1.0);
    assert!(map.cast_ray([2.0, 2.0], 0.0, 4.0).is_some());
    map.clear();
    assert!(map.cast_ray([2.0, 2.0], 0.0, 4.0).is_none());
}

#[test]
fn marking_a_cell_off_the_map_does_nothing() {
    let mut map: TestMap<4, 4> = TestMap::new(0.5, [0.0, 0.0]);
    map.occupy_point([100.0, 100.0]);
    map.occupy_point([-100.0, 0.0]);
    for row in 0..4 {
        for column in 0..4 {
            assert!(!map.is_occupied(row, column), "({row}, {column})");
        }
    }
}

/// The same wall distance at `f32`, where the walk accumulates more rounding than at `f64`.
#[test]
fn ray_casting_holds_at_f32() {
    struct NarrowMap {
        cells: [[bool; 16]; 16],
    }

    impl OccupancyMap<f32> for NarrowMap {
        fn columns(&self) -> usize {
            16
        }
        fn rows(&self) -> usize {
            16
        }
        fn resolution(&self) -> f32 {
            0.25
        }
        fn origin(&self) -> [f32; 2] {
            [0.0, 0.0]
        }
        fn is_occupied(&self, row: usize, column: usize) -> bool {
            self.cells
                .get(row)
                .and_then(|row| row.get(column))
                .copied()
                .unwrap_or(false)
        }
    }

    let blocked_column = 5;
    let cells = core::array::from_fn(|_| core::array::from_fn(|column| column == blocked_column));
    let map = NarrowMap { cells };
    let start = [0.5_f32, 0.5];
    let expected = blocked_column as f32 * map.resolution() - start[0];
    let distance = map.cast_ray(start, 0.0, 4.0);
    assert!(
        distance.is_some_and(|met| (met - expected).abs() < 1e-4),
        "{distance:?}, expected {expected}"
    );
}

// ---- scan casting ----

/// A 20 by 20 map of quarter-metre cells with a wall running up column 12.
fn walled_map() -> TestMap<20, 20> {
    let mut map = TestMap::<20, 20>::new(0.25, [0.0, 0.0]);
    for row in 0..20 {
        map.set_cell(row, 12, true);
    }
    map
}

#[test]
fn cast_scan_matches_per_beam_cast_ray_f64() {
    let map = walled_map();
    let scan: ScanGeometry<32> = ScanGeometry::try_new(core::f64::consts::PI, 6.0).unwrap();

    for (x, y, heading) in [(1.0, 1.0, 0.0), (2.0, 3.5, 0.9), (0.5, 4.0, -2.1)] {
        let pose = SE2::from_parts(SO2::from_angle(heading), Vector2D::new([x, y]));
        let by_helper = map.cast_scan(pose, &scan);
        let by_hand: [f64; 32] = core::array::from_fn(|beam| {
            scan.beam_angle(beam)
                .and_then(|offset| map.cast_ray([x, y], heading + offset, scan.maximum_range()))
                .unwrap_or(scan.maximum_range())
        });
        // The pose carries a rotation, not an angle, so the heading comes back through `atan2`
        // and the two agree to rounding rather than bit for bit.
        for (helper, hand) in by_helper.iter().zip(by_hand) {
            assert!((helper - hand).abs() < 1e-12, "{helper} against {hand}");
        }
    }
}

#[test]
fn cast_scan_matches_per_beam_cast_ray_f32() {
    /// The same walled shape at single precision.
    struct WalledMap;

    impl OccupancyMap<f32> for WalledMap {
        fn columns(&self) -> usize {
            20
        }
        fn rows(&self) -> usize {
            20
        }
        fn resolution(&self) -> f32 {
            0.25
        }
        fn origin(&self) -> [f32; 2] {
            [0.0, 0.0]
        }
        fn is_occupied(&self, _row: usize, column: usize) -> bool {
            column == 12
        }
    }

    let map = WalledMap;
    let scan: ScanGeometry<32, f32> = ScanGeometry::try_new(core::f32::consts::PI, 6.0).unwrap();
    let position = [1.0_f32, 1.5];
    let heading = 0.4_f32;
    let pose = SE2::from_parts(SO2::from_angle(heading), Vector2D::new(position));

    for (beam, range) in map.cast_scan(pose, &scan).iter().enumerate() {
        let offset = scan.beam_angle(beam).unwrap();
        let expected = map
            .cast_ray(position, heading + offset, scan.maximum_range())
            .unwrap_or(scan.maximum_range());
        assert!((range - expected).abs() < 1e-4);
    }
}

#[test]
fn cast_scan_reports_maximum_range_when_clear_f64() {
    let map = TestMap::<20, 20>::new(0.25, [0.0, 0.0]);
    let scan: ScanGeometry<16> = ScanGeometry::try_new(core::f64::consts::PI, 6.0).unwrap();
    let pose = SE2::from_parts(SO2::from_angle(0.3), Vector2D::new([2.5, 2.5]));

    for range in map.cast_scan(pose, &scan) {
        assert_eq!(range, scan.maximum_range());
    }
}

#[test]
fn scan_endpoints_lie_on_their_bearings_f64() {
    let map = walled_map();
    let scan: ScanGeometry<16> = ScanGeometry::try_new(core::f64::consts::PI, 6.0).unwrap();
    let heading = 0.7;
    let position = [1.0, 2.0];
    let pose = SE2::from_parts(SO2::from_angle(heading), Vector2D::new(position));

    let ranges = map.cast_scan(pose, &scan);
    let endpoints = map.scan_endpoints(pose, &scan);

    for (beam, endpoint) in endpoints.iter().enumerate() {
        let separation = [endpoint[0] - position[0], endpoint[1] - position[1]];
        let distance = separation[0].hypot(separation[1]);
        assert!((distance - ranges[beam]).abs() < 1e-12);

        let bearing = separation[1].atan2(separation[0]);
        let expected = (heading + scan.beam_angle(beam).unwrap()).wrap_to_pi();
        assert!((bearing.wrap_to_pi() - expected).abs() < 1e-12);
    }
}

#[test]
fn beams_yields_num_beams_angles_f64() {
    let scan: ScanGeometry<9> = ScanGeometry::try_new(core::f64::consts::FRAC_PI_2, 4.0).unwrap();

    let angles: Vec<f64> = scan.beams().collect();
    assert_eq!(angles.len(), 9);
    for (index, angle) in angles.iter().enumerate() {
        assert_eq!(Some(*angle), scan.beam_angle(index));
    }
}

// ---- cell state ----

#[test]
fn default_cell_state_follows_is_occupied_f64() {
    /// A map answering only the five required questions, so this is the standing proof that the
    /// trait's additions stayed provided rather than required.
    struct BareMap;

    impl OccupancyMap for BareMap {
        fn columns(&self) -> usize {
            4
        }
        fn rows(&self) -> usize {
            4
        }
        fn resolution(&self) -> f64 {
            0.5
        }
        fn origin(&self) -> [f64; 2] {
            [0.0, 0.0]
        }
        fn is_occupied(&self, row: usize, column: usize) -> bool {
            row < 4 && row == column
        }
    }

    let map = BareMap;
    for row in 0..4 {
        for column in 0..4 {
            let expected = if row == column {
                CellState::Occupied
            } else {
                CellState::Free
            };
            assert_eq!(map.cell_state(row, column), expected);
        }
    }

    // Off the grid reads free here, as `is_occupied` does. Only a belief map says `Unknown`.
    assert_eq!(map.cell_state(4, 0), CellState::Free);
    assert_eq!(map.cell_state(0, 4), CellState::Free);
    assert_eq!(map.cell_state(usize::MAX, usize::MAX), CellState::Free);
}
