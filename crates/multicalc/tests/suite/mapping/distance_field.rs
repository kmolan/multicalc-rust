//! The distance field, pinned against a brute-force nearest-obstacle search.
//!
//! The oracle is an independent O(cells²) minimum over every obstacle cell: it shares no loop
//! shape, data structure or termination argument with the separable transform under test.

use multicalc::error::MappingError;
use multicalc::mapping::{
    CellState, DistanceField, DistanceTransformWorkspace, LogOddsGrid, MutableOccupancyMap,
    OccupancyGrid, OccupancyMap, ScanGeometry,
};
use multicalc::scalar::{Numeric, Primal};
use multicalc::{SE2, SO2, Vector2D};

/// A map of `ROWS` by `COLUMNS` cells whose blocked cells are given outright.
struct PlainMap<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal> {
    cells: [[bool; COLUMNS]; ROWS],
    resolution: T,
}

impl<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal> OccupancyMap<T>
    for PlainMap<ROWS, COLUMNS, T>
{
    fn columns(&self) -> usize {
        COLUMNS
    }
    fn rows(&self) -> usize {
        ROWS
    }
    fn resolution(&self) -> T {
        self.resolution
    }
    fn origin(&self) -> [T; 2] {
        [T::ZERO, T::ZERO]
    }
    fn is_occupied(&self, row: usize, column: usize) -> bool {
        self.cells
            .get(row)
            .and_then(|row_cells| row_cells.get(column))
            .copied()
            .unwrap_or(false)
    }
}

/// The distance from every cell to the nearest blocked cell, by direct search.
fn brute_force_distances<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal>(
    map: &PlainMap<ROWS, COLUMNS, T>,
) -> [[T; COLUMNS]; ROWS] {
    let mut expected = [[T::INFINITY; COLUMNS]; ROWS];
    for (row, row_expected) in expected.iter_mut().enumerate() {
        for (column, cell) in row_expected.iter_mut().enumerate() {
            let mut nearest = T::INFINITY;
            for other_row in 0..ROWS {
                for other_column in 0..COLUMNS {
                    if !map.is_occupied(other_row, other_column) {
                        continue;
                    }
                    let down = T::from_usize(row.abs_diff(other_row));
                    let across = T::from_usize(column.abs_diff(other_column));
                    nearest = nearest.min(down.hypot(across) * map.resolution);
                }
            }
            *cell = nearest;
        }
    }
    expected
}

/// A deterministic obstacle pattern at roughly the given density, so a failure is reproducible.
fn scattered_map<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal>(
    seed: u64,
    density_in_percent: u64,
    resolution: T,
) -> PlainMap<ROWS, COLUMNS, T> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut cells = [[false; COLUMNS]; ROWS];
    for row_cells in cells.iter_mut() {
        for cell in row_cells.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *cell = (state >> 33) % 100 < density_in_percent;
        }
    }
    PlainMap { cells, resolution }
}

fn assert_matches_brute_force<T: Numeric + Primal>(tolerance: T) {
    let mut workspace: DistanceTransformWorkspace<13, T> = DistanceTransformWorkspace::new();

    for seed in 0..20u64 {
        let density = 10 + (seed % 4) * 10;
        let map = scattered_map::<12, 12, T>(seed, density, T::from_f64(0.25));

        // A map with no obstacle at all has nothing to be a distance to; skip it.
        let any_obstacle = (0..12).any(|row| (0..12).any(|column| map.is_occupied(row, column)));
        if !any_obstacle {
            continue;
        }

        let field: DistanceField<12, 12, T> =
            DistanceField::try_build(&map, &mut workspace).unwrap();
        let expected = brute_force_distances(&map);

        for row in 0..12 {
            for column in 0..12 {
                let actual = field.distance_of(row, column).unwrap();
                let wanted = expected
                    .get(row)
                    .and_then(|row_expected| row_expected.get(column))
                    .copied()
                    .unwrap();
                assert!(
                    (actual - wanted).abs() <= tolerance,
                    "seed {seed}, cell ({row}, {column}): {actual:?} against {wanted:?}"
                );
            }
        }
    }
}

#[test]
fn matches_brute_force_nearest_obstacle_f64() {
    assert_matches_brute_force::<f64>(1e-12);
}

#[test]
fn matches_brute_force_nearest_obstacle_f32() {
    assert_matches_brute_force::<f32>(1e-4);
}

#[test]
fn try_build_rejects_a_shape_mismatch() {
    let map: PlainMap<8, 8, f64> = PlainMap {
        cells: [[false; 8]; 8],
        resolution: 0.1,
    };
    let mut workspace: DistanceTransformWorkspace<16> = DistanceTransformWorkspace::new();

    assert_eq!(
        DistanceField::<9, 8, f64>::try_build(&map, &mut workspace).err(),
        Some(MappingError::GridShapeMismatch)
    );
    assert_eq!(
        DistanceField::<8, 9, f64>::try_build(&map, &mut workspace).err(),
        Some(MappingError::GridShapeMismatch)
    );
}

#[test]
fn try_build_rejects_a_short_workspace() {
    let map: PlainMap<8, 12, f64> = PlainMap {
        cells: [[false; 12]; 8],
        resolution: 0.1,
    };

    // The longest span is 12, so the envelope needs 13 slots.
    let mut too_short: DistanceTransformWorkspace<12> = DistanceTransformWorkspace::new();
    assert_eq!(
        DistanceField::<8, 12, f64>::try_build(&map, &mut too_short).err(),
        Some(MappingError::WorkspaceTooSmall)
    );

    let mut just_enough: DistanceTransformWorkspace<13> = DistanceTransformWorkspace::new();
    assert!(DistanceField::<8, 12, f64>::try_build(&map, &mut just_enough).is_ok());
    assert_eq!(just_enough.capacity(), 13);
}

#[test]
fn distance_is_zero_on_an_obstacle_f64() {
    let mut room: OccupancyGrid<9, 9, 1> = OccupancyGrid::try_new(0.2, [0.0, 0.0]).unwrap();
    for column in 0..9 {
        room.set_cell(4, column, true);
    }
    let mut workspace: DistanceTransformWorkspace<10> = DistanceTransformWorkspace::new();
    let field: DistanceField<9, 9> = DistanceField::try_build(&room, &mut workspace).unwrap();

    for column in 0..9 {
        assert_eq!(field.distance_of(4, column), Some(0.0));
    }
}

#[test]
fn a_single_obstacle_gives_exact_euclidean_distance_f64() {
    let mut room: OccupancyGrid<9, 9, 1> = OccupancyGrid::try_new(0.2, [0.0, 0.0]).unwrap();
    room.set_cell(4, 4, true);
    let mut workspace: DistanceTransformWorkspace<10> = DistanceTransformWorkspace::new();
    let field: DistanceField<9, 9> = DistanceField::try_build(&room, &mut workspace).unwrap();

    for row in 0..9 {
        for column in 0..9 {
            let down = (row as f64) - 4.0;
            let across = (column as f64) - 4.0;
            let expected = down.hypot(across) * 0.2;
            let actual = field.distance_of(row, column).unwrap();
            assert!(
                (actual - expected).abs() < 1e-12,
                "cell ({row}, {column}): {actual} against {expected}"
            );
        }
    }
}

#[test]
fn an_empty_map_is_infinitely_far_from_anything_f64() {
    let room: OccupancyGrid<9, 9, 1> = OccupancyGrid::try_new(0.2, [0.0, 0.0]).unwrap();
    let mut workspace: DistanceTransformWorkspace<10> = DistanceTransformWorkspace::new();
    let field: DistanceField<9, 9> = DistanceField::try_build(&room, &mut workspace).unwrap();

    for row in 0..9 {
        for column in 0..9 {
            assert_eq!(field.distance_of(row, column), Some(f64::INFINITY));
        }
    }
}

#[test]
fn distance_at_interpolates_between_cell_centres_f64() {
    let mut room: OccupancyGrid<9, 9, 1> = OccupancyGrid::try_new(1.0, [0.0, 0.0]).unwrap();
    room.set_cell(0, 0, true);
    let mut workspace: DistanceTransformWorkspace<10> = DistanceTransformWorkspace::new();
    let field: DistanceField<9, 9> = DistanceField::try_build(&room, &mut workspace).unwrap();

    // At a cell centre the interpolation is the cell's own value.
    for (row, column) in [(3, 4), (5, 5), (7, 2)] {
        let centre = field.geometry().center_of(row, column).unwrap();
        let interpolated = field.distance_at(centre).unwrap();
        let exact = field.distance_of(row, column).unwrap();
        assert!((interpolated - exact).abs() < 1e-12);
    }

    // Halfway between two neighbours it is their mean.
    let left = field.geometry().center_of(4, 4).unwrap();
    let right = field.geometry().center_of(4, 5).unwrap();
    let midpoint = [(left[0] + right[0]) / 2.0, left[1]];
    let expected = (field.distance_of(4, 4).unwrap() + field.distance_of(4, 5).unwrap()) / 2.0;
    assert!((field.distance_at(midpoint).unwrap() - expected).abs() < 1e-12);
}

#[test]
fn distance_at_returns_none_outside_f64() {
    let mut room: OccupancyGrid<9, 9, 1> = OccupancyGrid::try_new(1.0, [0.0, 0.0]).unwrap();
    room.set_cell(0, 0, true);
    let mut workspace: DistanceTransformWorkspace<10> = DistanceTransformWorkspace::new();
    let field: DistanceField<9, 9> = DistanceField::try_build(&room, &mut workspace).unwrap();

    // Below the first cell centre, and past the last: the outer half-cell rim has no four cells
    // to blend.
    assert_eq!(field.distance_at([-1.0, 4.5]), None);
    assert_eq!(field.distance_at([0.4, 4.5]), None);
    assert_eq!(field.distance_at([8.6, 4.5]), None);
    assert_eq!(field.distance_at([4.5, 100.0]), None);
    assert_eq!(field.distance_at([f64::NAN, 4.5]), None);
}

#[test]
fn gradient_points_away_from_a_wall_f64() {
    // A wall along the whole of row 2, so the field varies only up the columns.
    let mut room: OccupancyGrid<20, 20, 1> = OccupancyGrid::try_new(0.1, [0.0, 0.0]).unwrap();
    for column in 0..20 {
        room.set_cell(2, column, true);
    }
    let mut workspace: DistanceTransformWorkspace<21> = DistanceTransformWorkspace::new();
    let field: DistanceField<20, 20> = DistanceField::try_build(&room, &mut workspace).unwrap();

    // Well above the wall the gradient is the wall normal: straight up, unit length.
    for row in 8..14 {
        let point = field.geometry().center_of(row, 10).unwrap();
        let gradient = field.gradient_at(point).unwrap();
        assert!(gradient[0].abs() < 1e-6, "row {row}: {gradient:?}");
        assert!((gradient[1] - 1.0).abs() < 1e-6, "row {row}: {gradient:?}");
    }
}

#[test]
fn unknown_cells_do_not_seed_obstacles_f64() {
    // A belief grid observed only in a band: the rest is unknown, which must not read as blocked.
    let mut belief: LogOddsGrid<20, 20> = LogOddsGrid::try_new(0.1, [0.0, 0.0]).unwrap();
    let scan: ScanGeometry<3> = ScanGeometry::try_new(0.02, 4.0).unwrap();
    let pose = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([0.05, 1.05]));
    for _ in 0..6 {
        belief.integrate_scan(pose, &scan, &[1.0; 3]);
    }

    // There is unmapped space, and a real obstacle where the beams stopped.
    assert_eq!(belief.cell_state(18, 18), CellState::Unknown);
    assert_eq!(belief.cell_state(10, 10), CellState::Occupied);

    let mut workspace: DistanceTransformWorkspace<21> = DistanceTransformWorkspace::new();
    let field: DistanceField<20, 20> = DistanceField::try_build(&belief, &mut workspace).unwrap();

    // Distances across the unmapped band are finite: they measure to the nearest *known*
    // obstacle rather than treating the unknown as a wall.
    for row in 15..20 {
        for column in 15..20 {
            let distance = field.distance_of(row, column).unwrap();
            assert!(distance.is_finite() && distance > 0.0, "({row}, {column})");
        }
    }
}
