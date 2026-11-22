//! The heap-backed grid: how it is built, and how it stores and reads its cells.

use multicalc::error::MappingError;
use multicalc::mapping::{DynamicOccupancyGrid, MutableOccupancyMap, OccupancyMap};

#[test]
fn a_grid_with_no_cells_is_rejected() {
    let cell_size = 0.5_f64;
    let corner = [0.0, 0.0];
    assert_eq!(
        DynamicOccupancyGrid::try_new(0, 4, cell_size, corner).unwrap_err(),
        MappingError::EmptyGrid
    );
    assert_eq!(
        DynamicOccupancyGrid::try_new(4, 0, cell_size, corner).unwrap_err(),
        MappingError::EmptyGrid
    );
}

/// Columns times rows must be countable: the check has to happen before the multiplication, not
/// after it.
#[test]
fn a_grid_too_large_to_count_is_rejected() {
    assert_eq!(
        DynamicOccupancyGrid::try_new(usize::MAX, 4, 0.5_f64, [0.0, 0.0]).unwrap_err(),
        MappingError::GridTooLarge
    );
}

#[test]
fn an_unusable_cell_size_or_corner_is_rejected() {
    let corner = [0.0, 0.0];
    assert_eq!(
        DynamicOccupancyGrid::try_new(4, 4, 0.0_f64, corner).unwrap_err(),
        MappingError::NonPositiveResolution
    );
    assert_eq!(
        DynamicOccupancyGrid::try_new(4, 4, -0.5_f64, corner).unwrap_err(),
        MappingError::NonPositiveResolution
    );
    assert_eq!(
        DynamicOccupancyGrid::try_new(4, 4, f64::NAN, corner).unwrap_err(),
        MappingError::NonFinite
    );
    assert_eq!(
        DynamicOccupancyGrid::try_new(4, 4, 0.5_f64, [0.0, f64::INFINITY]).unwrap_err(),
        MappingError::NonFinite
    );
}

#[test]
fn a_new_grid_is_empty_and_keeps_the_shape_it_was_asked_for() {
    let grid = DynamicOccupancyGrid::try_new(4, 3, 0.5_f64, [-1.0, -1.0]).unwrap();
    assert_eq!(grid.columns(), 4);
    assert_eq!(grid.rows(), 3);
    assert_eq!(grid.resolution(), 0.5);
    assert_eq!(grid.origin(), [-1.0, -1.0]);
    for row in 0..3 {
        for column in 0..4 {
            assert!(!grid.is_occupied(row, column), "({row}, {column})");
        }
    }
}

#[test]
fn cells_are_named_row_first() {
    let mut grid = DynamicOccupancyGrid::try_new(4, 3, 0.5_f64, [0.0, 0.0]).unwrap();
    grid.set_cell(1, 0, true);
    assert!(grid.is_occupied(1, 0));
    assert!(
        !grid.is_occupied(0, 1),
        "the pair is (row, column), not (column, row)"
    );
    // Cell (1, 0) spans x in [0, 0.5] and y in [0.5, 1.0].
    assert_eq!(grid.cell_of([0.25, 0.75]), Some((1, 0)));
    assert_eq!(grid.cell_of([0.75, 0.25]), Some((0, 1)));
}

/// An index past the end must read free and write nothing — including one large enough that working
/// out where it would sit could overflow.
#[test]
fn indices_off_the_map_do_nothing() {
    let mut grid = DynamicOccupancyGrid::try_new(4, 3, 0.5_f64, [0.0, 0.0]).unwrap();
    grid.set_cell(3, 0, true);
    grid.set_cell(0, 4, true);
    grid.set_cell(usize::MAX, usize::MAX, true);
    assert!(!grid.is_occupied(3, 0));
    assert!(!grid.is_occupied(0, 4));
    assert!(!grid.is_occupied(usize::MAX, usize::MAX));
    for row in 0..3 {
        for column in 0..4 {
            assert!(!grid.is_occupied(row, column), "({row}, {column})");
        }
    }
}

#[test]
fn clearing_frees_every_cell_and_leaves_the_shape_alone() {
    let mut grid = DynamicOccupancyGrid::try_new(4, 3, 0.5_f64, [-1.0, -1.0]).unwrap();
    grid.occupy_polyline(&[[-1.0, -1.0], [1.0, 0.5]], false);
    assert!((0..3).any(|row| (0..4).any(|column| grid.is_occupied(row, column))));
    grid.clear();
    assert!((0..3).all(|row| (0..4).all(|column| !grid.is_occupied(row, column))));
    assert_eq!(grid.columns(), 4);
    assert_eq!(grid.rows(), 3);
}

#[test]
fn a_grid_away_from_the_world_origin_places_its_cells_correctly() {
    let mut grid = DynamicOccupancyGrid::try_new(4, 4, 0.5_f64, [-1.0, -1.0]).unwrap();
    grid.occupy_point([-0.9, -0.9]);
    assert!(grid.is_occupied(0, 0));
    // That cell spans x in [-1.0, -0.5], so its near face is 1.4 m from a beam fired back at it.
    let distance = grid.cast_ray([0.9, -0.9], core::f64::consts::PI, 4.0);
    assert!(
        distance.is_some_and(|met| (met - 1.4).abs() < 1e-12),
        "{distance:?}"
    );
}

#[test]
fn a_one_cell_grid_works() {
    let mut grid = DynamicOccupancyGrid::try_new(1, 1, 1.0_f64, [0.0, 0.0]).unwrap();
    assert_eq!(grid.cast_ray([0.5, 0.5], 0.0, 4.0), None);
    grid.set_cell(0, 0, true);
    assert_eq!(
        grid.cast_ray([0.5, 0.5], 0.0, 4.0),
        Some(0.0),
        "a beam starting on a blocked cell reads zero"
    );
}

#[test]
fn a_wall_reads_the_same_distance_at_f32() {
    let mut grid = DynamicOccupancyGrid::try_new(16, 16, 0.25_f32, [0.0, 0.0]).unwrap();
    let blocked_column = 5;
    for row in 0..16 {
        grid.set_cell(row, blocked_column, true);
    }
    let start = [0.5_f32, 0.5];
    let expected = blocked_column as f32 * 0.25 - start[0];
    let distance = grid.cast_ray(start, 0.0, 4.0);
    assert!(
        distance.is_some_and(|met| (met - expected).abs() < 1e-4),
        "{distance:?}, expected {expected}"
    );
}
