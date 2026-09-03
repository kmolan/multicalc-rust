//! The shared grid placement: what it rejects, and that its cell arithmetic round-trips and agrees
//! with the map trait it was extracted from.

use multicalc::error::MappingError;
use multicalc::mapping::{GridGeometry, OccupancyMap};
use multicalc::scalar::{Numeric, Primal};

/// A map of `COLUMNS` by `ROWS` free cells, for comparing the geometry against the trait.
struct FreeMap<const COLUMNS: usize, const ROWS: usize, T: Numeric + Primal> {
    resolution: T,
    origin: [T; 2],
}

impl<const COLUMNS: usize, const ROWS: usize, T: Numeric + Primal> OccupancyMap<T>
    for FreeMap<COLUMNS, ROWS, T>
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
        self.origin
    }
    fn is_occupied(&self, _row: usize, _column: usize) -> bool {
        false
    }
}

#[test]
fn try_new_rejects_zero_rows() {
    assert_eq!(
        GridGeometry::<f64>::try_new(0, 4, 0.5, [0.0, 0.0]),
        Err(MappingError::EmptyGrid)
    );
    assert_eq!(
        GridGeometry::<f64>::try_new(4, 0, 0.5, [0.0, 0.0]),
        Err(MappingError::EmptyGrid)
    );
}

#[test]
fn try_new_rejects_non_positive_resolution() {
    assert_eq!(
        GridGeometry::<f64>::try_new(4, 4, 0.0, [0.0, 0.0]),
        Err(MappingError::NonPositiveResolution)
    );
    assert_eq!(
        GridGeometry::<f64>::try_new(4, 4, -0.5, [0.0, 0.0]),
        Err(MappingError::NonPositiveResolution)
    );
}

#[test]
fn try_new_rejects_non_finite_origin() {
    assert_eq!(
        GridGeometry::<f64>::try_new(4, 4, 0.5, [f64::NAN, 0.0]),
        Err(MappingError::NonFinite)
    );
    assert_eq!(
        GridGeometry::<f64>::try_new(4, 4, 0.5, [0.0, f64::INFINITY]),
        Err(MappingError::NonFinite)
    );
    assert_eq!(
        GridGeometry::<f64>::try_new(4, 4, f64::NAN, [0.0, 0.0]),
        Err(MappingError::NonFinite)
    );
}

#[test]
fn try_new_rejects_a_grid_too_large_to_index() {
    assert_eq!(
        GridGeometry::<f64>::try_new(usize::MAX, 2, 0.5, [0.0, 0.0]),
        Err(MappingError::GridTooLarge)
    );
    assert_eq!(
        GridGeometry::<f64>::try_new(u32::MAX as usize, 2, 0.5, [0.0, 0.0]),
        Err(MappingError::GridTooLarge)
    );
}

/// Twenty points across and beyond a map, each landing in the same cell either way.
fn assert_cell_of_matches_the_map<T: Numeric + Primal>() {
    let resolution = T::from_f64(0.5);
    let origin = [T::from_f64(-1.0), T::from_f64(-2.0)];
    let map: FreeMap<6, 4, T> = FreeMap { resolution, origin };
    let geometry = map.geometry();

    for index in 0..20 {
        let along = T::from_f64(-2.0) + T::from_usize(index) * T::from_f64(0.35);
        let point = [origin[0] + along, origin[1] + along];
        assert_eq!(geometry.cell_of(point), map.cell_of(point));
    }
}

#[test]
fn cell_of_matches_occupancy_map_f64() {
    assert_cell_of_matches_the_map::<f64>();
}

#[test]
fn cell_of_matches_occupancy_map_f32() {
    assert_cell_of_matches_the_map::<f32>();
}

/// Every cell's middle is inside that cell — a discrete equality, so no tolerance.
fn assert_center_of_round_trips<T: Numeric + Primal>() {
    let geometry: GridGeometry<T> = GridGeometry::try_new(
        7,
        5,
        T::from_f64(0.3),
        [T::from_f64(1.5), T::from_f64(-4.0)],
    )
    .unwrap();

    for row in 0..geometry.rows() {
        for column in 0..geometry.columns() {
            let center = geometry.center_of(row, column).unwrap();
            assert_eq!(geometry.cell_of(center), Some((row, column)));
        }
    }
    assert_eq!(geometry.center_of(geometry.rows(), 0), None);
    assert_eq!(geometry.center_of(0, geometry.columns()), None);
}

#[test]
fn center_of_round_trips_through_cell_of_f64() {
    assert_center_of_round_trips::<f64>();
}

#[test]
fn center_of_round_trips_through_cell_of_f32() {
    assert_center_of_round_trips::<f32>();
}

fn assert_index_of_and_cell_at_are_inverse<T: Numeric + Primal>() {
    let geometry: GridGeometry<T> =
        GridGeometry::try_new(7, 5, T::from_f64(0.3), [T::ZERO, T::ZERO]).unwrap();
    assert_eq!(geometry.cell_count(), 35);

    for row in 0..geometry.rows() {
        for column in 0..geometry.columns() {
            let index = geometry.index_of(row, column).unwrap();
            assert_eq!(index, row * geometry.columns() + column);
            assert_eq!(geometry.cell_at(index), Some((row, column)));
        }
    }
    assert_eq!(geometry.index_of(7, 0), None);
    assert_eq!(geometry.index_of(0, 5), None);
    assert_eq!(geometry.cell_at(35), None);
}

#[test]
fn index_of_and_cell_at_are_inverse_f64() {
    assert_index_of_and_cell_at_are_inverse::<f64>();
}

#[test]
fn index_of_and_cell_at_are_inverse_f32() {
    assert_index_of_and_cell_at_are_inverse::<f32>();
}
