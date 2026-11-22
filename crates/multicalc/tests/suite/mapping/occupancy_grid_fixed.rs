//! The bit-packed grid: that its packing is honest across word boundaries, and that it answers the
//! map trait exactly as the heap grid does.

use multicalc::error::MappingError;
use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid, OccupancyMap};
use multicalc::scalar::{Numeric, Primal};

#[test]
fn try_new_rejects_words_per_row_below_the_formula() {
    // 65 columns need three words; two is one short.
    assert_eq!(
        OccupancyGrid::<33, 65, 2, f64>::try_new(0.1, [0.0, 0.0]).err(),
        Some(MappingError::WordsPerRowTooSmall)
    );
    assert!(OccupancyGrid::<33, 65, 3, f64>::try_new(0.1, [0.0, 0.0]).is_ok());
    assert_eq!(OccupancyGrid::<33, 65, 3, f64>::WORDS_NEEDED, 3);
}

#[test]
fn try_new_rejects_a_bad_placement() {
    assert_eq!(
        OccupancyGrid::<4, 4, 1, f64>::try_new(0.0, [0.0, 0.0]).err(),
        Some(MappingError::NonPositiveResolution)
    );
    assert_eq!(
        OccupancyGrid::<4, 4, 1, f64>::try_new(0.1, [f64::NAN, 0.0]).err(),
        Some(MappingError::NonFinite)
    );
}

/// A 33 by 65 grid is deliberately astride word boundaries on both axes.
fn assert_set_and_read_round_trip_every_cell<T: Numeric + Primal>() {
    let mut grid: OccupancyGrid<33, 65, 3, T> =
        OccupancyGrid::try_new(T::from_f64(0.1), [T::ZERO, T::ZERO]).unwrap();

    for row in 0..33 {
        for column in 0..65 {
            assert!(!grid.is_occupied(row, column));
            grid.set_cell(row, column, true);
            assert!(grid.is_occupied(row, column));

            // Setting one cell left every other cell alone.
            let occupied_count: usize = (0..33)
                .flat_map(|other_row| (0..65).map(move |other_column| (other_row, other_column)))
                .filter(|&(other_row, other_column)| grid.is_occupied(other_row, other_column))
                .count();
            assert_eq!(occupied_count, 1);

            grid.set_cell(row, column, false);
            assert!(!grid.is_occupied(row, column));
        }
    }
}

#[test]
fn set_and_read_round_trip_every_cell_f64() {
    assert_set_and_read_round_trip_every_cell::<f64>();
}

#[test]
fn bits_do_not_leak_across_word_boundaries_f64() {
    let mut grid: OccupancyGrid<2, 65, 3, f64> = OccupancyGrid::try_new(0.1, [0.0, 0.0]).unwrap();

    grid.set_cell(0, 31, true);
    assert!(grid.is_occupied(0, 31));
    assert!(!grid.is_occupied(0, 30));
    assert!(!grid.is_occupied(0, 32));
    assert!(!grid.is_occupied(1, 31));

    grid.clear();
    grid.set_cell(0, 32, true);
    assert!(grid.is_occupied(0, 32));
    assert!(!grid.is_occupied(0, 31));
    assert!(!grid.is_occupied(0, 33));

    // The last column lives in the third word, whose upper bits are unused.
    grid.clear();
    grid.set_cell(1, 64, true);
    assert!(grid.is_occupied(1, 64));
    assert!(!grid.is_occupied(1, 63));
    assert!(!grid.is_occupied(0, 64));
}

#[test]
fn out_of_range_reads_free_and_writes_are_ignored_f64() {
    let mut grid: OccupancyGrid<4, 4, 1, f64> = OccupancyGrid::try_new(0.5, [0.0, 0.0]).unwrap();

    // A column past the extent must not reach the padding bits of the row's word.
    grid.set_cell(0, 4, true);
    grid.set_cell(0, 31, true);
    grid.set_cell(4, 0, true);
    grid.set_cell(0, usize::MAX, true);

    assert!(!grid.is_occupied(0, 4));
    assert!(!grid.is_occupied(0, 31));
    assert!(!grid.is_occupied(4, 0));
    assert!(!grid.is_occupied(0, usize::MAX));
    for row in 0..4 {
        for column in 0..4 {
            assert!(!grid.is_occupied(row, column));
        }
    }
}

#[cfg(feature = "alloc")]
mod against_the_heap_grid {
    use multicalc::mapping::{
        DynamicOccupancyGrid, MutableOccupancyMap, OccupancyGrid, OccupancyMap,
    };
    use multicalc::scalar::{Numeric, Primal};

    /// The same walled room drawn into both grids.
    fn walled_room<T: Numeric + Primal>() -> (OccupancyGrid<20, 20, 1, T>, DynamicOccupancyGrid<T>)
    {
        let resolution = T::from_f64(0.25);
        let origin = [T::ZERO, T::ZERO];
        let mut fixed: OccupancyGrid<20, 20, 1, T> =
            OccupancyGrid::try_new(resolution, origin).unwrap();
        let mut dynamic = DynamicOccupancyGrid::try_new(20, 20, resolution, origin).unwrap();

        let walls = [
            [T::from_f64(0.5), T::from_f64(0.5)],
            [T::from_f64(4.5), T::from_f64(0.5)],
            [T::from_f64(4.5), T::from_f64(4.5)],
            [T::from_f64(0.5), T::from_f64(4.5)],
        ];
        let joined_up = true;
        fixed.occupy_polyline(&walls, joined_up);
        dynamic.occupy_polyline(&walls, joined_up);
        (fixed, dynamic)
    }

    #[test]
    fn cast_ray_matches_dynamic_grid_f64() {
        let (fixed, dynamic) = walled_room::<f64>();
        let standing_at = [2.5, 2.5];
        let maximum_range = 6.0;

        for step in 0..64 {
            let bearing = core::f64::consts::TAU * step as f64 / 64.0;
            assert_eq!(
                fixed.cast_ray(standing_at, bearing, maximum_range),
                dynamic.cast_ray(standing_at, bearing, maximum_range),
            );
        }
    }

    #[test]
    fn occupy_circle_matches_dynamic_grid_f32() {
        let resolution = 0.25_f32;
        let origin = [0.0_f32, 0.0];
        let mut fixed: OccupancyGrid<20, 20, 1, f32> =
            OccupancyGrid::try_new(resolution, origin).unwrap();
        let mut dynamic = DynamicOccupancyGrid::try_new(20, 20, resolution, origin).unwrap();

        let centre = [2.5_f32, 2.5];
        let radius = 1.5_f32;
        fixed.occupy_circle(centre, radius);
        dynamic.occupy_circle(centre, radius);

        for row in 0..20 {
            for column in 0..20 {
                assert_eq!(
                    fixed.is_occupied(row, column),
                    dynamic.is_occupied(row, column),
                    "cell ({row}, {column})"
                );
            }
        }
    }
}
