//! The beam traversal on its own: which cells it yields, in what order, and where it stops.

use multicalc::mapping::GridGeometry;
use multicalc::scalar::{Numeric, Primal};

/// A 5 by 5 grid of unit cells with its lowest corner at the world origin.
fn unit_grid<T: Numeric + Primal>() -> GridGeometry<T> {
    GridGeometry::try_new(5, 5, T::ONE, [T::ZERO, T::ZERO]).unwrap()
}

fn assert_walk_visits_cells_in_order<T: Numeric + Primal>() {
    let geometry = unit_grid::<T>();
    let middle_of_the_first_cell = [T::HALF, T::from_f64(2.5)];
    let along_the_row = T::ZERO;

    let steps: Vec<_> = geometry
        .walk(middle_of_the_first_cell, along_the_row, T::from_f64(10.0))
        .collect();

    assert_eq!(steps.len(), 5);
    for (column, step) in steps.iter().enumerate() {
        assert_eq!((step.row, step.column), (2, column));
    }
}

fn assert_entry_distance_is_monotone<T: Numeric + Primal>() {
    let geometry = unit_grid::<T>();
    let diagonally = T::PI / T::from_f64(5.0);

    let mut previous = T::NEG_INFINITY;
    for step in geometry.walk([T::HALF, T::HALF], diagonally, T::from_f64(20.0)) {
        assert!(step.entry_distance >= previous);
        previous = step.entry_distance;
    }
    assert!(previous > T::ZERO);
}

fn assert_walk_from_outside_enters_the_grid<T: Numeric + Primal>() {
    let geometry = unit_grid::<T>();
    let two_cells_short_of_the_edge = [T::from_f64(-2.0), T::from_f64(2.5)];
    let along_the_row = T::ZERO;

    let mut walk = geometry.walk(
        two_cells_short_of_the_edge,
        along_the_row,
        T::from_f64(20.0),
    );
    let first = walk.next().unwrap();

    assert_eq!((first.row, first.column), (2, 0));
    // It enters where the grid starts, two metres along.
    assert!((first.entry_distance - T::from_f64(2.0)).abs() < T::from_f64(1e-6));
}

fn assert_walk_that_never_enters_yields_nothing<T: Numeric + Primal>() {
    let geometry = unit_grid::<T>();
    let well_below_the_grid = [T::from_f64(-2.0), T::from_f64(-3.0)];
    let along_the_row = T::ZERO;

    assert_eq!(
        geometry
            .walk(well_below_the_grid, along_the_row, T::from_f64(20.0))
            .count(),
        0
    );

    // Aimed away from a grid it starts beside.
    let facing_away = T::PI;
    assert_eq!(
        geometry
            .walk([T::from_f64(-0.5), T::HALF], facing_away, T::from_f64(20.0))
            .count(),
        0
    );
}

fn assert_walk_respects_maximum_range<T: Numeric + Primal>() {
    let geometry = unit_grid::<T>();
    let start = [T::HALF, T::from_f64(2.5)];
    let along_the_row = T::ZERO;

    // Entry distances are 0, 0.5, 1.5, 2.5, 3.5, and a cell entered at exactly the range still
    // counts, so two and a half metres of reach covers four cells.
    let reach = T::from_f64(2.5);
    assert_eq!(geometry.walk(start, along_the_row, reach).count(), 4);

    let just_short = T::from_f64(2.4);
    assert_eq!(geometry.walk(start, along_the_row, just_short).count(), 3);

    // A beam whose range ends before the grid does reaches no cell at all.
    let short_of_the_grid = T::from_f64(0.5);
    assert_eq!(
        geometry
            .walk(
                [T::from_f64(-2.0), T::HALF],
                along_the_row,
                short_of_the_grid
            )
            .count(),
        0
    );
}

#[test]
fn walk_visits_cells_in_order_f64() {
    assert_walk_visits_cells_in_order::<f64>();
}

#[test]
fn walk_visits_cells_in_order_f32() {
    assert_walk_visits_cells_in_order::<f32>();
}

#[test]
fn entry_distance_is_monotone_f64() {
    assert_entry_distance_is_monotone::<f64>();
}

#[test]
fn walk_from_outside_enters_the_grid_f64() {
    assert_walk_from_outside_enters_the_grid::<f64>();
}

#[test]
fn walk_that_never_enters_yields_nothing_f64() {
    assert_walk_that_never_enters_yields_nothing::<f64>();
}

#[test]
fn walk_respects_maximum_range_f64() {
    assert_walk_respects_maximum_range::<f64>();
}
