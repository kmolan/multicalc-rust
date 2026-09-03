//! The inflation costmap: its guards, and that its decay matches the closed form it is defined by.

use multicalc::error::MappingError;
use multicalc::mapping::{
    CostGrid, DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid,
};
use multicalc::scalar::{Numeric, Primal};

/// A 2 m square at 10 cm cells with a wall along row 2, and its distance field.
fn walled_field<T: Numeric + Primal>() -> DistanceField<20, 20, T> {
    let mut room: OccupancyGrid<20, 20, 1, T> =
        OccupancyGrid::try_new(T::from_f64(0.1), [T::ZERO, T::ZERO]).unwrap();
    for column in 0..20 {
        room.set_cell(2, column, true);
    }
    let mut workspace: DistanceTransformWorkspace<21, T> = DistanceTransformWorkspace::new();
    DistanceField::try_build(&room, &mut workspace).unwrap()
}

#[test]
fn try_build_rejects_reversed_radii() {
    let field = walled_field::<f64>();
    assert_eq!(
        CostGrid::try_build(&field, 1.0, 0.5, 3.0).err(),
        Some(MappingError::RadiiNotOrdered)
    );
}

#[test]
fn try_build_rejects_non_positive_scaling() {
    let field = walled_field::<f64>();
    assert_eq!(
        CostGrid::try_build(&field, 0.2, 1.0, 0.0).err(),
        Some(MappingError::NonPositiveScaling)
    );
    assert_eq!(
        CostGrid::try_build(&field, 0.2, 1.0, -1.0).err(),
        Some(MappingError::NonPositiveScaling)
    );
}

#[test]
fn try_build_rejects_a_bad_range_or_a_non_finite_value() {
    let field = walled_field::<f64>();
    assert_eq!(
        CostGrid::try_build(&field, -0.1, 1.0, 3.0).err(),
        Some(MappingError::NonPositiveRange)
    );
    assert_eq!(
        CostGrid::try_build(&field, 0.0, 0.0, 3.0).err(),
        Some(MappingError::NonPositiveRange)
    );
    assert_eq!(
        CostGrid::try_build(&field, 0.2, f64::NAN, 3.0).err(),
        Some(MappingError::NonFinite)
    );
}

#[test]
fn cost_is_lethal_inside_the_inscribed_radius_f64() {
    let field = walled_field::<f64>();
    let inscribed_radius = 0.25;
    let costmap: CostGrid<20, 20> =
        CostGrid::try_build(&field, inscribed_radius, 1.0, 3.0).unwrap();

    for row in 0..20 {
        for column in 0..20 {
            let distance = field.distance_of(row, column).unwrap();
            if distance <= inscribed_radius {
                assert_eq!(
                    costmap.cost_of(row, column),
                    Some(CostGrid::<20, 20>::LETHAL),
                    "({row}, {column}) at {distance}"
                );
            }
        }
    }
}

#[test]
fn cost_is_zero_beyond_the_inflation_radius_f64() {
    let field = walled_field::<f64>();
    let inflation_radius = 0.5;
    let costmap: CostGrid<20, 20> =
        CostGrid::try_build(&field, 0.1, inflation_radius, 3.0).unwrap();

    for row in 0..20 {
        for column in 0..20 {
            let distance = field.distance_of(row, column).unwrap();
            if distance > inflation_radius {
                assert_eq!(costmap.cost_of(row, column), Some(0), "({row}, {column})");
            }
        }
    }
}

fn assert_cost_decays_monotonically<T: Numeric + Primal>() {
    let field = walled_field::<T>();
    let costmap: CostGrid<20, 20, T> =
        CostGrid::try_build(&field, T::from_f64(0.1), T::from_f64(1.2), T::from_f64(3.0)).unwrap();

    // Walking straight up away from the wall, cost never rises.
    let mut previous = 255u8;
    for row in 2..20 {
        let cost = costmap.cost_of(row, 10).unwrap();
        assert!(cost <= previous, "row {row}: {cost} after {previous}");
        previous = cost;
    }
    assert_eq!(previous, 0);
}

#[test]
fn cost_decays_monotonically_with_distance_f64() {
    assert_cost_decays_monotonically::<f64>();
}

#[test]
fn cost_decays_monotonically_with_distance_f32() {
    assert_cost_decays_monotonically::<f32>();
}

#[test]
fn cost_matches_the_closed_form_f64() {
    let field = walled_field::<f64>();
    let inscribed_radius = 0.15;
    let inflation_radius = 1.0;
    let cost_scaling_factor = 2.5;
    let costmap: CostGrid<20, 20> = CostGrid::try_build(
        &field,
        inscribed_radius,
        inflation_radius,
        cost_scaling_factor,
    )
    .unwrap();

    for row in 0..20 {
        for column in 0..20 {
            let distance = field.distance_of(row, column).unwrap();
            let expected = if distance <= inscribed_radius {
                255
            } else if distance <= inflation_radius {
                (254.0 * (-cost_scaling_factor * (distance - inscribed_radius)).exp()) as u8
            } else {
                0
            };
            assert_eq!(
                costmap.cost_of(row, column),
                Some(expected),
                "({row}, {column}) at distance {distance}"
            );
        }
    }
}

#[test]
fn cost_at_reads_through_the_geometry_f64() {
    let field = walled_field::<f64>();
    let costmap: CostGrid<20, 20> = CostGrid::try_build(&field, 0.15, 1.0, 2.5).unwrap();

    for (row, column) in [(0, 0), (5, 7), (19, 19)] {
        let centre = costmap.geometry().center_of(row, column).unwrap();
        assert_eq!(costmap.cost_at(centre), costmap.cost_of(row, column));
    }
    assert_eq!(costmap.cost_at([-1.0, 0.5]), None);
    assert_eq!(costmap.cost_at([100.0, 0.5]), None);
}
