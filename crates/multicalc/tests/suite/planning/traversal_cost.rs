//! What entering a cell costs, over a plain map and over an inflation costmap.

use multicalc::mapping::{
    CostGrid, DistanceField, DistanceTransformWorkspace, LogOddsGrid, MutableOccupancyMap,
    OccupancyGrid, ScanGeometry,
};
use multicalc::planning::{CostmapCost, TraversalCost, UniformCost};
use multicalc::{SE2, SO2, Vector2D};

#[test]
fn uniform_cost_blocks_occupied_cells_f64() {
    let mut room: OccupancyGrid<8, 8, 1> = OccupancyGrid::try_new(0.5, [0.0, 0.0]).unwrap();
    room.set_cell(3, 3, true);
    let cost = UniformCost::new(&room);

    assert_eq!(cost.cost_of(3, 3), None);
    assert_eq!(cost.cost_of(0, 0), Some(1.0));
    assert_eq!(cost.cost_of(7, 7), Some(1.0));
}

/// A belief grid with one observed band, so most of it is genuinely unknown.
fn partly_observed() -> LogOddsGrid<10, 10> {
    let mut belief: LogOddsGrid<10, 10> = LogOddsGrid::try_new(1.0, [0.0, 0.0]).unwrap();
    let scan: ScanGeometry<3> = ScanGeometry::try_new(0.02, 20.0).unwrap();
    let pose = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([0.5, 0.5]));
    for _ in 0..6 {
        belief.integrate_scan(pose, &scan, &[20.0; 3]);
    }
    belief
}

#[test]
fn uniform_cost_blocks_unknown_by_default_f64() {
    let belief = partly_observed();
    let cost = UniformCost::new(&belief);

    // The observed row is passable; the unobserved rest is not.
    assert_eq!(cost.cost_of(0, 5), Some(1.0));
    assert_eq!(cost.cost_of(5, 5), None);
    // Off the grid a belief map reads unknown, so it is impassable too.
    assert_eq!(cost.cost_of(10, 0), None);
}

#[test]
fn uniform_cost_admits_unknown_when_asked_f64() {
    let belief = partly_observed();
    let cost = UniformCost::new(&belief).with_unknown_passable(true);

    assert_eq!(cost.cost_of(0, 5), Some(1.0));
    assert_eq!(cost.cost_of(5, 5), Some(1.0));
}

/// A costmap over a 2 m square with a wall along row 2.
fn walled_costmap() -> (DistanceField<20, 20>, CostGrid<20, 20>) {
    let mut room: OccupancyGrid<20, 20, 1> = OccupancyGrid::try_new(0.1, [0.0, 0.0]).unwrap();
    for column in 0..20 {
        room.set_cell(2, column, true);
    }
    let mut workspace: DistanceTransformWorkspace<21> = DistanceTransformWorkspace::new();
    let field = DistanceField::try_build(&room, &mut workspace).unwrap();
    let costmap = CostGrid::try_build(&field, 0.15, 1.0, 2.0).unwrap();
    (field, costmap)
}

#[test]
fn costmap_cost_blocks_lethal_cells_f64() {
    let (_, costmap) = walled_costmap();
    let cost = CostmapCost::new(&costmap);

    for column in 0..20 {
        assert_eq!(costmap.cost_of(2, column), Some(CostGrid::<20, 20>::LETHAL));
        assert_eq!(cost.cost_of(2, column), None);
    }
}

#[test]
fn costmap_cost_rises_with_cost_f64() {
    let (_, costmap) = walled_costmap();
    let cost = CostmapCost::new(&costmap);

    // Row 3 sits inside the inscribed radius, so it is lethal and has no multiplier at all.
    assert_eq!(cost.cost_of(3, 10), None);

    // From there out, the multiplier falls toward one and never rises.
    let mut previous = f64::INFINITY;
    for row in 4..20 {
        let multiplier = cost.cost_of(row, 10).unwrap();
        assert!(multiplier >= 1.0);
        assert!(multiplier <= previous, "row {row}");
        previous = multiplier;
    }
    assert!((previous - 1.0).abs() < 1e-12);

    // The weight scales how far above one the multiplier sits.
    let heavier = CostmapCost::new(&costmap).with_weight(4.0);
    let near_the_wall = 4;
    let plain = cost.cost_of(near_the_wall, 10).unwrap();
    let scaled = heavier.cost_of(near_the_wall, 10).unwrap();
    assert!((scaled - 1.0 - 4.0 * (plain - 1.0)).abs() < 1e-12);
}

#[test]
fn costmap_cost_blocks_outside_the_grid_f64() {
    let (_, costmap) = walled_costmap();
    let cost = CostmapCost::new(&costmap);

    assert_eq!(cost.cost_of(20, 0), None);
    assert_eq!(cost.cost_of(0, 20), None);
    assert_eq!(cost.cost_of(usize::MAX, usize::MAX), None);
}
