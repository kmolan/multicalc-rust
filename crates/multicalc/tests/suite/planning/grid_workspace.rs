//! The grid search workspace: what a fresh one holds, and that a reset costs the map rather than
//! the capacity.

use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid};
use multicalc::planning::{GridPlanner, GridSearchWorkspace, UniformCost};

#[test]
fn capacity_reports_the_const_parameter_f64() {
    let workspace: GridSearchWorkspace<256> = GridSearchWorkspace::new();
    assert_eq!(workspace.capacity(), 256);

    let narrow: GridSearchWorkspace<16, f32> = GridSearchWorkspace::default();
    assert_eq!(narrow.capacity(), 16);
}

#[test]
fn a_fresh_workspace_and_a_default_one_plan_alike_f64() {
    let mut room: OccupancyGrid<8, 8, 1> = OccupancyGrid::try_new(1.0, [0.0, 0.0]).unwrap();
    room.set_cell(4, 4, true);
    let cost = UniformCost::new(&room);
    let planner = GridPlanner::<f64>::new();

    let mut built: GridSearchWorkspace<64> = GridSearchWorkspace::new();
    let mut defaulted: GridSearchWorkspace<64> = GridSearchWorkspace::default();

    let first = planner
        .try_plan::<64, 64, _, _>(&room, &cost, [0.5, 0.5], [7.5, 7.5], &mut built)
        .unwrap();
    let second = planner
        .try_plan::<64, 64, _, _>(&room, &cost, [0.5, 0.5], [7.5, 7.5], &mut defaulted)
        .unwrap();
    assert_eq!(first, second);
}

#[test]
fn a_workspace_larger_than_the_map_plans_the_same_f64() {
    // `reset` clears only the used prefix, so an oversized workspace must not carry stale state
    // into the next plan.
    let mut room: OccupancyGrid<8, 8, 1> = OccupancyGrid::try_new(1.0, [0.0, 0.0]).unwrap();
    room.set_cell(4, 4, true);
    let cost = UniformCost::new(&room);
    let planner = GridPlanner::<f64>::new();

    let mut exact: GridSearchWorkspace<64> = GridSearchWorkspace::new();
    let mut roomy: GridSearchWorkspace<1024> = GridSearchWorkspace::new();

    let tight = planner
        .try_plan::<64, 64, _, _>(&room, &cost, [0.5, 0.5], [7.5, 7.5], &mut exact)
        .unwrap();
    let loose = planner
        .try_plan::<1024, 64, _, _>(&room, &cost, [0.5, 0.5], [7.5, 7.5], &mut roomy)
        .unwrap();

    assert_eq!(tight.cost(), loose.cost());
    assert_eq!(tight.path().waypoints(), loose.path().waypoints());
}
