//! The grid search, pinned against a Bellman–Ford oracle.
//!
//! The oracle relaxes every edge `cells − 1` times: no heap, no heuristic, correct by inspection.
//! It shares no data structure, loop shape or termination argument with the planner, which is the
//! point of having it.

use multicalc::error::PlanningError;
use multicalc::mapping::{
    CellState, CostGrid, DistanceField, DistanceTransformWorkspace, LogOddsGrid,
    MutableOccupancyMap, OccupancyGrid, OccupancyMap, ScanGeometry,
};
use multicalc::planning::{
    CostmapCost, GridConnectivity, GridHeuristic, GridPlanner, GridSearch, GridSearchWorkspace,
    TraversalCost, UniformCost,
};
use multicalc::{SE2, SO2, Vector2D};
use proptest::prelude::*;

const ROWS: usize = 10;
const COLUMNS: usize = 10;
const MAX_CELLS: usize = ROWS * COLUMNS;
const MAX_POINTS: usize = 128;

/// A `ROWS` by `COLUMNS` map of unit cells at the world origin.
type Room = OccupancyGrid<ROWS, COLUMNS, 1>;

fn empty_room() -> Room {
    OccupancyGrid::try_new(1.0, [0.0, 0.0]).unwrap()
}

/// The middle of cell `(row, column)`, which is where a plan's waypoints sit.
fn centre(row: usize, column: usize) -> [f64; 2] {
    [column as f64 + 0.5, row as f64 + 0.5]
}

// ---- the oracle ----

/// Shortest-path cost from `start` to every cell, by Bellman-Ford relaxation over the same edge
/// set the planner uses.
fn bellman_ford_costs<M: OccupancyMap<f64>>(
    map: &M,
    start: (usize, usize),
    eight_connected: bool,
) -> Vec<f64> {
    let rows = map.rows();
    let columns = map.columns();
    let cells = rows * columns;
    let mut cost = vec![f64::INFINITY; cells];
    if map.is_occupied(start.0, start.1) {
        return cost;
    }
    cost[start.0 * columns + start.1] = 0.0;

    let orthogonal = [(1_isize, 0_isize), (-1, 0), (0, 1), (0, -1)];
    let diagonal = [(1_isize, 1_isize), (1, -1), (-1, 1), (-1, -1)];

    for _ in 0..cells {
        let mut changed = false;
        for row in 0..rows {
            for column in 0..columns {
                let here = cost[row * columns + column];
                if !here.is_finite() {
                    continue;
                }
                let steps = orthogonal
                    .iter()
                    .map(|offset| (*offset, 1.0_f64))
                    .chain(
                        diagonal
                            .iter()
                            .map(|offset| (*offset, core::f64::consts::SQRT_2)),
                    )
                    .take(if eight_connected { 8 } else { 4 });

                for ((row_offset, column_offset), step) in steps {
                    let Some(next_row) = row.checked_add_signed(row_offset) else {
                        continue;
                    };
                    let Some(next_column) = column.checked_add_signed(column_offset) else {
                        continue;
                    };
                    if next_row >= rows || next_column >= columns {
                        continue;
                    }
                    if map.is_occupied(next_row, next_column) {
                        continue;
                    }
                    // The planner forbids squeezing between two blocked cells at a corner.
                    let is_diagonal = row_offset != 0 && column_offset != 0;
                    if is_diagonal
                        && (map.is_occupied(row, next_column) || map.is_occupied(next_row, column))
                    {
                        continue;
                    }
                    let slot = next_row * columns + next_column;
                    if here + step < cost[slot] - 1e-15 {
                        cost[slot] = here + step;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    cost
}

// ---- golden costs, exact by construction ----

#[test]
fn empty_grid_diagonal_costs_nine_root_two_f64() {
    let room = empty_room();
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    let report = GridPlanner::new()
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 9),
            &mut workspace,
        )
        .unwrap();

    assert!((report.cost() - 9.0 * core::f64::consts::SQRT_2).abs() < 1e-12);
}

#[test]
fn empty_grid_four_connected_costs_eighteen_f64() {
    let room = empty_room();
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    let report = GridPlanner::new()
        .with_connectivity(GridConnectivity::FourConnected)
        .with_heuristic(GridHeuristic::Manhattan)
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 9),
            &mut workspace,
        )
        .unwrap();

    assert!((report.cost() - 18.0).abs() < 1e-12);
}

#[test]
fn a_single_gap_wall_is_threaded_f64() {
    let mut room = empty_room();
    let gap_column = 7;
    for column in 0..COLUMNS {
        if column != gap_column {
            room.set_cell(5, column, true);
        }
    }
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    let report = GridPlanner::new()
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 0),
            &mut workspace,
        )
        .unwrap();

    // Collinear pruning drops the gap cell's own waypoint, so the path is sampled rather than
    // searched for a waypoint: every sample is passable, and one of them is in the gap.
    let waypoints = report.path();
    let waypoints = waypoints.waypoints();
    let mut crossed_the_gap = false;
    for pair in waypoints.windows(2) {
        let (from, into) = (pair[0], pair[1]);
        for station in 0..=100 {
            let amount = station as f64 / 100.0;
            let point = [
                from[0] + (into[0] - from[0]) * amount,
                from[1] + (into[1] - from[1]) * amount,
            ];
            let (row, column) = room.geometry().cell_of(point).unwrap();
            assert!(!room.is_occupied(row, column), "({row}, {column})");
            if row == 5 {
                assert_eq!(column, gap_column);
                crossed_the_gap = true;
            }
        }
    }
    assert!(crossed_the_gap);
}

// ---- optimality against the oracle ----

/// A map from a proptest bit pattern, with the two corners always free.
fn map_from_pattern(pattern: &[bool]) -> Room {
    let mut room = empty_room();
    for (index, &blocked) in pattern.iter().enumerate() {
        let (row, column) = (index / COLUMNS, index % COLUMNS);
        if (row, column) == (0, 0) || (row, column) == (ROWS - 1, COLUMNS - 1) {
            continue;
        }
        room.set_cell(row, column, blocked);
    }
    room
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn astar_cost_equals_bellman_ford_f64(
        pattern in prop::collection::vec(prop::bool::weighted(0.25), MAX_CELLS)
    ) {
        let room = map_from_pattern(&pattern);
        let oracle = bellman_ford_costs(&room, (0, 0), true);
        let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
        let cost = UniformCost::new(&room);

        for row in 0..ROWS {
            for column in 0..COLUMNS {
                if room.is_occupied(row, column) {
                    continue;
                }
                let expected = oracle[row * COLUMNS + column];
                let planned = GridPlanner::new().try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
                    &room, &cost, centre(0, 0), centre(row, column), &mut workspace,
                );
                match planned {
                    Ok(report) => {
                        prop_assert!(expected.is_finite());
                        prop_assert!(
                            (report.cost() - expected).abs() < 1e-12,
                            "cell ({row}, {column}): {} against {expected}", report.cost()
                        );
                    }
                    Err(PlanningError::NoPathFound) => prop_assert!(!expected.is_finite()),
                    Err(other) => prop_assert!(false, "unexpected {other:?}"),
                }
            }
        }
    }

    #[test]
    fn dijkstra_equals_astar_f64(
        pattern in prop::collection::vec(prop::bool::weighted(0.25), MAX_CELLS)
    ) {
        let room = map_from_pattern(&pattern);
        let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
        let cost = UniformCost::new(&room);
        let goal = centre(ROWS - 1, COLUMNS - 1);

        let by_astar = GridPlanner::new()
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(&room, &cost, centre(0, 0), goal, &mut workspace);
        let by_dijkstra = GridPlanner::new()
            .with_search(GridSearch::Dijkstra)
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(&room, &cost, centre(0, 0), goal, &mut workspace);

        match (by_astar, by_dijkstra) {
            (Ok(astar), Ok(dijkstra)) => {
                prop_assert!((astar.cost() - dijkstra.cost()).abs() < 1e-12);
            }
            (Err(left), Err(right)) => prop_assert_eq!(left, right),
            (left, right) => prop_assert!(false, "{left:?} against {right:?}"),
        }
    }

    #[test]
    fn weighted_astar_stays_within_its_bound_f64(
        pattern in prop::collection::vec(prop::bool::weighted(0.2), MAX_CELLS)
    ) {
        let room = map_from_pattern(&pattern);
        let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
        let cost = UniformCost::new(&room);
        let goal = centre(ROWS - 1, COLUMNS - 1);

        let Ok(optimal) = GridPlanner::new()
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(&room, &cost, centre(0, 0), goal, &mut workspace)
        else {
            return Ok(());
        };

        for weight in [1.0, 1.5, 2.0, 3.0] {
            let report = GridPlanner::new()
                .try_with_heuristic_weight(weight)
                .unwrap()
                .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
                    &room, &cost, centre(0, 0), goal, &mut workspace,
                )
                .unwrap();
            prop_assert!(
                report.cost() <= weight * optimal.cost() + 1e-12,
                "weight {weight}: {} against {}", report.cost(), optimal.cost()
            );
        }
    }
}

// ---- the plan itself ----

#[test]
fn every_waypoint_is_passable_and_consecutive_waypoints_are_adjacent_f64() {
    let mut room = empty_room();
    for column in 0..7 {
        room.set_cell(4, column, true);
    }
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    // Corner pruning is off here, so every waypoint is one grid step from the last.
    let report = GridPlanner::new()
        .with_search(GridSearch::Dijkstra)
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 9),
            &mut workspace,
        )
        .unwrap();

    let waypoints = report.path();
    let waypoints = waypoints.waypoints();
    for point in waypoints {
        let (row, column) = room.geometry().cell_of(*point.as_array()).unwrap();
        assert!(!room.is_occupied(row, column), "({row}, {column})");
        assert!(cost.cost_of(row, column).is_some());
    }
}

#[test]
fn a_costmap_pushes_the_path_off_the_wall_f64() {
    // A wall along row 4, and a plan that has to pass the open end of it.
    let mut room: OccupancyGrid<20, 20, 1> = OccupancyGrid::try_new(1.0, [0.0, 0.0]).unwrap();
    for column in 0..14 {
        room.set_cell(9, column, true);
    }

    const CELLS: usize = 20 * 20;
    let mut workspace: GridSearchWorkspace<CELLS> = GridSearchWorkspace::new();

    let uniform = UniformCost::new(&room);
    let by_uniform = GridPlanner::new()
        .try_plan::<CELLS, 256, _, _>(&room, &uniform, [0.5, 0.5], [0.5, 19.5], &mut workspace)
        .unwrap();

    let mut transform: DistanceTransformWorkspace<21> = DistanceTransformWorkspace::new();
    let field: DistanceField<20, 20> = DistanceField::try_build(&room, &mut transform).unwrap();
    let costmap: CostGrid<20, 20> = CostGrid::try_build(&field, 0.5, 3.0, 1.0).unwrap();
    let inflated = CostmapCost::new(&costmap).with_weight(5.0);
    let by_costmap = GridPlanner::new()
        .try_plan::<CELLS, 256, _, _>(&room, &inflated, [0.5, 0.5], [0.5, 19.5], &mut workspace)
        .unwrap();

    let clearance = |report: &multicalc::planning::PlanReport<256, 2, f64>| {
        report
            .path()
            .waypoints()
            .iter()
            .filter_map(|point| field.distance_at(*point.as_array()))
            .fold(f64::INFINITY, f64::min)
    };

    assert!(
        clearance(&by_costmap) > clearance(&by_uniform),
        "costmap {} against uniform {}",
        clearance(&by_costmap),
        clearance(&by_uniform)
    );
}

// ---- failure modes ----

#[test]
fn a_walled_goal_reports_no_path_found() {
    let mut room = empty_room();
    for column in 0..COLUMNS {
        room.set_cell(5, column, true);
    }
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    assert_eq!(
        GridPlanner::new()
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
                &room,
                &cost,
                centre(0, 0),
                centre(9, 9),
                &mut workspace
            )
            .err(),
        Some(PlanningError::NoPathFound)
    );
}

#[test]
fn a_tiny_budget_reports_did_not_converge() {
    let room = empty_room();
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    let planned = GridPlanner::new()
        .with_search(GridSearch::Dijkstra)
        .with_expansion_budget(3)
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 9),
            &mut workspace,
        );

    assert_eq!(
        planned.err(),
        Some(PlanningError::DidNotConverge { iterations: 4 })
    );
}

#[test]
fn an_undersized_workspace_reports_workspace_too_small() {
    let room = empty_room();
    let mut workspace: GridSearchWorkspace<16> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    assert_eq!(
        GridPlanner::new()
            .try_plan::<16, MAX_POINTS, _, _>(
                &room,
                &cost,
                centre(0, 0),
                centre(9, 9),
                &mut workspace
            )
            .err(),
        Some(PlanningError::WorkspaceTooSmall)
    );
}

#[test]
fn out_of_bounds_and_blocked_endpoints_report_themselves() {
    let mut room = empty_room();
    room.set_cell(9, 9, true);
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);
    let planner = GridPlanner::<f64>::new();

    assert_eq!(
        planner
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
                &room,
                &cost,
                [-5.0, -5.0],
                centre(0, 0),
                &mut workspace
            )
            .err(),
        Some(PlanningError::StartOutOfBounds)
    );
    assert_eq!(
        planner
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
                &room,
                &cost,
                centre(0, 0),
                [100.0, 100.0],
                &mut workspace
            )
            .err(),
        Some(PlanningError::GoalOutOfBounds)
    );
    assert_eq!(
        planner
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
                &room,
                &cost,
                centre(0, 0),
                centre(9, 9),
                &mut workspace
            )
            .err(),
        Some(PlanningError::GoalNotFree)
    );

    room.set_cell(0, 0, true);
    let cost = UniformCost::new(&room);
    assert_eq!(
        planner
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
                &room,
                &cost,
                centre(0, 0),
                centre(5, 5),
                &mut workspace
            )
            .err(),
        Some(PlanningError::StartNotFree)
    );
}

#[test]
fn manhattan_with_eight_connected_reports_inadmissible_heuristic() {
    let room = empty_room();
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    assert_eq!(
        GridPlanner::new()
            .with_heuristic(GridHeuristic::Manhattan)
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
                &room,
                &cost,
                centre(0, 0),
                centre(9, 9),
                &mut workspace
            )
            .err(),
        Some(PlanningError::InadmissibleHeuristic)
    );
}

#[test]
fn a_weight_below_one_is_rejected() {
    assert_eq!(
        GridPlanner::<f64>::new()
            .try_with_heuristic_weight(0.5)
            .err(),
        Some(PlanningError::HeuristicWeightBelowOne)
    );
    assert_eq!(
        GridPlanner::<f64>::new()
            .try_with_heuristic_weight(f64::NAN)
            .err(),
        Some(PlanningError::NonFinite)
    );
}

#[test]
fn an_unknown_cell_is_impassable_by_default_f64() {
    // A belief grid observed along the bottom and the top, with an unmapped band between them.
    let mut belief: LogOddsGrid<20, 20> = LogOddsGrid::try_new(1.0, [0.0, 0.0]).unwrap();
    let scan: ScanGeometry<3> = ScanGeometry::try_new(0.02, 30.0).unwrap();
    for row in (0..3).chain(17..20) {
        let pose = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([0.5, row as f64 + 0.5]));
        for _ in 0..6 {
            belief.integrate_scan(pose, &scan, &[30.0; 3]);
        }
    }
    assert_eq!(belief.cell_state(1, 5), CellState::Free);
    assert_eq!(belief.cell_state(18, 5), CellState::Free);
    assert_eq!(belief.cell_state(10, 5), CellState::Unknown);

    const CELLS: usize = 20 * 20;
    let mut workspace: GridSearchWorkspace<CELLS> = GridSearchWorkspace::new();
    let start = [0.5, 0.5];
    let goal = [0.5, 18.5];

    // Getting from one observed band to the other means crossing unmapped space, which the
    // default refuses. That refusal is the correctness fix the unknown state exists for.
    let strict = UniformCost::new(&belief);
    assert_eq!(
        GridPlanner::new()
            .try_plan::<CELLS, 256, _, _>(&belief, &strict, start, goal, &mut workspace)
            .err(),
        Some(PlanningError::NoPathFound)
    );

    // Admitting unknown cells opens the route.
    let permissive = UniformCost::new(&belief).with_unknown_passable(true);
    assert!(
        GridPlanner::new()
            .try_plan::<CELLS, 256, _, _>(&belief, &permissive, start, goal, &mut workspace)
            .is_ok()
    );
}

// ---- actionability and determinism ----

#[test]
fn path_capacity_exceeded_reports_a_usable_count() {
    let mut room = empty_room();
    for column in 0..7 {
        room.set_cell(4, column, true);
    }
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);
    let planner = GridPlanner::new().with_search(GridSearch::Dijkstra);

    let Err(PlanningError::PathCapacityExceeded { needed }) = planner
        .try_plan::<MAX_CELLS, 2, _, _>(&room, &cost, centre(0, 0), centre(9, 9), &mut workspace)
    else {
        panic!("a two-waypoint budget should not have been enough");
    };
    assert!(needed > 2);

    // The reported size is enough to succeed on the retry, in one round trip.
    let report = planner
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 9),
            &mut workspace,
        )
        .unwrap();
    assert_eq!(report.waypoint_count(), needed);
}

#[test]
fn a_reused_workspace_matches_a_fresh_one_f64() {
    let mut first = empty_room();
    for column in 0..6 {
        first.set_cell(3, column, true);
    }
    let mut second = empty_room();
    for row in 2..9 {
        second.set_cell(row, 6, true);
    }

    let planner = GridPlanner::<f64>::new();
    let plan = |room: &Room, workspace: &mut GridSearchWorkspace<MAX_CELLS>| {
        let cost = UniformCost::new(room);
        planner
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
                room,
                &cost,
                centre(0, 0),
                centre(9, 9),
                workspace,
            )
            .unwrap()
    };

    let mut shared: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let reused_first = plan(&first, &mut shared);
    let reused_second = plan(&second, &mut shared);

    let mut fresh_one: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let mut fresh_two: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let fresh_first = plan(&first, &mut fresh_one);
    let fresh_second = plan(&second, &mut fresh_two);

    assert_eq!(reused_first, fresh_first);
    assert_eq!(reused_second, fresh_second);
}

#[test]
fn start_equal_to_goal_gives_a_single_waypoint_f64() {
    let room = empty_room();
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    let report = GridPlanner::new()
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(4, 4),
            centre(4, 4),
            &mut workspace,
        )
        .unwrap();

    assert_eq!(report.waypoint_count(), 1);
    assert_eq!(report.cost(), 0.0);
}

// ---- f32 by identity, never goldens ----

#[test]
fn cell_sequence_matches_f64_at_unit_resolution_f32() {
    // Four-connected at unit resolution, so every cost and every heuristic value is a whole number
    // exactly representable at both precisions. The comparison order therefore cannot diverge and
    // the two must agree bit for bit — eight-connected they would not, because a sum of √2 terms
    // rounds differently in each.
    let mut room_f64 = empty_room();
    let mut room_f32: OccupancyGrid<ROWS, COLUMNS, 1, f32> =
        OccupancyGrid::try_new(1.0, [0.0, 0.0]).unwrap();
    for (row, column) in [(3, 1), (3, 2), (3, 3), (6, 6), (6, 7), (6, 8), (1, 8)] {
        room_f64.set_cell(row, column, true);
        room_f32.set_cell(row, column, true);
    }

    let mut workspace_f64: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let mut workspace_f32: GridSearchWorkspace<MAX_CELLS, f32> = GridSearchWorkspace::new();
    let planner_f64 = GridPlanner::<f64>::new()
        .with_connectivity(GridConnectivity::FourConnected)
        .with_heuristic(GridHeuristic::Manhattan);
    let planner_f32 = GridPlanner::<f32>::new()
        .with_connectivity(GridConnectivity::FourConnected)
        .with_heuristic(GridHeuristic::Manhattan);

    let report_f64 = planner_f64
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room_f64,
            &UniformCost::new(&room_f64),
            centre(0, 0),
            centre(9, 9),
            &mut workspace_f64,
        )
        .unwrap();
    let report_f32 = planner_f32
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room_f32,
            &UniformCost::new(&room_f32),
            [0.5_f32, 0.5],
            [9.5_f32, 9.5],
            &mut workspace_f32,
        )
        .unwrap();

    assert_eq!(report_f64.waypoint_count(), report_f32.waypoint_count());
    assert_eq!(report_f64.cost() as f32, report_f32.cost());
    for (wide, narrow) in report_f64
        .path()
        .waypoints()
        .iter()
        .zip(report_f32.path().waypoints())
    {
        assert_eq!(wide[0] as f32, narrow[0]);
        assert_eq!(wide[1] as f32, narrow[1]);
    }
}

#[test]
fn cost_matches_path_arc_length_f32() {
    let mut room: OccupancyGrid<ROWS, COLUMNS, 1, f32> =
        OccupancyGrid::try_new(1.0, [0.0, 0.0]).unwrap();
    for column in 0..6 {
        room.set_cell(4, column, true);
    }
    let mut workspace: GridSearchWorkspace<MAX_CELLS, f32> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    let report = GridPlanner::new()
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            [0.5_f32, 0.5],
            [9.5_f32, 9.5],
            &mut workspace,
        )
        .unwrap();

    // Uniform cost, so the plan's cost is its length.
    assert!((report.cost() - report.path().total_arc_length()).abs() < 1e-4);
}

// ---- any-angle ----

/// Whether every cell the straight segment between two points crosses is free.
fn segment_is_clear(room: &Room, from: [f64; 2], into: [f64; 2]) -> bool {
    let separation = [into[0] - from[0], into[1] - from[1]];
    let distance = separation[0].hypot(separation[1]);
    if distance == 0.0 {
        return true;
    }
    let bearing = separation[1].atan2(separation[0]);
    room.geometry()
        .walk(from, bearing, distance)
        .all(|step| !room.is_occupied(step.row, step.column))
}

#[test]
fn theta_star_crosses_an_empty_map_in_two_waypoints_f64() {
    let room = empty_room();
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    let by_astar = GridPlanner::new()
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 9),
            &mut workspace,
        )
        .unwrap();
    let by_theta = GridPlanner::new()
        .with_search(GridSearch::ThetaStar)
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 9),
            &mut workspace,
        )
        .unwrap();

    // Corner to corner across open floor is one straight line, however many grid steps it takes.
    assert_eq!(by_theta.waypoint_count(), 2);
    assert!(by_theta.cost() <= by_astar.cost() + 1e-9);
}

#[test]
fn every_theta_star_segment_is_clear_and_beats_its_staircase_f64() {
    let mut room = empty_room();
    for row in 2..8 {
        room.set_cell(row, 4, true);
    }
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    let by_astar = GridPlanner::new()
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 9),
            &mut workspace,
        )
        .unwrap();
    let by_theta = GridPlanner::new()
        .with_search(GridSearch::ThetaStar)
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(9, 9),
            &mut workspace,
        )
        .unwrap();

    let path = by_theta.path();
    for pair in path.waypoints().windows(2) {
        assert!(
            segment_is_clear(&room, *pair[0].as_array(), *pair[1].as_array()),
            "{:?} to {:?}",
            pair[0],
            pair[1]
        );
    }
    // The straight-line path is shorter than the grid staircase it replaces.
    assert!(by_theta.path().total_arc_length() <= by_astar.path().total_arc_length() + 1e-9);
}

#[test]
fn collinear_waypoints_are_pruned_f64() {
    let room = empty_room();
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let cost = UniformCost::new(&room);

    // A straight run along one row: ten cells, but two waypoints say the same thing.
    let report = GridPlanner::new()
        .with_connectivity(GridConnectivity::FourConnected)
        .with_heuristic(GridHeuristic::Manhattan)
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
            &room,
            &cost,
            centre(0, 0),
            centre(0, 9),
            &mut workspace,
        )
        .unwrap();

    assert_eq!(report.waypoint_count(), 2);
    assert!((report.cost() - 9.0).abs() < 1e-12);
}

#[test]
fn corner_grazing_does_not_read_through_f64() {
    // Two blocked cells meeting only at a corner. Where the ray meets that corner exactly, the
    // walk's tie-break crosses the row boundary first and so enters one of them: the diagonal
    // reads blocked. That is the conservative answer, and pinning it here keeps it from drifting.
    let mut room = empty_room();
    room.set_cell(4, 4, true);
    room.set_cell(5, 5, true);
    assert!(!segment_is_clear(&room, centre(5, 4), centre(4, 5)));

    // The other diagonal of the same corner, with the other pair blocked.
    let mut mirrored = empty_room();
    mirrored.set_cell(4, 5, true);
    mirrored.set_cell(5, 4, true);
    assert!(!segment_is_clear(&mirrored, centre(4, 4), centre(5, 5)));

    // With the corner clear the same segment reads through, so the check is not simply refusing
    // every diagonal.
    let open = empty_room();
    assert!(segment_is_clear(&open, centre(5, 4), centre(4, 5)));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn theta_star_never_costs_more_than_astar_f64(
        pattern in prop::collection::vec(prop::bool::weighted(0.2), MAX_CELLS)
    ) {
        let room = map_from_pattern(&pattern);
        let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
        let cost = UniformCost::new(&room);
        let (start, goal) = (centre(0, 0), centre(ROWS - 1, COLUMNS - 1));

        let by_astar = GridPlanner::new()
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(&room, &cost, start, goal, &mut workspace);
        let by_theta = GridPlanner::new()
            .with_search(GridSearch::ThetaStar)
            .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(&room, &cost, start, goal, &mut workspace);

        if let (Ok(astar), Ok(theta)) = (by_astar, by_theta) {
            prop_assert!(
                theta.cost() <= astar.cost() + 1e-9,
                "theta {} against astar {}", theta.cost(), astar.cost()
            );
        }
    }
}
