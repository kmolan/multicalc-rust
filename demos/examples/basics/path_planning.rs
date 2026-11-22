//! Grid and sampling path planning across a maze: A\* against Dijkstra, any-angle Theta\*, a
//! costmap that pushes the route off the walls, a seeded RRT that reproduces itself exactly, and
//! the plan fed onward to a smoother, a motion profile and a path follower.
//!
//! Run with: `cargo run -p multicalc-demos --example path_planning`

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use multicalc::control::pure_pursuit_curvature;
use multicalc::mapping::{
    CostGrid, DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid,
};
use multicalc::motion::{
    MinimumSnapPlanner, MotionProfilePlanner, ProfileLimits, durations_from_average_speed,
};
use multicalc::planning::{
    BoxSpace, CostmapCost, GridPlanner, GridSearch, GridSearchWorkspace, PlanReport, Rrt,
    RrtWorkspace, UniformCost,
};
use multicalc::{Pcg32, SE2, SO2, Vector, Vector2D};

const ROWS: usize = 64;
const COLUMNS: usize = 64;
const WORDS_PER_ROW: usize = 2;
const MAX_CELLS: usize = ROWS * COLUMNS;
const MAX_POINTS: usize = 512;
const SEED: u64 = 20260830;

type Maze = OccupancyGrid<ROWS, COLUMNS, WORDS_PER_ROW>;

fn check(label: &str, condition: bool) {
    assert!(condition, "{label}: failed");
    println!("  {label:<40} ok");
}

fn report(label: &str, value: f64) {
    println!("  {label:<40} = {value:>10.4}");
}

fn report_count(label: &str, value: usize) {
    println!("  {label:<40} = {value:>10}");
}

/// A 6.4 m square of 10 cm cells with three staggered walls, so a route has to weave.
fn maze() -> Maze {
    let mut map: Maze = OccupancyGrid::try_new(0.1, [0.0, 0.0]).unwrap();

    // Walls across the map, each leaving a gap at alternating ends.
    for column in 0..48 {
        map.set_cell(16, column, true);
    }
    for column in 16..64 {
        map.set_cell(32, column, true);
    }
    for column in 0..48 {
        map.set_cell(48, column, true);
    }
    map
}

/// The plan's smallest clearance from any obstacle, measured with the distance field.
fn minimum_clearance(
    report: &PlanReport<MAX_POINTS, 2, f64>,
    field: &DistanceField<ROWS, COLUMNS>,
) -> f64 {
    report
        .path()
        .waypoints()
        .iter()
        .filter_map(|point| field.distance_at(*point.as_array()))
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let map = maze();
    let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
    let uniform = UniformCost::new(&map);

    // Bottom-left corner to top-left, which means threading all three walls.
    let start = [0.55, 0.55];
    let goal = [0.55, 6.25];

    println!("Grid search");

    let by_dijkstra = GridPlanner::new()
        .with_search(GridSearch::Dijkstra)
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(&map, &uniform, start, goal, &mut workspace)
        .expect("dijkstra");
    let by_astar = GridPlanner::new()
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(&map, &uniform, start, goal, &mut workspace)
        .expect("a-star");

    report("dijkstra cost", by_dijkstra.cost());
    report("a-star cost", by_astar.cost());
    report_count("dijkstra expansions", by_dijkstra.iterations());
    report_count("a-star expansions", by_astar.iterations());

    // A* is admissible at weight one, so it agrees with Dijkstra exactly — and the heuristic pays
    // for itself by settling far fewer cells.
    check(
        "a-star cost equals dijkstra",
        (by_astar.cost() - by_dijkstra.cost()).abs() < 1e-12,
    );
    check(
        "a-star expands fewer cells",
        by_astar.iterations() < by_dijkstra.iterations(),
    );

    println!("\nAny-angle search");

    let by_theta = GridPlanner::new()
        .with_search(GridSearch::ThetaStar)
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(&map, &uniform, start, goal, &mut workspace)
        .expect("theta-star");

    report("theta-star cost", by_theta.cost());
    report_count("a-star waypoints", by_astar.waypoint_count());
    report_count("theta-star waypoints", by_theta.waypoint_count());

    check(
        "theta-star costs no more than a-star",
        by_theta.cost() <= by_astar.cost() + 1e-9,
    );
    check(
        "theta-star needs fewer waypoints",
        by_theta.waypoint_count() < by_astar.waypoint_count(),
    );

    println!("\nCostmap clearance");

    let mut transform: DistanceTransformWorkspace<{ COLUMNS + 1 }> =
        DistanceTransformWorkspace::new();
    let field: DistanceField<ROWS, COLUMNS> =
        DistanceField::try_build(&map, &mut transform).expect("distance field");

    // Inflated by three cells beyond the robot's own radius.
    let inscribed_radius = 0.1;
    let inflation_radius = 0.4;
    let cost_scaling_factor = 4.0;
    let costmap: CostGrid<ROWS, COLUMNS> = CostGrid::try_build(
        &field,
        inscribed_radius,
        inflation_radius,
        cost_scaling_factor,
    )
    .expect("costmap");

    let inflated = CostmapCost::new(&costmap).with_weight(8.0);
    let by_costmap = GridPlanner::new()
        .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(&map, &inflated, start, goal, &mut workspace)
        .expect("costmap plan");

    let plain_clearance = minimum_clearance(&by_astar, &field);
    let inflated_clearance = minimum_clearance(&by_costmap, &field);
    report("uniform-cost minimum clearance", plain_clearance);
    report("costmap minimum clearance", inflated_clearance);

    check(
        "costmap keeps further from the walls",
        inflated_clearance > plain_clearance,
    );

    println!("\nSampling planner");

    // The same maze as a continuous space, with the map itself as the validity oracle.
    let space: BoxSpace<2> =
        BoxSpace::try_new(Vector::new([0.05, 0.05]), Vector::new([6.35, 6.35])).unwrap();
    let is_state_valid = |state: &Vector<2, f64>| {
        map.geometry()
            .cell_of([state[0], state[1]])
            .is_some_and(|(row, column)| !map_is_blocked(&map, row, column))
    };

    let planner = Rrt::new()
        .try_with_step_size(0.25)
        .unwrap()
        .try_with_goal_tolerance(0.25)
        .unwrap()
        .try_with_goal_bias(0.1)
        .unwrap()
        .with_iteration_budget(20_000);

    let plan_once = || {
        let mut arena: RrtWorkspace<8000, 2> = RrtWorkspace::new();
        let mut source = Pcg32::new(SEED);
        planner
            .try_plan::<8000, MAX_POINTS, _, _, _>(
                &space,
                &is_state_valid,
                Vector::new(start),
                Vector::new(goal),
                &mut source,
                &mut arena,
            )
            .expect("rrt")
    };

    let first = plan_once();
    let again = plan_once();
    report("rrt cost", first.cost());
    report_count("rrt samples drawn", first.iterations());

    check("rrt found a path", first.waypoint_count() >= 2);
    check("the same seed reproduces it exactly", first == again);

    println!("\nDownstream chain");

    // Theta*'s straight segments are what a smoother wants: a grid staircase hands it corners that
    // are not really there.
    let path = by_theta.path();
    let segments = path.len() - 1;
    let mut durations = [0.0; MAX_POINTS];
    durations_from_average_speed(path.waypoints(), 0.5, &mut durations[..segments])
        .expect("durations");

    // The smoother solves for three free derivatives a joint, so it needs `3·(segments − 1)` of
    // them; 64 covers the plans this maze produces.
    let smoother = MinimumSnapPlanner::<MAX_POINTS, 64, 2, f64>::new();
    let trajectory = smoother
        .plan(path.waypoints(), &durations[..segments])
        .expect("minimum snap");
    let [position, velocity, _acceleration] = trajectory
        .evaluate_with_derivatives::<3>(durations[0] * 0.5)
        .expect("evaluate");

    check("smoothed trajectory is finite", position.is_finite());
    report("speed at the first sample", velocity.norm());

    // A time parameterization for the whole run, within speed and acceleration limits.
    let limits = ProfileLimits::<f64>::try_new(0.8, 0.6, None).expect("limits");
    let profile = MotionProfilePlanner::new(limits)
        .plan(path.total_arc_length())
        .expect("profile");
    report("profile duration", profile.duration());
    check("profile covers the path", profile.duration() > 0.0);

    // And a path follower steering to a lookahead point on the plan.
    let lookahead_distance = 0.5;
    let pose = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new(start));
    let projection = path
        .closest_point(Vector2D::new(start))
        .expect("projection");
    let target = path
        .lookahead_point(projection.arc_length(), lookahead_distance)
        .expect("lookahead");
    let curvature = pure_pursuit_curvature(pose, target, lookahead_distance).expect("pure pursuit");

    report("steering curvature", curvature.value());
    check("curvature is finite", curvature.value().is_finite());

    println!("\nAll checks passed.");
}

/// Whether a cell is blocked, without needing the trait in scope at every call site.
fn map_is_blocked(map: &Maze, row: usize, column: usize) -> bool {
    use multicalc::mapping::OccupancyMap;
    map.is_occupied(row, column)
}
