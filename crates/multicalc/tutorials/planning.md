# Planning

Route search over a grid map, or over a continuous state space by sampling.

- `GridPlanner`: Dijkstra, A\*, weighted A\* and any-angle Theta\* over an `OccupancyMap`.
- `TraversalCost`: the cost to enter a cell, with `UniformCost` over a plain map and `CostmapCost`
  over an inflation `CostGrid`.
- `StateSpace` and `StateValidity`: the space of states and which states are free, with `BoxSpace`
  over a box in Euclidean space.
- `Rrt`, `RrtStar` and `Prm`: sampling planners over that space.
- `PlanReport`: the plan, its cost, and the search effort.

## Planning is off-loop

A search runs when the goal or the map changes, not on each tick. The control loop consumes the
`PolylinePath` that the search returns, through [`MinimumSnapPlanner`](motion.md) or
[`pure_pursuit_curvature`](control.md). A 128 by 128 grid search touches approximately 16 000 cells,
which is milliseconds of work and too much for a 1 kHz loop.

Each search runs in a **caller-owned workspace**, so nothing here allocates, and the caller sizes
and places the memory: a `static`, or a `Box` where `alloc` is available. See
[Sizing the workspaces](#sizing-the-workspaces) for the sizes.

## A first path

`MAX_CELLS` must be at least `rows · columns` of the map; write it as a `const` at the call site.
`MAX_POINTS` is the waypoint capacity of the returned path. If it is too small,
`PlanningError::PathCapacityExceeded { needed }` gives the necessary size, so one resize and retry
is sufficient.

```rust
use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid, OccupancyMap};
use multicalc::planning::{GridPlanner, GridSearchWorkspace, UniformCost};

// A 10 m square of 1 m cells with a wall up the middle, open at the top.
const ROWS: usize = 10;
const COLUMNS: usize = 10;
const MAX_CELLS: usize = ROWS * COLUMNS;

let mut room: OccupancyGrid<ROWS, COLUMNS, 1> = OccupancyGrid::try_new(1.0, [0.0, 0.0])?;
for row in 0..8 {
    room.set_cell(row, 5, true);
}

let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
let cost = UniformCost::new(&room);

let start = [0.5, 0.5];
let goal = [9.5, 0.5];
let report = GridPlanner::new()
    .try_plan::<MAX_CELLS, 64, _, _>(&room, &cost, start, goal, &mut workspace)?;

// The plan starts and ends at the requested points, and goes round the wall.
let path = report.path();
assert_eq!(path.waypoints().first().map(|point| *point.as_array()), Some(start));
assert_eq!(path.waypoints().last().map(|point| *point.as_array()), Some(goal));
assert!(report.cost() > 9.0);
# Ok::<(), multicalc::CalcError>(())
```

`report.iterations()` is the search effort: expansions for a grid search, samples for a sampling
planner. `report.cost()` is the plan cost under its traversal cost. Under a uniform cost, that is
the path length. A costmap scales each step by the cost of the cell it enters, so the cost is more
than the length where the path is near an obstacle.

## Choosing a search

`GridPlanner::new()` runs A\*, eight-connected, with the Octile heuristic at weight one, no corner
cutting, and a budget of one expansion per cell.

| Search | Property | Cost |
| --- | --- | --- |
| `Dijkstra` | Settles each reachable cell in cost order. Use it for the cost field, not for one route. | Expands the full reachable set. |
| `AStar` | The same optimal path, with a heuristic. The default. | No loss of optimality at weight one. |
| Weighted A\* | `try_with_heuristic_weight(weight)`: a path within `weight` of optimal, in fewer expansions. | Suboptimality, bounded by the weight. |
| `ThetaStar` | Any-angle: straight segments, not a grid staircase, and fewer waypoints. | A line-of-sight check per relaxation. |

`try_with_heuristic_weight` returns `PlanningError::HeuristicWeightBelowOne` for a weight below one.

The heuristics are `Manhattan` (`|Δrow| + |Δcolumn|`), `Octile` (`max + (√2 − 1)·min`, the exact
cost of an eight-connected straight run, the default) and `Euclidean`. **Manhattan with
eight-connected movement is inadmissible**: it can overestimate a diagonal run and return a
suboptimal path. `try_plan` rejects that pair with `PlanningError::InadmissibleHeuristic`.

```rust
# use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid};
# use multicalc::planning::{GridPlanner, GridSearchWorkspace, UniformCost};
use multicalc::planning::{GridConnectivity, GridHeuristic, GridSearch};

const MAX_CELLS: usize = 100;
let mut room: OccupancyGrid<10, 10, 1> = OccupancyGrid::try_new(1.0, [0.0, 0.0])?;
for row in 2..8 {
    room.set_cell(row, 4, true);
}
let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
let cost = UniformCost::new(&room);
let (start, goal) = ([0.5, 0.5], [9.5, 9.5]);

let by_astar = GridPlanner::new()
    .try_plan::<MAX_CELLS, 64, _, _>(&room, &cost, start, goal, &mut workspace)?;

// Weighted A*: the cost is within the weight of optimal.
let by_weighted = GridPlanner::new()
    .try_with_heuristic_weight(2.0)?
    .try_plan::<MAX_CELLS, 64, _, _>(&room, &cost, start, goal, &mut workspace)?;
assert!(by_weighted.cost() <= 2.0 * by_astar.cost() + 1e-12);

// Theta*: no higher cost, in fewer waypoints.
let by_theta = GridPlanner::new()
    .with_search(GridSearch::ThetaStar)
    .try_plan::<MAX_CELLS, 64, _, _>(&room, &cost, start, goal, &mut workspace)?;
assert!(by_theta.cost() <= by_astar.cost() + 1e-9);

// Four-connected movement with Manhattan, its admissible heuristic.
let four_connected = GridPlanner::new()
    .with_connectivity(GridConnectivity::FourConnected)
    .with_heuristic(GridHeuristic::Manhattan)
    .try_plan::<MAX_CELLS, 64, _, _>(&room, &cost, start, goal, &mut workspace)?;
assert!(four_connected.cost() >= by_astar.cost());
# Ok::<(), multicalc::CalcError>(())
```

By default, a diagonal step cannot pass between two occupied cells that meet at a corner, because no
robot with width can make that move. `with_corner_cutting(true)` allows it. The Theta\*
line-of-sight check is equally conservative: where a ray meets a cell corner exactly, it enters one
of the two neighbours and reports the ray blocked if that neighbour is occupied.

## Keeping off the walls

A uniform cost has no preference for the middle of a corridor, so its path can run along a wall. A
[`CostGrid`](mapping.md) inflated around the obstacles adds that preference.

```rust
use multicalc::mapping::{
    CostGrid, DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid,
};
use multicalc::planning::{CostmapCost, GridPlanner, GridSearchWorkspace, UniformCost};

const ROWS: usize = 20;
const COLUMNS: usize = 20;
const MAX_CELLS: usize = ROWS * COLUMNS;

let mut room: OccupancyGrid<ROWS, COLUMNS, 1> = OccupancyGrid::try_new(1.0, [0.0, 0.0])?;
for column in 0..14 {
    room.set_cell(9, column, true);
}

let mut transform: DistanceTransformWorkspace<21> = DistanceTransformWorkspace::new();
let field: DistanceField<ROWS, COLUMNS> = DistanceField::try_build(&room, &mut transform)?;
let costmap: CostGrid<ROWS, COLUMNS> = CostGrid::try_build(&field, 0.5, 3.0, 1.0)?;

let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
let (start, goal) = ([0.5, 0.5], [0.5, 19.5]);

let hugging = GridPlanner::new().try_plan::<MAX_CELLS, 256, _, _>(
    &room, &UniformCost::new(&room), start, goal, &mut workspace,
)?;
let clear_of_it = GridPlanner::new().try_plan::<MAX_CELLS, 256, _, _>(
    &room, &CostmapCost::new(&costmap).with_weight(5.0), start, goal, &mut workspace,
)?;

// The inflated plan has more clearance from the wall, measured with the field.
let clearance = |report: &multicalc::planning::PlanReport<256, 2, f64>| {
    report.path().waypoints().iter()
        .filter_map(|point| field.distance_at(*point.as_array()))
        .fold(f64::INFINITY, f64::min)
};
assert!(clearance(&clear_of_it) > clearance(&hugging));
# Ok::<(), multicalc::CalcError>(())
```

**An unknown cell is impassable by default.** A [`LogOddsGrid`](mapping.md) reports `Unknown` for
space that no sensor saw, and `UniformCost` does not route through it. A planner that treats
unmapped space as free drives into it. `with_unknown_passable(true)` allows unknown cells, for
exploration.

## From a plan to a trajectory

The full chain: plan a route, smooth it, time-parameterize it, and follow it.

```rust
use multicalc::control::pure_pursuit_curvature;
use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid};
use multicalc::motion::{MinimumSnapPlanner, durations_from_average_speed};
use multicalc::planning::{GridPlanner, GridSearch, GridSearchWorkspace, UniformCost};
use multicalc::{SE2, SO2, Vector2D};

const MAX_CELLS: usize = 100;
const MAX_POINTS: usize = 32;

let mut room: OccupancyGrid<10, 10, 1> = OccupancyGrid::try_new(1.0, [0.0, 0.0])?;
for row in 2..8 {
    room.set_cell(row, 4, true);
}
let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();

// Theta*, because a smoother needs straight segments: a grid staircase gives it false corners.
let report = GridPlanner::new()
    .with_search(GridSearch::ThetaStar)
    .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
        &room, &UniformCost::new(&room), [0.5, 0.5], [9.5, 9.5], &mut workspace,
    )?;
let path = report.path();
let segments = path.len() - 1;
assert!(segments >= 1);

// Initial segment durations, then one polynomial per segment.
let mut durations = [0.0; MAX_POINTS];
durations_from_average_speed(path.waypoints(), 1.0, &mut durations[..segments])?;
// Three free derivatives per interior waypoint, so `3·(segments − 1)` in total.
let planner = MinimumSnapPlanner::<MAX_POINTS, 64, 2, f64>::new();
let trajectory = planner.plan(path.waypoints(), &durations[..segments])?;

// Position and its derivatives at any time on the trajectory.
let [position, velocity, _acceleration] = trajectory.evaluate_with_derivatives::<3>(0.5)?;
assert!(position.is_finite() && velocity.norm() >= 0.0);

// A path follower steers to a lookahead point on the original waypoints.
let lookahead_distance = 1.5;
let start_pose = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([0.5, 0.5]));
let projection = path.closest_point(Vector2D::new([0.5, 0.5]))?;
let target = path.lookahead_point(projection.arc_length(), lookahead_distance)?;
let curvature = pure_pursuit_curvature(start_pose, target, lookahead_distance)?;
assert!(curvature.value().is_finite());
# Ok::<(), multicalc::CalcError>(())
```

## Planning for a robot arm

A grid does not suit a seven-joint arm: the configuration space has seven dimensions, and the
obstacles are in the workspace, not in the configuration space. Sampling planners draw
configurations and query a validity oracle for each one.

`StateValidity` is that oracle. It has a blanket implementation over `Fn(&Vector<N, T>) -> bool`, so
the usual oracle is a closure:

```rust,ignore
let is_state_valid = |configuration: &Vector<MAX_CONFIG, f64>| {
    tree.forward_kinematics(configuration)
        .and_then(|state| query.check(&state))
        .is_ok_and(|report| report.is_clear(clearance))
};
```

It returns `bool`, not `Result`. The fallible parts of a real oracle fail only on construction bugs,
never per state, and a generic error would add a type parameter to each `try_plan` with no useful
action for the planner. If a check cannot be evaluated, the oracle must report the state invalid.

`StateSpace` defines where the configurations are. `BoxSpace` covers joint limits in Euclidean
space:

```rust
use multicalc::planning::{BoxSpace, Rrt, RrtWorkspace, StateSpace};
use multicalc::{Pcg32, Vector};

// The limits of a two-joint arm, and an obstacle that excludes a band of configurations.
let space: BoxSpace<2> =
    BoxSpace::try_new(Vector::new([-3.0, -3.0]), Vector::new([3.0, 3.0]))?;
let is_state_valid = |configuration: &Vector<2, f64>| {
    let in_the_band = (configuration[1]).abs() < 0.3;
    !(in_the_band && configuration[0] < 2.0)
};

let mut workspace: RrtWorkspace<3000, 2> = RrtWorkspace::new();
let mut source = Pcg32::new(20260830);
let report = Rrt::new()
    .try_with_step_size(0.3)?
    .try_with_goal_tolerance(0.3)?
    .try_plan::<3000, 256, _, _, _>(
        &space,
        &is_state_valid,
        Vector::new([-2.0, -2.0]),
        Vector::new([-2.0, 2.0]),
        &mut source,
        &mut workspace,
    )?;

for waypoint in report.path().waypoints() {
    assert!(is_state_valid(waypoint));
}
# Ok::<(), multicalc::CalcError>(())
```

**`StateSpace` is a trait for correctness, not for flexibility.** A continuous joint at `+179°` and
`−179°` is `2°` apart, but raw subtraction gives `358°` and interpolates the long way round. A
planner that assumes Euclidean coordinates returns wrong nearest neighbours for such joints. Thus a
wrapped joint needs its own `StateSpace` implementation, of approximately thirty lines:

- `sample` draws each actuated joint across its limits.
- `distance` uses the configuration metric of the kinematic tree.
- `interpolate` is a per-axis lerp, with `wrap_to_pi` on the continuous joints.

That implementation is in user code, not in the library, so the crate has no
`planning → kinematics` dependency.

Edge validation is **discrete**: it tests a fixed number of interior stations on a segment (default
8) plus the far end. It misses an obstacle thinner than the station spacing. Where that matters,
increase `with_edge_checks`.

## RRT\*, and PRM

The three sampling planners give different guarantees, so they are three types, not one type with a
mode:

- **`Rrt`** returns the **first** path it finds. It is the cheapest, and the path is the first branch
  that reaches the goal. An exhausted budget returns `DidNotConverge`.
- **`RrtStar`** returns the **best** path within its sample budget. Each new node takes the
  cheapest reachable parent and rewires each neighbour that it can improve, so the cost decreases as
  the budget increases. It does not stop at the first contact with the goal. For RRT\* the full
  budget is the plan, so a failure to reach the goal returns `NoPathFound`, not `DidNotConverge`.
- **`Prm`** builds a roadmap once and answers many queries against it. Use it where the obstacles
  are static and the queries repeat. For a single query, a tree planner is better. If the node
  array fills, `try_build` stops sampling with no error and gives a sparser roadmap. If the edge
  array fills, it returns `WorkspaceTooSmall`.

```rust
use multicalc::planning::{BoxSpace, Prm, PrmWorkspace};
use multicalc::{Pcg32, Vector};

let space: BoxSpace<2> = BoxSpace::try_new(Vector::new([0.0, 0.0]), Vector::new([4.0, 4.0]))?;
let anywhere = |_state: &Vector<2, f64>| true;

let mut workspace: PrmWorkspace<400, 8000, 2> = PrmWorkspace::new();
let mut source = Pcg32::new(20260830);
let roadmap = Prm::new().try_with_connection_radius(0.8)?.with_sample_count(300);
roadmap.try_build(&space, &anywhere, &mut source, &mut workspace)?;

let nodes = workspace.node_count();
for index in 0..20 {
    let offset = index as f64 * 0.15;
    roadmap.try_query::<400, 8000, 256, _, _>(
        &space,
        &anywhere,
        Vector::new([0.2 + offset, 0.2]),
        Vector::new([3.8 - offset, 3.8]),
        &mut workspace,
    )?;
}
// After twenty queries, the roadmap is the same as it was after the build.
assert_eq!(workspace.node_count(), nodes);
# Ok::<(), multicalc::CalcError>(())
```

Nearest-neighbour search is a linear scan: O(n) per call and O(n²) per build. At 2000 nodes, that is
approximately 4 M distance evaluations, milliseconds off-loop. A k-d tree is *not* a valid upgrade:
its axis-split rule assumes a Euclidean product space. That is false for the wrapped and
manifold-valued joints that `StateSpace` supports, so the results would be wrong for those states.

A fixed seed reproduces a plan exactly. Each iteration uses one draw, whether the goal bias takes it
or not, so the same `Pcg32` seed gives bit-identical waypoints.

## Sizing the workspaces

The workspaces are the largest stack objects in the crate, so they are caller-owned: a workspace can
be in a `static`, or in a `Box` where `alloc` is available, and not on the stack.

`GridSearchWorkspace` is `2·size_of::<T>() + 13` bytes per cell plus padding:

| Map | `f32` | `f64` |
| --- | --- | --- |
| 64 × 64 | ≈ 86 KB | ≈ 135 KB |
| 128 × 128 | ≈ 344 KB | ≈ 541 KB |
| 256 × 256 | ≈ 1.4 MB | ≈ 2.2 MB |

`RrtWorkspace` is `DIMENSION·size_of::<T>() + 12` bytes per node at `f64`:

| Nodes | 2-D `f64` | 7-D `f64` |
| --- | --- | --- |
| 1 000 | ≈ 28 KB | ≈ 68 KB |
| 4 000 | ≈ 112 KB | ≈ 272 KB |

`PrmWorkspace` adds `2·size_of::<T>() + 12` bytes per node for its search arrays and
`size_of::<T>() + 8` bytes per edge for the edge list.

## Errors

Errors are [`PlanningError`](error-handling.md). The important distinctions:

- `NoPathFound` and `DidNotConverge { iterations }`: for Dijkstra and A\*, `NoPathFound` is a *proof*
  that the goal is not reachable. `DidNotConverge` means that the budget stopped the search, and
  gives the effort spent.
- `StartNotFree` and `GoalNotFree`: an endpoint is occupied. `StartOutOfBounds` and
  `GoalOutOfBounds`: an endpoint is outside the map or the state space.
- `WorkspaceTooSmall`: more cells, nodes or edges than the workspace holds.
- `PathCapacityExceeded { needed }`: the plan has more waypoints than `MAX_POINTS`, and `needed` is
  the size that works.
- `InadmissibleHeuristic`: Manhattan with eight-connected movement.
- `NonFinite`, `NonPositiveParameter`, `InvalidGoalBias`, `HeuristicWeightBelowOne`,
  `BoundsReversed`, `MapTooLarge`: invalid configuration.

Full demo:
[path_planning.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/path_planning.rs).

## Next

[Mapping](mapping.md) gives the maps, the distance field and the costmap;
[Motion](motion.md) smooths a plan and time-parameterizes it;
[Control](control.md) follows it.

---

[Back to the tutorial index](README.md)
