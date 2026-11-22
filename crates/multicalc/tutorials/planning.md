# Planning

Working out a route: a search over a grid map, or over a continuous state space by sampling.

- `GridPlanner`: Dijkstra, A\*, weighted A\* and any-angle Theta\* over an `OccupancyMap`.
- `TraversalCost`: what entering a cell costs, with `UniformCost` over a plain map and
  `CostmapCost` over an inflation `CostGrid`.
- `StateSpace` and `StateValidity`: where states live and which of them are free, with `BoxSpace`
  over a box in Euclidean space.
- `Rrt`, `RrtStar` and `Prm`: sampling planners over that space.
- `PlanReport`: the plan, its cost, and what the search spent finding it.

## Planning is off-loop

A search runs when the goal or the map changes — not every tick. What the control loop consumes is
the `PolylinePath` that comes back, through
[`MinimumSnapPlanner`](motion.md) or [`pure_pursuit_curvature`](control.md). Sizing the search that
way is deliberate: a 128 by 128 grid search touches sixteen thousand cells, which is milliseconds of
work and nothing a 1 kHz loop should be asked to do.

Every search runs in a **caller-owned workspace**, so nothing here allocates and the memory is sized
and placed by you — a `static`, or a `Box` where `alloc` is available. See
[Sizing the workspaces](#sizing-the-workspaces) for what that costs.

## A first path

`MAX_CELLS` must be at least the map's `rows · columns`; the natural way to write it is a `const` at
the call site. `MAX_POINTS` is the waypoint capacity of the path that comes back — if it is short,
`PlanningError::PathCapacityExceeded` says how many it would take, so you can resize and retry in
one round trip.

```rust
use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid, OccupancyMap};
use multicalc::planning::{GridPlanner, GridSearchWorkspace, UniformCost};

// A 10 m square of one-metre cells with a wall up the middle, open at the top.
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

// The plan starts and ends where it was asked to, and had to go round the wall.
let path = report.path();
assert_eq!(path.waypoints().first().map(|point| *point.as_array()), Some(start));
assert_eq!(path.waypoints().last().map(|point| *point.as_array()), Some(goal));
assert!(report.cost() > 9.0);
# Ok::<(), multicalc::CalcError>(())
```

`report.iterations()` is what the search spent — expansions for a grid search, samples drawn for a
sampling planner — and `report.cost()` is what the plan costs under the traversal cost it was
planned against. Under a uniform cost that is the path's length; a costmap scales each step by the
cell it enters, so the cost exceeds the length wherever the path runs near an obstacle.

## Choosing a search

| Search | What it buys | What it costs |
| --- | --- | --- |
| `Dijkstra` | Settles every reachable cell in cost order. Useful when you want the cost field, not one route. | Expands the whole reachable set. |
| `AStar` | The same optimal path, guided by a heuristic. The default. | Nothing, at weight one — it is optimal and strictly faster. |
| Weighted A\* | A path within `weight` of optimal, found sooner. | Suboptimality, bounded by the weight. |
| `ThetaStar` | Any-angle: straight segments rather than a grid staircase, and fewer waypoints. | A line-of-sight check per relaxation. |

Heuristics are `Manhattan`, `Octile` and `Euclidean`. **Manhattan paired with eight-connected
movement is inadmissible** — it can overestimate a diagonal run and return a path that is not
optimal — so that pairing is rejected outright with `PlanningError::InadmissibleHeuristic` rather
than quietly returning a worse answer. Octile is the exact cost of an eight-connected straight run
and is the default.

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

// Weighted A* trades optimality for speed, within the weight it was given.
let by_weighted = GridPlanner::new()
    .try_with_heuristic_weight(2.0)?
    .try_plan::<MAX_CELLS, 64, _, _>(&room, &cost, start, goal, &mut workspace)?;
assert!(by_weighted.cost() <= 2.0 * by_astar.cost() + 1e-12);

// Theta* costs no more and says it in fewer waypoints.
let by_theta = GridPlanner::new()
    .with_search(GridSearch::ThetaStar)
    .try_plan::<MAX_CELLS, 64, _, _>(&room, &cost, start, goal, &mut workspace)?;
assert!(by_theta.cost() <= by_astar.cost() + 1e-9);

// Four-connected movement needs a heuristic that never overestimates it.
let four_connected = GridPlanner::new()
    .with_connectivity(GridConnectivity::FourConnected)
    .with_heuristic(GridHeuristic::Manhattan)
    .try_plan::<MAX_CELLS, 64, _, _>(&room, &cost, start, goal, &mut workspace)?;
assert!(four_connected.cost() >= by_astar.cost());
# Ok::<(), multicalc::CalcError>(())
```

By default a diagonal step may not squeeze between two blocked cells meeting at a corner, which is a
move no robot with width can make; `with_corner_cutting(true)` allows it. The same conservatism
applies to Theta\*'s line-of-sight check: where a ray meets a cell corner exactly it enters one of
the two neighbours and reports it blocked if it is.

## Keeping off the walls

A uniform cost has no reason to prefer the middle of a corridor, so it will happily plan a path that
scrapes along a wall. A [`CostGrid`](mapping.md) inflated around the obstacles gives it one.

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

// The inflated plan keeps further from the wall, measured with the field itself.
let clearance = |report: &multicalc::planning::PlanReport<256, 2, f64>| {
    report.path().waypoints().iter()
        .filter_map(|point| field.distance_at(*point.as_array()))
        .fold(f64::INFINITY, f64::min)
};
assert!(clearance(&clear_of_it) > clearance(&hugging));
# Ok::<(), multicalc::CalcError>(())
```

**An unknown cell is impassable by default.** A [`LogOddsGrid`](mapping.md) reports `Unknown` for
space no sensor has seen, and `UniformCost` refuses to route through it, because a planner that
treats unmapped space as free will confidently drive into it. `with_unknown_passable(true)` opts back
in where exploring is the point.

## From a plan to a trajectory

This is the chain the module exists to close: plan a route, smooth it, give it a time
parameterization, and follow it.

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

// Theta* first, because its straight segments are what a smoother wants: a grid staircase hands
// the smoother corners that are not really there.
let report = GridPlanner::new()
    .with_search(GridSearch::ThetaStar)
    .try_plan::<MAX_CELLS, MAX_POINTS, _, _>(
        &room, &UniformCost::new(&room), [0.5, 0.5], [9.5, 9.5], &mut workspace,
    )?;
let path = report.path();
let segments = path.len() - 1;
assert!(segments >= 1);

// A first guess at how long each segment should take, then a polynomial per segment.
let mut durations = [0.0; MAX_POINTS];
durations_from_average_speed(path.waypoints(), 1.0, &mut durations[..segments])?;
// Three free derivatives a joint, so `3·(segments − 1)` of them.
let planner = MinimumSnapPlanner::<MAX_POINTS, 64, 2, f64>::new();
let trajectory = planner.plan(path.waypoints(), &durations[..segments])?;

// The trajectory gives position and its derivatives at any time along it.
let [position, velocity, _acceleration] = trajectory.evaluate_with_derivatives::<3>(0.5)?;
assert!(position.is_finite() && velocity.norm() >= 0.0);

// And a path follower steers to a lookahead point on the original waypoints.
let lookahead_distance = 1.5;
let start_pose = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([0.5, 0.5]));
let projection = path.closest_point(Vector2D::new([0.5, 0.5]))?;
let target = path.lookahead_point(projection.arc_length(), lookahead_distance)?;
let curvature = pure_pursuit_curvature(start_pose, target, lookahead_distance)?;
assert!(curvature.value().is_finite());
# Ok::<(), multicalc::CalcError>(())
```

## Planning for a robot arm

A grid is the wrong shape for a seven-joint arm: the configuration space is seven-dimensional and
the obstacles live in the workspace, not in it. Sampling planners work the other way round — they
draw configurations and ask an oracle whether each one is free.

`StateValidity` is that oracle, and its blanket implementation over `Fn(&Vector<N, T>) -> bool`
means the everyday case is a closure:

```rust,ignore
let is_state_valid = |configuration: &Vector<MAX_CONFIG, f64>| {
    tree.forward_kinematics(configuration)
        .and_then(|state| query.check(&state))
        .is_ok_and(|report| report.is_clear(clearance))
};
```

It answers `bool` rather than `Result` deliberately: the fallible parts of a real oracle fail only
on construction bugs, never per state, and a generic error would add a type parameter to every
`try_plan` while giving the planner nothing useful to do with it. A check that cannot be evaluated
must report the state invalid.

`StateSpace` says where the configurations live. `BoxSpace` covers joint limits in Euclidean space:

```rust
use multicalc::planning::{BoxSpace, Rrt, RrtWorkspace, StateSpace};
use multicalc::{Pcg32, Vector};

// A two-joint arm's limits, and an obstacle that rules out a band of configurations.
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

**`StateSpace` is a trait on correctness grounds, not for flexibility.** A continuous joint at
`+179°` and `−179°` is `2°` apart, but raw coordinate subtraction makes it `358°` and interpolates
the long way round. A planner assuming Euclidean coordinates returns wrong nearest neighbours on
exactly the arm configurations this crate cares about, so a wrapped joint needs its own `distance`
and `interpolate` — about thirty lines: `sample` redrawing each actuated joint across its limits,
`distance` delegating to the tree's own configuration metric, and `interpolate` a per-axis lerp with
`wrap_to_pi` on the continuous joints.

Keeping that implementation in your code rather than in the library is what keeps a
`planning → kinematics` edge out of the crate's layering.

Edge checking is **discrete**: `edge_is_valid` tests a fixed number of stations along a segment plus
its far end, so an obstacle thinner than the spacing between two stations is missed. Raise
`with_edge_checks` where that matters.

## RRT\*, and PRM

The three sampling planners promise different things, which is why they are three types rather than
one with a mode:

- **`Rrt`** returns the **first** path it finds. Cheapest, and the answer is whatever the tree
  happened to reach. An exhausted budget is `DidNotConverge`.
- **`RrtStar`** returns the **best** path within its sample budget. Each new node picks the cheapest
  reachable parent and rewires any neighbour it can improve, so cost falls as the budget grows — it
  does not stop on first contact with the goal. For it the budget *is* the plan, so failing to reach
  the goal is `NoPathFound` rather than `DidNotConverge`.
- **`Prm`** builds a roadmap once and answers many queries against it. Worth its build cost where
  the obstacles stay put and the queries keep coming; a tree planner is the better answer for a
  single query.

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
// Twenty queries later the roadmap is exactly as it was built. That reuse is the whole point.
assert_eq!(workspace.node_count(), nodes);
# Ok::<(), multicalc::CalcError>(())
```

Nearest-neighbour search is a linear scan, so O(n) a call and O(n²) over a build: at 2000 nodes that
is about 4 M distance evaluations, milliseconds off-loop. A k-d tree is *not* the upgrade — its
axis-split rule assumes a Euclidean product space, which is false for the wrapped and
manifold-valued joints `StateSpace` exists to support, so it would make the answer wrong on exactly
those states.

A fixed seed reproduces a plan exactly: one draw is spent per iteration whether or not the goal bias
takes it, so the same `Pcg32` seed gives bit-identical waypoints.

## Sizing the workspaces

These are the largest stack objects in the crate, which is precisely why they are caller-owned: a
workspace can live in a `static`, or in a `Box` where `alloc` is available, rather than on the stack.

`GridSearchWorkspace` is `2·size_of::<T>() + 13` bytes a cell plus padding:

| Map | `f32` | `f64` |
| --- | --- | --- |
| 64 × 64 | ≈ 86 KB | ≈ 135 KB |
| 128 × 128 | ≈ 344 KB | ≈ 541 KB |
| 256 × 256 | ≈ 1.4 MB | ≈ 2.2 MB |

`RrtWorkspace` is `DIMENSION·size_of::<T>() + 12` bytes a node:

| Nodes | 2-D `f64` | 7-D `f64` |
| --- | --- | --- |
| 1 000 | ≈ 28 KB | ≈ 68 KB |
| 4 000 | ≈ 112 KB | ≈ 272 KB |

`PrmWorkspace` adds `2·size_of::<T>() + 12` bytes a node for its search arrays and
`size_of::<T>() + 8` an edge for the edge list.

## Errors

Errors are [`PlanningError`](error-handling.md). The ones worth knowing apart:

- `NoPathFound` versus `DidNotConverge` — the first is a *proof* that nothing reachable is the goal,
  for Dijkstra and A\*; the second means the search was cut short by its budget and says how much it
  spent.
- `StartNotFree` and `GoalNotFree` — an endpoint is blocked, so no plan can leave or reach it.
- `WorkspaceTooSmall` — more cells, nodes or edges than the workspace holds.
- `PathCapacityExceeded { needed }` — the plan is longer than `MAX_POINTS`, and `needed` is the size
  that would work, so a resize and retry is one round trip.
- `InadmissibleHeuristic` — Manhattan with eight-connected movement, rejected rather than silently
  returning a suboptimal path.

## Next

[Mapping](mapping.md) is where the maps, the distance field and the costmap come from;
[Motion](motion.md) is what smooths a plan and gives it a time parameterization;
[Control](control.md) is what follows it.
