//! The sampling tree planners: that they find a path, that every waypoint and segment is free,
//! that a seed reproduces a plan exactly, and that RRT\* keeps its costs consistent as it rewires.

use multicalc::error::PlanningError;
use multicalc::planning::{BoxSpace, Rrt, RrtStar, RrtWorkspace, StateSpace};
use multicalc::{Pcg32, Vector};

const MAX_NODES: usize = 3000;
const MAX_POINTS: usize = 512;
const SEED: u64 = 20260830;

fn empty_box() -> BoxSpace<2> {
    BoxSpace::try_new(Vector::new([0.0, 0.0]), Vector::new([4.0, 4.0])).unwrap()
}

/// Everything is free.
fn anywhere(_state: &Vector<2, f64>) -> bool {
    true
}

/// A wall across the middle with a gap at the right-hand end.
fn wide_corridor(state: &Vector<2, f64>) -> bool {
    let in_the_wall = (state[1] - 2.0).abs() < 0.2;
    !(in_the_wall && state[0] < 3.0)
}

fn planner() -> Rrt<2> {
    Rrt::new()
        .try_with_step_size(0.3)
        .unwrap()
        .try_with_goal_tolerance(0.3)
        .unwrap()
        .try_with_goal_bias(0.1)
        .unwrap()
}

#[test]
fn finds_a_path_in_an_empty_box_f64() {
    let space = empty_box();
    let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
    let mut source = Pcg32::new(SEED);

    let report = planner()
        .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
            &space,
            &anywhere,
            Vector::new([0.5, 0.5]),
            Vector::new([3.5, 3.5]),
            &mut source,
            &mut workspace,
        )
        .unwrap();

    assert!(report.waypoint_count() >= 2);
    // The plan cannot be shorter than the straight line between the two ends.
    let direct = space.distance(&Vector::new([0.5, 0.5]), &Vector::new([3.5, 3.5]));
    assert!(report.cost() >= direct - 0.3);
    assert!(report.iterations() > 0);
}

#[test]
fn every_waypoint_and_segment_is_valid_f64() {
    let space = empty_box();
    let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
    let mut source = Pcg32::new(SEED);

    let report = planner()
        .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
            &space,
            &wide_corridor,
            Vector::new([0.5, 0.5]),
            Vector::new([0.5, 3.5]),
            &mut source,
            &mut workspace,
        )
        .unwrap();

    let path = report.path();
    for point in path.waypoints() {
        assert!(space.contains(point), "{point:?}");
        assert!(wide_corridor(point), "{point:?}");
    }
    // Every segment passes the same discrete check the planner used to accept it.
    for pair in path.waypoints().windows(2) {
        for station in 1..=8 {
            let amount = station as f64 / 9.0;
            let between = space.interpolate(&pair[0], &pair[1], amount);
            assert!(wide_corridor(&between), "{between:?}");
        }
    }
}

#[test]
fn the_same_seed_reproduces_the_path_exactly_f64() {
    let space = empty_box();
    let plan = || {
        let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
        let mut source = Pcg32::new(SEED);
        planner()
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &wide_corridor,
                Vector::new([0.5, 0.5]),
                Vector::new([0.5, 3.5]),
                &mut source,
                &mut workspace,
            )
            .unwrap()
    };

    // Bit-identical, not within a tolerance.
    assert_eq!(plan(), plan());
}

#[test]
fn a_reused_workspace_matches_a_fresh_one_f64() {
    let space = empty_box();
    let mut shared: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();

    let plan = |workspace: &mut RrtWorkspace<MAX_NODES, 2>, goal: [f64; 2]| {
        let mut source = Pcg32::new(SEED);
        planner()
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &anywhere,
                Vector::new([0.5, 0.5]),
                Vector::new(goal),
                &mut source,
                workspace,
            )
            .unwrap()
    };

    let reused_first = plan(&mut shared, [3.5, 3.5]);
    let reused_second = plan(&mut shared, [3.5, 0.5]);

    let mut fresh_one: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
    let mut fresh_two: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
    assert_eq!(reused_first, plan(&mut fresh_one, [3.5, 3.5]));
    assert_eq!(reused_second, plan(&mut fresh_two, [3.5, 0.5]));
}

#[test]
fn bad_endpoints_and_settings_report_themselves() {
    let space = empty_box();
    let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
    let mut source = Pcg32::new(SEED);
    let blocked_start = |state: &Vector<2, f64>| state[0] > 1.0;

    assert_eq!(
        planner()
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &blocked_start,
                Vector::new([0.5, 0.5]),
                Vector::new([3.5, 3.5]),
                &mut source,
                &mut workspace
            )
            .err(),
        Some(PlanningError::StartNotFree)
    );
    assert_eq!(
        planner()
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &anywhere,
                Vector::new([0.5, 0.5]),
                Vector::new([9.0, 9.0]),
                &mut source,
                &mut workspace
            )
            .err(),
        Some(PlanningError::GoalOutOfBounds)
    );
    assert_eq!(
        Rrt::<2>::new().try_with_goal_bias(1.5).err(),
        Some(PlanningError::InvalidGoalBias)
    );
    assert_eq!(
        Rrt::<2>::new().try_with_goal_bias(-0.1).err(),
        Some(PlanningError::InvalidGoalBias)
    );
    assert_eq!(
        Rrt::<2>::new().try_with_step_size(0.0).err(),
        Some(PlanningError::NonPositiveParameter)
    );
    assert_eq!(
        Rrt::<2>::new().try_with_step_size(f64::NAN).err(),
        Some(PlanningError::NonFinite)
    );
}

#[test]
fn a_tiny_budget_reports_did_not_converge() {
    let space = empty_box();
    let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
    let mut source = Pcg32::new(SEED);

    let planned = planner()
        .with_iteration_budget(3)
        .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
            &space,
            &anywhere,
            Vector::new([0.5, 0.5]),
            Vector::new([3.5, 3.5]),
            &mut source,
            &mut workspace,
        );

    assert_eq!(
        planned.err(),
        Some(PlanningError::DidNotConverge { iterations: 3 })
    );
}

#[test]
fn a_one_node_arena_reports_workspace_too_small() {
    let space = empty_box();
    let mut workspace: RrtWorkspace<1, 2> = RrtWorkspace::new();
    let mut source = Pcg32::new(SEED);

    assert_eq!(
        planner()
            .try_plan::<1, MAX_POINTS, _, _, _>(
                &space,
                &anywhere,
                Vector::new([0.5, 0.5]),
                Vector::new([3.5, 3.5]),
                &mut source,
                &mut workspace
            )
            .err(),
        Some(PlanningError::WorkspaceTooSmall)
    );
}

#[test]
fn succeeds_on_a_fixed_seed_list_f64() {
    // A golden success count rather than a flaky "over 95%" threshold: if the sampler or the tree
    // changes, this moves and says so.
    let space = empty_box();
    let mut succeeded = 0;
    for seed in 0..200_u64 {
        let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
        let mut source = Pcg32::new(seed);
        let planned = planner()
            .with_iteration_budget(4000)
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &wide_corridor,
                Vector::new([0.5, 0.5]),
                Vector::new([0.5, 3.5]),
                &mut source,
                &mut workspace,
            );
        if planned.is_ok() {
            succeeded += 1;
        }
    }
    assert_eq!(succeeded, 200);
}

// ---- rrt star ----

fn star() -> RrtStar<2> {
    RrtStar::new()
        .try_with_step_size(0.3)
        .unwrap()
        .try_with_goal_tolerance(0.3)
        .unwrap()
        .try_with_goal_bias(0.1)
        .unwrap()
        .try_with_neighbour_radius(0.6)
        .unwrap()
}

#[test]
fn cost_does_not_increase_with_more_samples_f64() {
    let space = empty_box();
    let plan = |budget: usize| {
        let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
        let mut source = Pcg32::new(SEED);
        star()
            .with_iteration_budget(budget)
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &anywhere,
                Vector::new([0.5, 0.5]),
                Vector::new([3.5, 3.5]),
                &mut source,
                &mut workspace,
            )
            .unwrap()
    };

    // The same seed at N and 2N: more samples must never make the plan worse.
    let fewer = plan(400);
    let more = plan(800);
    assert!(
        more.cost() <= fewer.cost() + 1e-12,
        "{} against {}",
        more.cost(),
        fewer.cost()
    );
}

#[test]
fn rewiring_keeps_costs_consistent_f64() {
    // The invariant descendant propagation exists to hold: every node's cost is its parent's plus
    // the edge between them. Leaving a rewired node's descendants stale is the bug this catches.
    let space = empty_box();
    let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
    let mut source = Pcg32::new(SEED);

    star()
        .with_iteration_budget(1200)
        .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
            &space,
            &wide_corridor,
            Vector::new([0.5, 0.5]),
            Vector::new([0.5, 3.5]),
            &mut source,
            &mut workspace,
        )
        .unwrap();

    assert!(workspace.node_count() > 100);
    assert_eq!(workspace.node_cost(0), Some(0.0));
    for index in 1..workspace.node_count() {
        let Some(parent) = workspace.node_parent(index) else {
            panic!("node {index} has no parent but is not the root");
        };
        let state = workspace.node_state(index).unwrap();
        let parent_state = workspace.node_state(parent).unwrap();
        let expected = workspace.node_cost(parent).unwrap() + space.distance(&parent_state, &state);
        let held = workspace.node_cost(index).unwrap();
        assert!(
            (held - expected).abs() < 1e-12,
            "node {index}: {held} against {expected}"
        );
    }
}

#[test]
fn approaches_the_straight_line_in_an_empty_box_f64() {
    let space = empty_box();
    let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
    let mut source = Pcg32::new(SEED);
    let (start, goal) = (Vector::new([0.5, 0.5]), Vector::new([3.5, 3.5]));

    let report = star()
        .with_iteration_budget(2500)
        .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
            &space,
            &anywhere,
            start,
            goal,
            &mut source,
            &mut workspace,
        )
        .unwrap();

    // Within the goal tolerance the plan stops short of the goal itself, so the direct distance is
    // measured to where it actually ends.
    let direct = space.distance(&start, &goal);
    assert!(
        report.cost() <= direct * 1.05 + star_goal_tolerance(),
        "{} against {direct}",
        report.cost()
    );
}

fn star_goal_tolerance() -> f64 {
    0.3
}

#[test]
fn beats_plain_rrt_on_average_f64() {
    let space = empty_box();
    let (start, goal) = (Vector::new([0.5, 0.5]), Vector::new([3.5, 3.5]));
    let mut star_total = 0.0;
    let mut plain_total = 0.0;
    let mut counted = 0;

    for seed in 0..50_u64 {
        let mut star_workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
        let mut star_source = Pcg32::new(seed);
        let by_star = star()
            .with_iteration_budget(1200)
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &anywhere,
                start,
                goal,
                &mut star_source,
                &mut star_workspace,
            );

        let mut plain_workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
        let mut plain_source = Pcg32::new(seed);
        let by_plain = planner()
            .with_iteration_budget(1200)
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &anywhere,
                start,
                goal,
                &mut plain_source,
                &mut plain_workspace,
            );

        if let (Ok(star), Ok(plain)) = (by_star, by_plain) {
            star_total += star.cost();
            plain_total += plain.cost();
            counted += 1;
        }
    }

    assert!(counted > 40, "only {counted} seeds produced both plans");
    assert!(
        star_total < plain_total,
        "rrt* mean {} against rrt mean {}",
        star_total / counted as f64,
        plain_total / counted as f64
    );
}

#[test]
fn the_same_seed_reproduces_the_star_path_exactly_f64() {
    let space = empty_box();
    let plan = || {
        let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
        let mut source = Pcg32::new(SEED);
        star()
            .with_iteration_budget(600)
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &wide_corridor,
                Vector::new([0.5, 0.5]),
                Vector::new([0.5, 3.5]),
                &mut source,
                &mut workspace,
            )
            .unwrap()
    };

    assert_eq!(plan(), plan());
}

#[test]
fn a_budget_with_no_reachable_goal_reports_no_path_found() {
    // For RRT* an exhausted budget is the plan, so failing to reach the goal is `NoPathFound`
    // rather than `DidNotConverge` — the contract difference from plain RRT.
    let space = empty_box();
    let mut workspace: RrtWorkspace<MAX_NODES, 2> = RrtWorkspace::new();
    let mut source = Pcg32::new(SEED);
    let sealed_off = |state: &Vector<2, f64>| (state[1] - 2.0).abs() > 0.2;

    assert_eq!(
        star()
            .with_iteration_budget(500)
            .try_plan::<MAX_NODES, MAX_POINTS, _, _, _>(
                &space,
                &sealed_off,
                Vector::new([0.5, 0.5]),
                Vector::new([0.5, 3.5]),
                &mut source,
                &mut workspace
            )
            .err(),
        Some(PlanningError::NoPathFound)
    );
}
