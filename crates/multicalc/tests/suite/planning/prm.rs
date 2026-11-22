//! The roadmap planner: that a build then a query finds a path, and that one roadmap survives many
//! queries untouched — the reuse that is the whole point of a roadmap.

use multicalc::error::PlanningError;
use multicalc::planning::{BoxSpace, Prm, PrmWorkspace, StateSpace};
use multicalc::{Pcg32, Vector};

const MAX_NODES: usize = 400;
const MAX_EDGES: usize = 8000;
const MAX_POINTS: usize = 256;
const SEED: u64 = 20260830;

fn empty_box() -> BoxSpace<2> {
    BoxSpace::try_new(Vector::new([0.0, 0.0]), Vector::new([4.0, 4.0])).unwrap()
}

fn anywhere(_state: &Vector<2, f64>) -> bool {
    true
}

/// A wall across the middle with a gap at the right-hand end.
fn wide_corridor(state: &Vector<2, f64>) -> bool {
    let in_the_wall = (state[1] - 2.0).abs() < 0.2;
    !(in_the_wall && state[0] < 3.0)
}

/// A wall with no gap at all.
fn sealed_off(state: &Vector<2, f64>) -> bool {
    (state[1] - 2.0).abs() > 0.2
}

fn roadmap() -> Prm<2> {
    Prm::new()
        .try_with_connection_radius(0.8)
        .unwrap()
        .with_sample_count(300)
}

type Workspace = PrmWorkspace<MAX_NODES, MAX_EDGES, 2>;

#[test]
fn build_then_query_finds_a_path_f64() {
    let space = empty_box();
    let mut workspace: Workspace = PrmWorkspace::new();
    let mut source = Pcg32::new(SEED);

    roadmap()
        .try_build(&space, &anywhere, &mut source, &mut workspace)
        .unwrap();
    assert!(workspace.node_count() > 100);
    assert!(workspace.edge_count() > 100);

    let start = Vector::new([0.2, 0.2]);
    let goal = Vector::new([3.8, 3.8]);
    let report = roadmap()
        .try_query::<MAX_NODES, MAX_EDGES, MAX_POINTS, _, _>(
            &space,
            &anywhere,
            start,
            goal,
            &mut workspace,
        )
        .unwrap();

    let path = report.path();
    assert_eq!(path.waypoints().first(), Some(&start));
    assert_eq!(path.waypoints().last(), Some(&goal));
    assert!(report.iterations() > 0);
}

#[test]
fn path_cost_is_at_least_the_straight_line_distance_f64() {
    let space = empty_box();
    let mut workspace: Workspace = PrmWorkspace::new();
    let mut source = Pcg32::new(SEED);
    roadmap()
        .try_build(&space, &anywhere, &mut source, &mut workspace)
        .unwrap();

    let start = Vector::new([0.2, 0.2]);
    let goal = Vector::new([3.8, 3.8]);
    let report = roadmap()
        .try_query::<MAX_NODES, MAX_EDGES, MAX_POINTS, _, _>(
            &space,
            &anywhere,
            start,
            goal,
            &mut workspace,
        )
        .unwrap();

    assert!(report.cost() >= space.distance(&start, &goal) - 1e-12);
}

#[test]
fn one_roadmap_answers_many_queries_f64() {
    let space = empty_box();
    let mut workspace: Workspace = PrmWorkspace::new();
    let mut source = Pcg32::new(SEED);
    roadmap()
        .try_build(&space, &anywhere, &mut source, &mut workspace)
        .unwrap();

    let nodes_before = workspace.node_count();
    let edges_before = workspace.edge_count();

    for index in 0..20 {
        let offset = index as f64 * 0.15;
        let start = Vector::new([0.2 + offset, 0.2]);
        let goal = Vector::new([3.8 - offset, 3.8]);
        roadmap()
            .try_query::<MAX_NODES, MAX_EDGES, MAX_POINTS, _, _>(
                &space,
                &anywhere,
                start,
                goal,
                &mut workspace,
            )
            .unwrap();
    }

    // A query must leave the roadmap exactly as it found it.
    assert_eq!(workspace.node_count(), nodes_before);
    assert_eq!(workspace.edge_count(), edges_before);
}

#[test]
fn query_before_build_reports_no_path_found() {
    let space = empty_box();
    let mut workspace: Workspace = PrmWorkspace::new();

    assert_eq!(
        roadmap()
            .try_query::<MAX_NODES, MAX_EDGES, MAX_POINTS, _, _>(
                &space,
                &anywhere,
                Vector::new([0.2, 0.2]),
                Vector::new([3.8, 3.8]),
                &mut workspace
            )
            .err(),
        Some(PlanningError::NoPathFound)
    );
}

#[test]
fn every_roadmap_node_and_edge_is_valid_f64() {
    let space = empty_box();
    let mut workspace: Workspace = PrmWorkspace::new();
    let mut source = Pcg32::new(SEED);
    roadmap()
        .try_build(&space, &wide_corridor, &mut source, &mut workspace)
        .unwrap();

    for index in 0..workspace.node_count() {
        let node = workspace.node_state(index).unwrap();
        assert!(space.contains(&node), "{node:?}");
        assert!(wide_corridor(&node), "{node:?}");
    }
}

#[test]
fn a_blocked_corridor_reports_no_path_found_f64() {
    let space = empty_box();
    let mut workspace: Workspace = PrmWorkspace::new();
    let mut source = Pcg32::new(SEED);
    roadmap()
        .try_build(&space, &sealed_off, &mut source, &mut workspace)
        .unwrap();

    assert_eq!(
        roadmap()
            .try_query::<MAX_NODES, MAX_EDGES, MAX_POINTS, _, _>(
                &space,
                &sealed_off,
                Vector::new([0.5, 0.5]),
                Vector::new([0.5, 3.5]),
                &mut workspace
            )
            .err(),
        Some(PlanningError::NoPathFound)
    );
}

#[test]
fn a_path_exists_through_the_gap_f64() {
    let space = empty_box();
    let mut workspace: Workspace = PrmWorkspace::new();
    let mut source = Pcg32::new(SEED);
    roadmap()
        .try_build(&space, &wide_corridor, &mut source, &mut workspace)
        .unwrap();

    let report = roadmap()
        .try_query::<MAX_NODES, MAX_EDGES, MAX_POINTS, _, _>(
            &space,
            &wide_corridor,
            Vector::new([0.5, 0.5]),
            Vector::new([0.5, 3.5]),
            &mut workspace,
        )
        .unwrap();

    // The route has to detour to the open end, so it is much longer than the direct line.
    assert!(report.cost() > 3.0);
    for point in report.path().waypoints() {
        assert!(wide_corridor(point), "{point:?}");
    }
}

#[test]
fn edge_capacity_overflow_reports_workspace_too_small() {
    let space = empty_box();
    let mut workspace: PrmWorkspace<200, 16, 2> = PrmWorkspace::new();
    let mut source = Pcg32::new(SEED);

    // A generous connection radius over 200 samples needs far more than sixteen edges.
    assert_eq!(
        Prm::new()
            .try_with_connection_radius(2.0)
            .unwrap()
            .with_sample_count(200)
            .try_build(&space, &anywhere, &mut source, &mut workspace)
            .err(),
        Some(PlanningError::WorkspaceTooSmall)
    );
}

#[test]
fn the_same_seed_reproduces_the_roadmap_exactly_f64() {
    let space = empty_box();
    let build = || {
        let mut workspace: Workspace = PrmWorkspace::new();
        let mut source = Pcg32::new(SEED);
        roadmap()
            .try_build(&space, &wide_corridor, &mut source, &mut workspace)
            .unwrap();
        workspace
    };

    let first = build();
    let second = build();
    assert_eq!(first.node_count(), second.node_count());
    assert_eq!(first.edge_count(), second.edge_count());
    for index in 0..first.node_count() {
        assert_eq!(first.node_state(index), second.node_state(index));
    }
}

#[test]
fn bad_endpoints_and_settings_report_themselves() {
    let space = empty_box();
    let mut workspace: Workspace = PrmWorkspace::new();
    let mut source = Pcg32::new(SEED);
    roadmap()
        .try_build(&space, &anywhere, &mut source, &mut workspace)
        .unwrap();

    assert_eq!(
        roadmap()
            .try_query::<MAX_NODES, MAX_EDGES, MAX_POINTS, _, _>(
                &space,
                &anywhere,
                Vector::new([9.0, 9.0]),
                Vector::new([1.0, 1.0]),
                &mut workspace
            )
            .err(),
        Some(PlanningError::StartOutOfBounds)
    );
    assert_eq!(
        Prm::<2>::new().try_with_connection_radius(0.0).err(),
        Some(PlanningError::NonPositiveParameter)
    );
    assert_eq!(
        Prm::<2>::new().try_with_connection_radius(f64::NAN).err(),
        Some(PlanningError::NonFinite)
    );
}
