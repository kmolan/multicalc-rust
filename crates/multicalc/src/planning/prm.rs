#![deny(clippy::indexing_slicing)]

//! A probabilistic roadmap: sampled once over an environment, then queried many times.

use crate::error::PlanningError;
use crate::linear_algebra::Vector;
use crate::planning::frontier::Frontier;
use crate::planning::plan_report::PlanReport;
use crate::planning::sampling::{NO_PARENT, edge_is_valid, extract_tree_path};
use crate::planning::state_space::{StateSpace, StateValidity};
use crate::random::{RandomScalar, RandomSource};
use crate::scalar::Numeric;

/// The roadmap and the search over it, owned by the caller.
///
/// A query leaves the nodes and edges untouched, so one built roadmap answers many start-to-goal
/// pairs — that reuse is the whole point of a roadmap over a tree.
#[derive(Debug, Clone, Copy)]
pub struct PrmWorkspace<
    const MAX_NODES: usize,
    const MAX_EDGES: usize,
    const DIMENSION: usize,
    T: Numeric = f64,
> {
    states: [Vector<DIMENSION, T>; MAX_NODES],
    edge_from: [u32; MAX_EDGES],
    edge_to: [u32; MAX_EDGES],
    edge_cost: [T; MAX_EDGES],
    node_length: usize,
    edge_length: usize,
    cost_so_far: [T; MAX_NODES],
    parent: [u32; MAX_NODES],
    frontier: Frontier<MAX_NODES, T>,
}

impl<const MAX_NODES: usize, const MAX_EDGES: usize, const DIMENSION: usize, T: Numeric>
    PrmWorkspace<MAX_NODES, MAX_EDGES, DIMENSION, T>
{
    /// An empty roadmap.
    #[must_use]
    pub fn new() -> Self {
        PrmWorkspace {
            states: [Vector::zeros(); MAX_NODES],
            edge_from: [0; MAX_EDGES],
            edge_to: [0; MAX_EDGES],
            edge_cost: [T::ZERO; MAX_EDGES],
            node_length: 0,
            edge_length: 0,
            cost_so_far: [T::INFINITY; MAX_NODES],
            parent: [NO_PARENT; MAX_NODES],
            frontier: Frontier::new(),
        }
    }

    /// Throws the roadmap away.
    pub fn clear(&mut self) {
        self.node_length = 0;
        self.edge_length = 0;
        self.frontier.clear();
    }

    /// How many nodes the roadmap holds.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_length
    }

    /// How many undirected edges it holds.
    #[inline]
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edge_length
    }

    /// One roadmap node's state.
    #[inline]
    pub fn node_state(&self, index: usize) -> Option<Vector<DIMENSION, T>> {
        (index < self.node_length).then(|| self.states.get(index).copied())?
    }

    fn occupied(&self) -> &[Vector<DIMENSION, T>] {
        self.states.get(..self.node_length).unwrap_or(&[])
    }

    fn state_at(&self, index: usize) -> Option<Vector<DIMENSION, T>> {
        self.states.get(index).copied()
    }

    fn push_node(&mut self, state: Vector<DIMENSION, T>) -> Option<usize> {
        if self.node_length >= MAX_NODES {
            return None;
        }
        let index = self.node_length;
        *self.states.get_mut(index)? = state;
        self.node_length += 1;
        Some(index)
    }

    fn push_edge(&mut self, from: usize, into: usize, cost: T) -> Result<(), PlanningError> {
        if self.edge_length >= MAX_EDGES {
            return Err(PlanningError::WorkspaceTooSmall);
        }
        let index = self.edge_length;
        let (Some(stored_from), Some(stored_to), Some(stored_cost)) = (
            self.edge_from.get_mut(index),
            self.edge_to.get_mut(index),
            self.edge_cost.get_mut(index),
        ) else {
            return Err(PlanningError::WorkspaceTooSmall);
        };
        *stored_from = from as u32;
        *stored_to = into as u32;
        *stored_cost = cost;
        self.edge_length += 1;
        Ok(())
    }

    /// One edge, as `(from, to, cost)`.
    fn edge_at(&self, index: usize) -> Option<(usize, usize, T)> {
        Some((
            self.edge_from.get(index).copied()? as usize,
            self.edge_to.get(index).copied()? as usize,
            self.edge_cost.get(index).copied()?,
        ))
    }
}

impl<const MAX_NODES: usize, const MAX_EDGES: usize, const DIMENSION: usize, T: Numeric> Default
    for PrmWorkspace<MAX_NODES, MAX_EDGES, DIMENSION, T>
{
    fn default() -> Self {
        Self::new()
    }
}

/// A roadmap built once over an environment and queried many times.
///
/// Worth its build cost where the obstacles stay put and the queries keep coming; a tree planner is
/// the better answer for a single query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prm<const DIMENSION: usize, T: Numeric = f64> {
    connection_radius: T,
    sample_count: usize,
    edge_checks: usize,
}

impl<const DIMENSION: usize, T: Numeric> Prm<DIMENSION, T> {
    /// A roadmap of `200` samples joined within `0.5`, with eight edge checks.
    #[must_use]
    pub fn new() -> Self {
        Prm {
            connection_radius: T::from_f64(0.5),
            sample_count: 200,
            edge_checks: 8,
        }
    }

    /// How far apart two samples may be and still be joined.
    pub fn try_with_connection_radius(mut self, radius: T) -> Result<Self, PlanningError> {
        if !radius.is_finite() {
            return Err(PlanningError::NonFinite);
        }
        if radius <= T::ZERO {
            return Err(PlanningError::NonPositiveParameter);
        }
        self.connection_radius = radius;
        Ok(self)
    }

    /// How many valid samples to draw.
    #[must_use]
    pub fn with_sample_count(mut self, samples: usize) -> Self {
        self.sample_count = samples;
        self
    }

    /// How many interior stations an edge is tested at.
    #[must_use]
    pub fn with_edge_checks(mut self, checks: usize) -> Self {
        self.edge_checks = checks;
        self
    }

    /// Builds the roadmap, replacing whatever the workspace held.
    ///
    /// Returns [`PlanningError::WorkspaceTooSmall`] if the edge array fills. Sampling stops early
    /// and without error if the node array fills, so a small arena gives a sparser roadmap rather
    /// than a failure.
    pub fn try_build<const MAX_NODES: usize, const MAX_EDGES: usize, S, V, R>(
        &self,
        space: &S,
        validity: &V,
        source: &mut R,
        workspace: &mut PrmWorkspace<MAX_NODES, MAX_EDGES, DIMENSION, T>,
    ) -> Result<(), PlanningError>
    where
        S: StateSpace<DIMENSION, T>,
        V: StateValidity<DIMENSION, T>,
        R: RandomSource<T>,
        T: RandomScalar,
    {
        if MAX_NODES == 0 || MAX_NODES >= u32::MAX as usize {
            return Err(PlanningError::WorkspaceTooSmall);
        }
        workspace.clear();

        for _ in 0..self.sample_count {
            if workspace.node_length >= MAX_NODES {
                break;
            }
            let drawn = space.sample(source);
            if !validity.is_state_valid(&drawn) {
                continue;
            }
            workspace.push_node(drawn);
        }

        for first in 0..workspace.node_length {
            for second in (first + 1)..workspace.node_length {
                let (Some(first_state), Some(second_state)) =
                    (workspace.state_at(first), workspace.state_at(second))
                else {
                    continue;
                };
                let separation = space.distance(&first_state, &second_state);
                if separation > self.connection_radius {
                    continue;
                }
                if !edge_is_valid(
                    space,
                    validity,
                    &first_state,
                    &second_state,
                    self.edge_checks,
                ) {
                    continue;
                }
                workspace.push_edge(first, second, separation)?;
            }
        }
        Ok(())
    }

    /// Answers one start-to-goal query against a built roadmap.
    ///
    /// The start and goal are joined to their nearest roadmap nodes that a valid edge reaches, and
    /// Dijkstra runs over the flat edge list. That scan is O(V·E): at `MAX_NODES = 256` and
    /// `MAX_EDGES = 2048` it is 524 k comparisons, a fraction of the O(V²) validity checks the build
    /// already paid. Counting-sort to a CSR adjacency is the upgrade if profiling ever asks.
    ///
    /// Returns [`PlanningError::NoPathFound`] for an empty roadmap, for endpoints no edge reaches,
    /// and where the roadmap has no route between them.
    pub fn try_query<
        const MAX_NODES: usize,
        const MAX_EDGES: usize,
        const MAX_POINTS: usize,
        S,
        V,
    >(
        &self,
        space: &S,
        validity: &V,
        start: Vector<DIMENSION, T>,
        goal: Vector<DIMENSION, T>,
        workspace: &mut PrmWorkspace<MAX_NODES, MAX_EDGES, DIMENSION, T>,
    ) -> Result<PlanReport<MAX_POINTS, DIMENSION, T>, PlanningError>
    where
        S: StateSpace<DIMENSION, T>,
        V: StateValidity<DIMENSION, T>,
    {
        if !start.is_finite() || !goal.is_finite() {
            return Err(PlanningError::NonFinite);
        }
        if !space.contains(&start) {
            return Err(PlanningError::StartOutOfBounds);
        }
        if !space.contains(&goal) {
            return Err(PlanningError::GoalOutOfBounds);
        }
        if !validity.is_state_valid(&start) {
            return Err(PlanningError::StartNotFree);
        }
        if !validity.is_state_valid(&goal) {
            return Err(PlanningError::GoalNotFree);
        }
        if workspace.node_length == 0 {
            return Err(PlanningError::NoPathFound);
        }

        let entry = self
            .nearest_reachable(space, validity, workspace, &start)
            .ok_or(PlanningError::NoPathFound)?;
        let exit = self
            .nearest_reachable(space, validity, workspace, &goal)
            .ok_or(PlanningError::NoPathFound)?;

        // Dijkstra over the roadmap. The search arrays are separate from the roadmap itself, so a
        // query never disturbs what `try_build` produced.
        let nodes = workspace.node_length;
        for index in 0..nodes {
            if let Some(cost) = workspace.cost_so_far.get_mut(index) {
                *cost = T::INFINITY;
            }
            if let Some(parent) = workspace.parent.get_mut(index) {
                *parent = NO_PARENT;
            }
        }
        workspace.frontier.clear_prefix(nodes);
        if let Some(cost) = workspace.cost_so_far.get_mut(entry) {
            *cost = T::ZERO;
        }
        workspace
            .frontier
            .push_or_lower(entry, T::ZERO)
            .map_err(|_| PlanningError::WorkspaceTooSmall)?;

        let mut expansions = 0usize;
        let mut reached = false;
        while let Some((current, settled)) = workspace.frontier.pop_minimum() {
            expansions += 1;
            if current == exit {
                reached = true;
                break;
            }
            if settled
                > workspace
                    .cost_so_far
                    .get(current)
                    .copied()
                    .unwrap_or(T::INFINITY)
            {
                continue;
            }
            for edge in 0..workspace.edge_length {
                let Some((edge_start, edge_end, cost)) = workspace.edge_at(edge) else {
                    continue;
                };
                let next = if edge_start == current {
                    edge_end
                } else if edge_end == current {
                    edge_start
                } else {
                    continue;
                };
                let tentative = settled + cost;
                if tentative
                    < workspace
                        .cost_so_far
                        .get(next)
                        .copied()
                        .unwrap_or(T::INFINITY)
                {
                    if let Some(stored) = workspace.cost_so_far.get_mut(next) {
                        *stored = tentative;
                    }
                    if let Some(stored) = workspace.parent.get_mut(next) {
                        *stored = current as u32;
                    }
                    workspace
                        .frontier
                        .push_or_lower(next, tentative)
                        .map_err(|_| PlanningError::WorkspaceTooSmall)?;
                }
            }
        }
        if !reached {
            return Err(PlanningError::NoPathFound);
        }

        // The plan runs start, through the roadmap, to goal, so the two ends are added to the
        // roadmap chain rather than being roadmap nodes themselves.
        let through = extract_tree_path::<MAX_POINTS, DIMENSION, T>(
            workspace.occupied(),
            workspace.parent.get(..nodes).unwrap_or(&[]),
            exit,
        )?;
        let roadmap_cost = workspace
            .cost_so_far
            .get(exit)
            .copied()
            .unwrap_or(T::INFINITY);

        let mut path = crate::motion::PolylinePath::<MAX_POINTS, DIMENSION, T>::new();
        path.push(start)?;
        for waypoint in through.waypoints() {
            path.push(*waypoint)?;
        }
        path.push(goal)?;

        let entry_state = workspace
            .state_at(entry)
            .ok_or(PlanningError::NoPathFound)?;
        let exit_state = workspace.state_at(exit).ok_or(PlanningError::NoPathFound)?;
        let cost = space.distance(&start, &entry_state)
            + roadmap_cost
            + space.distance(&exit_state, &goal);
        Ok(PlanReport::new(path, cost, expansions))
    }

    /// The nearest roadmap node a valid edge from `state` reaches.
    ///
    /// An edge is only checked when the candidate is nearer than the best that already connects, so
    /// this costs at most one check a node.
    fn nearest_reachable<const MAX_NODES: usize, const MAX_EDGES: usize, S, V>(
        &self,
        space: &S,
        validity: &V,
        workspace: &PrmWorkspace<MAX_NODES, MAX_EDGES, DIMENSION, T>,
        state: &Vector<DIMENSION, T>,
    ) -> Option<usize>
    where
        S: StateSpace<DIMENSION, T>,
        V: StateValidity<DIMENSION, T>,
    {
        let mut best: Option<(usize, T)> = None;
        for index in 0..workspace.node_length {
            let Some(node) = workspace.state_at(index) else {
                continue;
            };
            let separation = space.distance(state, &node);
            if best.is_some_and(|(_, shortest)| separation >= shortest) {
                continue;
            }
            if edge_is_valid(space, validity, state, &node, self.edge_checks) {
                best = Some((index, separation));
            }
        }
        best.map(|(index, _)| index)
    }
}

impl<const DIMENSION: usize, T: Numeric> Default for Prm<DIMENSION, T> {
    fn default() -> Self {
        Self::new()
    }
}
