#![deny(clippy::indexing_slicing)]

//! A tree grown toward random samples, and its asymptotically optimal rewiring variant.
//!
//! Both are off-loop work: a plan is made when the goal or the world changes, not per tick.

use crate::error::PlanningError;
use crate::linear_algebra::Vector;
use crate::planning::plan_report::PlanReport;
use crate::planning::sampling::{
    NO_PARENT, edge_is_valid, extract_tree_path, nearest_index, neighbours_within, steer_towards,
};
use crate::planning::state_space::{StateSpace, StateValidity};
use crate::random::{RandomScalar, RandomSource};
use crate::scalar::Numeric;

/// The tree a sampling planner grows, owned by the caller.
///
/// `costs` ships even though plain [`Rrt`] never rewires, so [`RrtStar`] shares the same arena.
///
/// Memory is `DIMENSION·size_of::<T>() + 12` bytes a node plus padding.
#[derive(Debug, Clone, Copy)]
pub struct RrtWorkspace<const MAX_NODES: usize, const DIMENSION: usize, T: Numeric = f64> {
    states: [Vector<DIMENSION, T>; MAX_NODES],
    parents: [u32; MAX_NODES],
    costs: [T; MAX_NODES],
    length: usize,
}

impl<const MAX_NODES: usize, const DIMENSION: usize, T: Numeric>
    RrtWorkspace<MAX_NODES, DIMENSION, T>
{
    /// An empty tree.
    #[must_use]
    pub fn new() -> Self {
        RrtWorkspace {
            states: [Vector::zeros(); MAX_NODES],
            parents: [NO_PARENT; MAX_NODES],
            costs: [T::ZERO; MAX_NODES],
            length: 0,
        }
    }

    /// Empties the tree.
    pub fn clear(&mut self) {
        self.length = 0;
    }

    /// How many nodes the tree can hold.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        MAX_NODES
    }

    /// How many nodes it holds.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.length
    }

    /// One node's state.
    #[inline]
    pub fn node_state(&self, index: usize) -> Option<Vector<DIMENSION, T>> {
        (index < self.length).then(|| self.states.get(index).copied())?
    }

    /// One node's parent, or `None` for the root or an index past the tree.
    #[inline]
    #[must_use]
    pub fn node_parent(&self, index: usize) -> Option<usize> {
        if index >= self.length {
            return None;
        }
        match self.parents.get(index).copied() {
            Some(NO_PARENT) | None => None,
            Some(parent) => Some(parent as usize),
        }
    }

    /// One node's cost from the root.
    #[inline]
    #[must_use]
    pub fn node_cost(&self, index: usize) -> Option<T> {
        (index < self.length).then(|| self.costs.get(index).copied())?
    }

    /// Appends a node, returning where it landed.
    fn push(&mut self, state: Vector<DIMENSION, T>, parent: u32, cost: T) -> Option<usize> {
        if self.length >= MAX_NODES {
            return None;
        }
        let index = self.length;
        *self.states.get_mut(index)? = state;
        *self.parents.get_mut(index)? = parent;
        *self.costs.get_mut(index)? = cost;
        self.length += 1;
        Some(index)
    }

    /// The states in use, for the shared helpers to scan.
    fn occupied(&self) -> &[Vector<DIMENSION, T>] {
        self.states.get(..self.length).unwrap_or(&[])
    }

    fn cost_at(&self, index: usize) -> T {
        self.costs.get(index).copied().unwrap_or(T::INFINITY)
    }

    fn state_at(&self, index: usize) -> Option<Vector<DIMENSION, T>> {
        self.states.get(index).copied()
    }
}

impl<const MAX_NODES: usize, const DIMENSION: usize, T: Numeric> Default
    for RrtWorkspace<MAX_NODES, DIMENSION, T>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Settings both sampling planners share.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Growth<T: Numeric> {
    step_size: T,
    goal_bias: T,
    goal_tolerance: T,
    edge_checks: usize,
    iteration_budget: usize,
}

impl<T: Numeric> Growth<T> {
    fn new() -> Self {
        Growth {
            step_size: T::from_f64(0.1),
            goal_bias: T::from_f64(0.05),
            goal_tolerance: T::from_f64(0.1),
            edge_checks: 8,
            iteration_budget: 10_000,
        }
    }

    fn with_step_size(mut self, step: T) -> Result<Self, PlanningError> {
        self.step_size = positive(step)?;
        Ok(self)
    }

    fn with_goal_tolerance(mut self, tolerance: T) -> Result<Self, PlanningError> {
        self.goal_tolerance = positive(tolerance)?;
        Ok(self)
    }

    fn with_goal_bias(mut self, probability: T) -> Result<Self, PlanningError> {
        if !probability.is_finite() {
            return Err(PlanningError::NonFinite);
        }
        if probability < T::ZERO || probability > T::ONE {
            return Err(PlanningError::InvalidGoalBias);
        }
        self.goal_bias = probability;
        Ok(self)
    }
}

/// A tuning value that must be a real number strictly above zero.
fn positive<T: Numeric>(value: T) -> Result<T, PlanningError> {
    if !value.is_finite() {
        return Err(PlanningError::NonFinite);
    }
    if value <= T::ZERO {
        return Err(PlanningError::NonPositiveParameter);
    }
    Ok(value)
}

/// Checks a plan's endpoints against the space and the obstacles.
fn check_endpoints<const DIMENSION: usize, T: Numeric, S, V>(
    space: &S,
    validity: &V,
    start: &Vector<DIMENSION, T>,
    goal: &Vector<DIMENSION, T>,
    capacity: usize,
) -> Result<(), PlanningError>
where
    S: StateSpace<DIMENSION, T>,
    V: StateValidity<DIMENSION, T>,
{
    if capacity == 0 || capacity >= u32::MAX as usize {
        return Err(PlanningError::WorkspaceTooSmall);
    }
    if !start.is_finite() || !goal.is_finite() {
        return Err(PlanningError::NonFinite);
    }
    if !space.contains(start) {
        return Err(PlanningError::StartOutOfBounds);
    }
    if !space.contains(goal) {
        return Err(PlanningError::GoalOutOfBounds);
    }
    if !validity.is_state_valid(start) {
        return Err(PlanningError::StartNotFree);
    }
    if !validity.is_state_valid(goal) {
        return Err(PlanningError::GoalNotFree);
    }
    Ok(())
}

/// A rapidly-exploring random tree: the **first** path it finds, not the best.
///
/// Contrast [`RrtStar`], which keeps sampling to its budget and returns the best. They are separate
/// types because that promise is the difference, and a caller has to know which one they hold.
///
/// ```
/// use multicalc::planning::{BoxSpace, Rrt, RrtWorkspace};
/// use multicalc::{Pcg32, Vector};
///
/// // A 4 m square with a wall across the middle, open at one end.
/// let space: BoxSpace<2> = BoxSpace::try_new(Vector::new([0.0, 0.0]), Vector::new([4.0, 4.0]))?;
/// let is_state_valid = |state: &Vector<2, f64>| {
///     let blocked_band = (state[1] - 2.0).abs() < 0.2;
///     !(blocked_band && state[0] < 3.0)
/// };
///
/// let mut workspace: RrtWorkspace<2000, 2> = RrtWorkspace::new();
/// let mut source = Pcg32::new(20260830);
/// let report = Rrt::new()
///     .try_with_step_size(0.3)?
///     .try_plan::<2000, 256, _, _, _>(
///         &space,
///         &is_state_valid,
///         Vector::new([0.5, 0.5]),
///         Vector::new([0.5, 3.5]),
///         &mut source,
///         &mut workspace,
///     )?;
///
/// assert!(report.waypoint_count() >= 2);
/// assert!(report.cost() > 0.0);
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rrt<const DIMENSION: usize, T: Numeric = f64> {
    growth: Growth<T>,
}

impl<const DIMENSION: usize, T: Numeric> Rrt<DIMENSION, T> {
    /// A tree with step `0.1`, goal bias `0.05`, goal tolerance `0.1`, eight edge checks and a
    /// budget of `10_000` samples.
    #[must_use]
    pub fn new() -> Self {
        Rrt {
            growth: Growth::new(),
        }
    }

    /// How far the tree reaches toward a sample in one step.
    pub fn try_with_step_size(mut self, step: T) -> Result<Self, PlanningError> {
        self.growth = self.growth.with_step_size(step)?;
        Ok(self)
    }

    /// How often to sample the goal itself instead of the space, in zero to one.
    pub fn try_with_goal_bias(mut self, probability: T) -> Result<Self, PlanningError> {
        self.growth = self.growth.with_goal_bias(probability)?;
        Ok(self)
    }

    /// How near the goal counts as reaching it.
    pub fn try_with_goal_tolerance(mut self, tolerance: T) -> Result<Self, PlanningError> {
        self.growth = self.growth.with_goal_tolerance(tolerance)?;
        Ok(self)
    }

    /// How many interior stations an edge is tested at.
    #[must_use]
    pub fn with_edge_checks(mut self, checks: usize) -> Self {
        self.growth.edge_checks = checks;
        self
    }

    /// How many samples the search may draw.
    #[must_use]
    pub fn with_iteration_budget(mut self, samples: usize) -> Self {
        self.growth.iteration_budget = samples;
        self
    }

    /// Grows a tree from `start` until it reaches `goal`.
    ///
    /// Returns [`PlanningError::DidNotConverge`] when the sample budget ran out first, and
    /// [`PlanningError::WorkspaceTooSmall`] if the arena filled before the goal was reached.
    pub fn try_plan<const MAX_NODES: usize, const MAX_POINTS: usize, S, V, R>(
        &self,
        space: &S,
        validity: &V,
        start: Vector<DIMENSION, T>,
        goal: Vector<DIMENSION, T>,
        source: &mut R,
        workspace: &mut RrtWorkspace<MAX_NODES, DIMENSION, T>,
    ) -> Result<PlanReport<MAX_POINTS, DIMENSION, T>, PlanningError>
    where
        S: StateSpace<DIMENSION, T>,
        V: StateValidity<DIMENSION, T>,
        R: RandomSource<T>,
        T: RandomScalar,
    {
        check_endpoints(space, validity, &start, &goal, MAX_NODES)?;
        workspace.clear();
        workspace
            .push(start, NO_PARENT, T::ZERO)
            .ok_or(PlanningError::WorkspaceTooSmall)?;

        for sample in 1..=self.growth.iteration_budget {
            // One draw is spent either way, so a fixed seed reproduces the tree exactly.
            let toward_the_goal = source.next_unit() < self.growth.goal_bias;
            let target = if toward_the_goal {
                goal
            } else {
                space.sample(source)
            };

            let Some(nearest) = nearest_index(space, workspace.occupied(), &target) else {
                continue;
            };
            let Some(from) = workspace.state_at(nearest) else {
                continue;
            };
            let candidate = steer_towards(space, &from, &target, self.growth.step_size);

            if !validity.is_state_valid(&candidate)
                || !edge_is_valid(space, validity, &from, &candidate, self.growth.edge_checks)
            {
                continue;
            }

            let cost = workspace.cost_at(nearest) + space.distance(&from, &candidate);
            let Some(added) = workspace.push(candidate, nearest as u32, cost) else {
                // The arena filled without reaching the goal, so there is nothing to return.
                return Err(PlanningError::WorkspaceTooSmall);
            };

            if space.distance(&candidate, &goal) <= self.growth.goal_tolerance {
                let path = extract_tree_path::<MAX_POINTS, DIMENSION, T>(
                    workspace.occupied(),
                    workspace.parents.get(..workspace.length).unwrap_or(&[]),
                    added,
                )?;
                return Ok(PlanReport::new(path, workspace.cost_at(added), sample));
            }
        }

        Err(PlanningError::DidNotConverge {
            iterations: self.growth.iteration_budget,
        })
    }
}

impl<const DIMENSION: usize, T: Numeric> Default for Rrt<DIMENSION, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// RRT\*: the **best** path found within the sample budget, not the first.
///
/// Each new node picks the cheapest parent among its neighbours, and rewires any neighbour it can
/// reach more cheaply. Cost falls as the budget grows, which is why it does not stop on first
/// contact with the goal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrtStar<const DIMENSION: usize, T: Numeric = f64> {
    growth: Growth<T>,
    neighbour_radius: T,
}

impl<const DIMENSION: usize, T: Numeric> RrtStar<DIMENSION, T> {
    /// The same defaults as [`Rrt`], with a neighbour radius of `0.5`.
    #[must_use]
    pub fn new() -> Self {
        RrtStar {
            growth: Growth::new(),
            neighbour_radius: T::from_f64(0.5),
        }
    }

    /// How far the tree reaches toward a sample in one step.
    pub fn try_with_step_size(mut self, step: T) -> Result<Self, PlanningError> {
        self.growth = self.growth.with_step_size(step)?;
        Ok(self)
    }

    /// How often to sample the goal itself instead of the space, in zero to one.
    pub fn try_with_goal_bias(mut self, probability: T) -> Result<Self, PlanningError> {
        self.growth = self.growth.with_goal_bias(probability)?;
        Ok(self)
    }

    /// How near the goal counts as reaching it.
    pub fn try_with_goal_tolerance(mut self, tolerance: T) -> Result<Self, PlanningError> {
        self.growth = self.growth.with_goal_tolerance(tolerance)?;
        Ok(self)
    }

    /// How far around a new node to look for a cheaper parent and for nodes to rewire.
    pub fn try_with_neighbour_radius(mut self, radius: T) -> Result<Self, PlanningError> {
        self.neighbour_radius = positive(radius)?;
        Ok(self)
    }

    /// How many interior stations an edge is tested at.
    #[must_use]
    pub fn with_edge_checks(mut self, checks: usize) -> Self {
        self.growth.edge_checks = checks;
        self
    }

    /// How many samples the search may draw. For RRT\* the budget is the plan, not a cut-off.
    #[must_use]
    pub fn with_iteration_budget(mut self, samples: usize) -> Self {
        self.growth.iteration_budget = samples;
        self
    }

    /// Grows and rewires a tree from `start`, returning the best path to `goal` it found.
    ///
    /// Returns [`PlanningError::NoPathFound`] if the budget ended with no node inside the goal
    /// tolerance — for RRT\* an exhausted budget is the plan, so it is not a `DidNotConverge`.
    pub fn try_plan<const MAX_NODES: usize, const MAX_POINTS: usize, S, V, R>(
        &self,
        space: &S,
        validity: &V,
        start: Vector<DIMENSION, T>,
        goal: Vector<DIMENSION, T>,
        source: &mut R,
        workspace: &mut RrtWorkspace<MAX_NODES, DIMENSION, T>,
    ) -> Result<PlanReport<MAX_POINTS, DIMENSION, T>, PlanningError>
    where
        S: StateSpace<DIMENSION, T>,
        V: StateValidity<DIMENSION, T>,
        R: RandomSource<T>,
        T: RandomScalar,
    {
        check_endpoints(space, validity, &start, &goal, MAX_NODES)?;
        workspace.clear();
        workspace
            .push(start, NO_PARENT, T::ZERO)
            .ok_or(PlanningError::WorkspaceTooSmall)?;

        let mut best_leaf: Option<usize> = None;
        let mut samples = 0usize;

        for sample in 1..=self.growth.iteration_budget {
            samples = sample;
            let toward_the_goal = source.next_unit() < self.growth.goal_bias;
            let target = if toward_the_goal {
                goal
            } else {
                space.sample(source)
            };

            let Some(nearest) = nearest_index(space, workspace.occupied(), &target) else {
                continue;
            };
            let Some(from) = workspace.state_at(nearest) else {
                continue;
            };
            let candidate = steer_towards(space, &from, &target, self.growth.step_size);
            if !validity.is_state_valid(&candidate) {
                continue;
            }

            // Choose the cheapest reachable parent within the radius, falling back to the nearest.
            let mut chosen =
                if edge_is_valid(space, validity, &from, &candidate, self.growth.edge_checks) {
                    Some((
                        nearest,
                        workspace.cost_at(nearest) + space.distance(&from, &candidate),
                    ))
                } else {
                    None
                };
            let mut contenders = [(0usize, T::ZERO); 0];
            let _ = &mut contenders;
            neighbours_within(
                space,
                workspace.occupied(),
                &candidate,
                self.neighbour_radius,
                |index, separation| {
                    let through = workspace.cost_at(index) + separation;
                    let better = match chosen {
                        None => true,
                        Some((_, best)) => through < best,
                    };
                    if better
                        && let Some(neighbour) = workspace.state_at(index)
                        && edge_is_valid(
                            space,
                            validity,
                            &neighbour,
                            &candidate,
                            self.growth.edge_checks,
                        )
                    {
                        chosen = Some((index, through));
                    }
                },
            );
            let Some((parent, cost)) = chosen else {
                continue;
            };

            let Some(added) = workspace.push(candidate, parent as u32, cost) else {
                break;
            };

            self.rewire(space, validity, workspace, added);

            if space.distance(&candidate, &goal) <= self.growth.goal_tolerance {
                let better = match best_leaf {
                    None => true,
                    Some(current) => workspace.cost_at(added) < workspace.cost_at(current),
                };
                if better {
                    best_leaf = Some(added);
                }
            }
        }

        let leaf = best_leaf.ok_or(PlanningError::NoPathFound)?;
        let path = extract_tree_path::<MAX_POINTS, DIMENSION, T>(
            workspace.occupied(),
            workspace.parents.get(..workspace.length).unwrap_or(&[]),
            leaf,
        )?;
        Ok(PlanReport::new(path, workspace.cost_at(leaf), samples))
    }

    /// Hangs any neighbour of `added` that it can reach more cheaply off it, and carries the saving
    /// down to that neighbour's descendants.
    ///
    /// Leaving descendant costs stale is what would make a larger budget return a *worse* path, so
    /// the sweep is not optional. It is bounded rather than recursive, which is what keeps the
    /// no-panic guarantee.
    fn rewire<const MAX_NODES: usize, S, V>(
        &self,
        space: &S,
        validity: &V,
        workspace: &mut RrtWorkspace<MAX_NODES, DIMENSION, T>,
        added: usize,
    ) where
        S: StateSpace<DIMENSION, T>,
        V: StateValidity<DIMENSION, T>,
    {
        let Some(candidate) = workspace.state_at(added) else {
            return;
        };
        let added_cost = workspace.cost_at(added);

        let mut rewired_any = false;
        for index in 0..workspace.length {
            if index == added {
                continue;
            }
            let Some(neighbour) = workspace.state_at(index) else {
                continue;
            };
            let separation = space.distance(&candidate, &neighbour);
            if separation > self.neighbour_radius {
                continue;
            }
            let through = added_cost + separation;
            // `!(a < b)` rather than `a >= b`: the two differ on a non-finite cost, and asking for a
            // total order `T` does not have would mean a `partial_cmp().unwrap()`.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(through < workspace.cost_at(index)) {
                continue;
            }
            if !edge_is_valid(
                space,
                validity,
                &candidate,
                &neighbour,
                self.growth.edge_checks,
            ) {
                continue;
            }
            if let Some(parent) = workspace.parents.get_mut(index) {
                *parent = added as u32;
            }
            if let Some(cost) = workspace.costs.get_mut(index) {
                *cost = through;
            }
            rewired_any = true;
        }

        if rewired_any {
            self.settle_descendant_costs(space, workspace);
        }
    }

    /// Recomputes every node's cost from its parent's, sweeping until nothing moves.
    ///
    /// Nodes are appended after their parents, but rewiring breaks that order, so one forward sweep
    /// is not enough. The sweep count is bounded by the node count, which bounds the longest chain.
    fn settle_descendant_costs<const MAX_NODES: usize, S>(
        &self,
        space: &S,
        workspace: &mut RrtWorkspace<MAX_NODES, DIMENSION, T>,
    ) where
        S: StateSpace<DIMENSION, T>,
    {
        for _ in 0..workspace.length {
            let mut settled = true;
            for index in 0..workspace.length {
                let Some(parent) = workspace.node_parent(index) else {
                    continue;
                };
                let (Some(state), Some(parent_state)) =
                    (workspace.state_at(index), workspace.state_at(parent))
                else {
                    continue;
                };
                let expected = workspace.cost_at(parent) + space.distance(&parent_state, &state);
                if (expected - workspace.cost_at(index)).abs() > T::EPSILON_X4 {
                    if let Some(cost) = workspace.costs.get_mut(index) {
                        *cost = expected;
                    }
                    settled = false;
                }
            }
            if settled {
                return;
            }
        }
    }
}

impl<const DIMENSION: usize, T: Numeric> Default for RrtStar<DIMENSION, T> {
    fn default() -> Self {
        Self::new()
    }
}
