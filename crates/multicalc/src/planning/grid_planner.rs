#![deny(clippy::indexing_slicing)]

//! Best-first search over an occupancy map: Dijkstra, A\*, weighted A\*, and any-angle Theta\*.
//!
//! Planning is off-loop work. The search runs when the goal or the map changes; the control loop
//! consumes the [`PolylinePath`] that comes back.

use crate::error::PlanningError;
use crate::linear_algebra::Vector;
use crate::mapping::{GridGeometry, OccupancyMap};
use crate::motion::PolylinePath;
use crate::planning::grid_workspace::{CellVisit, GridSearchWorkspace, NO_PARENT};
use crate::planning::plan_report::PlanReport;
use crate::planning::traversal_cost::TraversalCost;
use crate::scalar::{Numeric, Primal};

/// The four orthogonal neighbours of a cell, as `(row, column)` offsets.
const ORTHOGONAL: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// The four diagonal neighbours of a cell.
const DIAGONAL: [(isize, isize); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

/// How far from a straight line three waypoints may sit and still be pruned to two.
const COLLINEAR_TOLERANCE: f64 = 1e-9;

/// Which search to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GridSearch {
    /// No heuristic: every reachable cell is settled in cost order.
    Dijkstra,
    /// Best-first with an admissible heuristic. Optimal at weight one.
    AStar,
    /// Any-angle: a relaxation that keeps line of sight to the grandparent skips the grid.
    ThetaStar,
}

/// Which neighbours a cell has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GridConnectivity {
    /// The four cells sharing an edge.
    FourConnected,
    /// Those four plus the four sharing only a corner.
    EightConnected,
}

/// The estimate of remaining cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GridHeuristic {
    /// `|Δrow| + |Δcolumn|`. Admissible only four-connected.
    Manhattan,
    /// `max + (√2 − 1)·min` of the two deltas: the exact cost of an eight-connected straight run.
    Octile,
    /// The hypotenuse. Admissible everywhere, and looser than Octile eight-connected.
    Euclidean,
}

/// A configured grid search.
///
/// ```
/// use multicalc::mapping::OccupancyMap;
/// use multicalc::planning::{GridPlanner, GridSearchWorkspace, UniformCost};
///
/// // An 8 by 8 room of unit cells with a wall up column 4, open at the top row.
/// struct Room {
///     cells: [[bool; 8]; 8],
/// }
///
/// impl OccupancyMap for Room {
///     fn columns(&self) -> usize { 8 }
///     fn rows(&self) -> usize { 8 }
///     fn resolution(&self) -> f64 { 1.0 }
///     fn origin(&self) -> [f64; 2] { [0.0, 0.0] }
///     fn is_occupied(&self, row: usize, column: usize) -> bool {
///         self.cells.get(row).and_then(|row| row.get(column)).copied().unwrap_or(false)
///     }
/// }
///
/// let cells = core::array::from_fn(|row| core::array::from_fn(|column| column == 4 && row < 7));
/// let room = Room { cells };
///
/// const MAX_CELLS: usize = 8 * 8;
/// let mut workspace: GridSearchWorkspace<MAX_CELLS> = GridSearchWorkspace::new();
/// let cost = UniformCost::new(&room);
///
/// let start = [0.5, 0.5];
/// let goal = [7.5, 0.5];
/// let report = GridPlanner::new()
///     .try_plan::<MAX_CELLS, 64, _, _>(&room, &cost, start, goal, &mut workspace)?;
///
/// // The path starts and ends where it was asked to, and had to go round the wall.
/// let path = report.path();
/// assert_eq!(path.waypoints().first().map(|point| *point.as_array()), Some(start));
/// assert_eq!(path.waypoints().last().map(|point| *point.as_array()), Some(goal));
/// assert!(report.cost() > 7.0);
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPlanner<T: Numeric + Primal = f64> {
    search: GridSearch,
    connectivity: GridConnectivity,
    heuristic: GridHeuristic,
    heuristic_weight: T,
    corner_cutting: bool,
    expansion_budget: usize,
}

impl<T: Numeric + Primal> GridPlanner<T> {
    /// A planner running A\* eight-connected with the Octile heuristic at weight one, no corner
    /// cutting, and a budget of one expansion per cell.
    #[must_use]
    pub fn new() -> Self {
        GridPlanner {
            search: GridSearch::AStar,
            connectivity: GridConnectivity::EightConnected,
            heuristic: GridHeuristic::Octile,
            heuristic_weight: T::ONE,
            corner_cutting: false,
            expansion_budget: 0,
        }
    }

    /// Which search to run.
    #[must_use]
    pub fn with_search(mut self, search: GridSearch) -> Self {
        self.search = search;
        self
    }

    /// Which neighbours a cell has.
    #[must_use]
    pub fn with_connectivity(mut self, connectivity: GridConnectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// The estimate of remaining cost.
    #[must_use]
    pub fn with_heuristic(mut self, heuristic: GridHeuristic) -> Self {
        self.heuristic = heuristic;
        self
    }

    /// Whether a diagonal step may pass between two blocked cells meeting at a corner.
    #[must_use]
    pub fn with_corner_cutting(mut self, allowed: bool) -> Self {
        self.corner_cutting = allowed;
        self
    }

    /// How many expansions the search may spend. Zero means one per cell.
    #[must_use]
    pub fn with_expansion_budget(mut self, expansions: usize) -> Self {
        self.expansion_budget = expansions;
        self
    }

    /// Inflates the heuristic, trading optimality for speed: the cost found is within `weight` of
    /// optimal.
    ///
    /// Returns [`PlanningError::NonFinite`] for a non-finite weight and
    /// [`PlanningError::HeuristicWeightBelowOne`] for one below one.
    pub fn try_with_heuristic_weight(mut self, weight: T) -> Result<Self, PlanningError> {
        if !weight.is_finite() {
            return Err(PlanningError::NonFinite);
        }
        if weight < T::ONE {
            return Err(PlanningError::HeuristicWeightBelowOne);
        }
        self.heuristic_weight = weight;
        Ok(self)
    }

    /// Plans from `start` to `goal` across `map`, charging `cost` for each cell entered.
    ///
    /// `MAX_CELLS` must be at least the map's `rows · columns`, and `MAX_POINTS` at least the
    /// waypoints the plan needs — [`PlanningError::PathCapacityExceeded`] reports how many that
    /// would be, so a caller can resize and retry in one round trip.
    ///
    /// Returns [`PlanningError::NoPathFound`] when everything reachable has been searched without
    /// meeting the goal, which for Dijkstra and A\* is a proof that no path exists, and
    /// [`PlanningError::DidNotConverge`] when the expansion budget ran out first.
    pub fn try_plan<const MAX_CELLS: usize, const MAX_POINTS: usize, M, C>(
        &self,
        map: &M,
        cost: &C,
        start: [T; 2],
        goal: [T; 2],
        workspace: &mut GridSearchWorkspace<MAX_CELLS, T>,
    ) -> Result<PlanReport<MAX_POINTS, 2, T>, PlanningError>
    where
        M: OccupancyMap<T>,
        C: TraversalCost<T>,
    {
        let resolution = map.resolution();
        if !start[0].is_finite()
            || !start[1].is_finite()
            || !goal[0].is_finite()
            || !goal[1].is_finite()
            || !resolution.is_finite()
        {
            return Err(PlanningError::NonFinite);
        }
        if self.heuristic == GridHeuristic::Manhattan
            && self.connectivity == GridConnectivity::EightConnected
        {
            return Err(PlanningError::InadmissibleHeuristic);
        }

        let cells = map
            .rows()
            .checked_mul(map.columns())
            .ok_or(PlanningError::MapTooLarge)?;
        if cells == 0 || cells >= u32::MAX as usize {
            return Err(PlanningError::MapTooLarge);
        }
        if cells > MAX_CELLS {
            return Err(PlanningError::WorkspaceTooSmall);
        }

        let geometry = map.geometry();
        let (start_row, start_column) = geometry
            .cell_of(start)
            .ok_or(PlanningError::StartOutOfBounds)?;
        let (goal_row, goal_column) = geometry
            .cell_of(goal)
            .ok_or(PlanningError::GoalOutOfBounds)?;
        if cost.cost_of(start_row, start_column).is_none() {
            return Err(PlanningError::StartNotFree);
        }
        if cost.cost_of(goal_row, goal_column).is_none() {
            return Err(PlanningError::GoalNotFree);
        }

        let start_index = geometry
            .index_of(start_row, start_column)
            .ok_or(PlanningError::StartOutOfBounds)?;
        let goal_index = geometry
            .index_of(goal_row, goal_column)
            .ok_or(PlanningError::GoalOutOfBounds)?;

        workspace.reset(cells);
        set(&mut workspace.cost_so_far, start_index, T::ZERO);
        workspace
            .frontier
            .push_or_lower(
                start_index,
                self.heuristic_to(&geometry, start_row, start_column, goal_row, goal_column),
            )
            .map_err(|_| PlanningError::WorkspaceTooSmall)?;

        let budget = if self.expansion_budget == 0 {
            cells
        } else {
            self.expansion_budget
        };
        let diagonal_step = resolution * T::from_f64(core::f64::consts::SQRT_2);
        let mut expansions = 0usize;

        while let Some((current, _)) = workspace.frontier.pop_minimum() {
            if current == goal_index {
                let settled = read(&workspace.cost_so_far, current);
                return self.assemble(workspace, &geometry, goal_index, settled, expansions);
            }
            if read_visit(&workspace.visit, current) == CellVisit::Closed {
                continue;
            }
            set(&mut workspace.visit, current, CellVisit::Closed);
            expansions += 1;
            if expansions > budget {
                return Err(PlanningError::DidNotConverge {
                    iterations: expansions,
                });
            }

            let Some((row, column)) = geometry.cell_at(current) else {
                continue;
            };
            let settled = read(&workspace.cost_so_far, current);

            let diagonal_allowed = self.connectivity == GridConnectivity::EightConnected;
            let neighbours = ORTHOGONAL.iter().map(|offset| (*offset, false)).chain(
                DIAGONAL
                    .iter()
                    .map(|offset| (*offset, true))
                    .take(if diagonal_allowed { 4 } else { 0 }),
            );

            for ((row_offset, column_offset), is_diagonal) in neighbours {
                let Some((next_row, next_column)) =
                    step_from(&geometry, row, column, row_offset, column_offset)
                else {
                    continue;
                };
                let Some(multiplier) = cost.cost_of(next_row, next_column) else {
                    continue;
                };
                // A diagonal squeezing between two blocked cells meeting at a corner is a step no
                // robot with width can take.
                if is_diagonal && !self.corner_cutting {
                    let across = step_from(&geometry, row, column, 0, column_offset)
                        .and_then(|(side_row, side_column)| cost.cost_of(side_row, side_column));
                    let down = step_from(&geometry, row, column, row_offset, 0)
                        .and_then(|(side_row, side_column)| cost.cost_of(side_row, side_column));
                    if across.is_none() || down.is_none() {
                        continue;
                    }
                }
                let Some(next_index) = geometry.index_of(next_row, next_column) else {
                    continue;
                };
                // A settled cell is never reopened. Under a consistent heuristic it never needs to
                // be; under an inflated one this is weighted A* without re-expansion, which keeps
                // the `weight · optimal` bound and holds the search to one expansion a cell.
                if read_visit(&workspace.visit, next_index) == CellVisit::Closed {
                    continue;
                }

                let base = if is_diagonal {
                    diagonal_step
                } else {
                    resolution
                };
                let mut tentative = settled + base * multiplier;
                let mut parent_of_next = current as u32;

                // Theta*: if the grandparent can see the neighbour directly, hang it there instead
                // and take the straight-line cost, which is what makes the path any-angle.
                if self.search == GridSearch::ThetaStar {
                    let grandparent = read_parent(&workspace.parent, current);
                    if grandparent != NO_PARENT
                        && let Some(from) = centre_of_index(&geometry, grandparent as usize)
                        && let Some(into) = geometry.center_of(next_row, next_column)
                        && line_of_sight(&geometry, cost, from, into)
                    {
                        let direct = read(&workspace.cost_so_far, grandparent as usize)
                            + (into[0] - from[0]).hypot(into[1] - from[1]) * multiplier;
                        if direct < tentative {
                            tentative = direct;
                            parent_of_next = grandparent;
                        }
                    }
                }

                if tentative < read(&workspace.cost_so_far, next_index) {
                    set(&mut workspace.cost_so_far, next_index, tentative);
                    set(&mut workspace.parent, next_index, parent_of_next);
                    set(&mut workspace.visit, next_index, CellVisit::Open);
                    let estimate =
                        self.heuristic_to(&geometry, next_row, next_column, goal_row, goal_column);
                    workspace
                        .frontier
                        .push_or_lower(next_index, tentative + estimate)
                        .map_err(|_| PlanningError::WorkspaceTooSmall)?;
                }
            }
        }

        Err(PlanningError::NoPathFound)
    }

    /// The weighted estimate of what is left to travel, in the same units as the edge costs.
    fn heuristic_to(
        &self,
        geometry: &GridGeometry<T>,
        row: usize,
        column: usize,
        goal_row: usize,
        goal_column: usize,
    ) -> T {
        if self.search == GridSearch::Dijkstra {
            return T::ZERO;
        }
        let down = T::from_usize(row.abs_diff(goal_row));
        let across = T::from_usize(column.abs_diff(goal_column));
        let in_cells = match self.heuristic {
            GridHeuristic::Manhattan => down + across,
            GridHeuristic::Octile => {
                let (larger, smaller) = if down > across {
                    (down, across)
                } else {
                    (across, down)
                };
                larger + (T::from_f64(core::f64::consts::SQRT_2) - T::ONE) * smaller
            }
            GridHeuristic::Euclidean => down.hypot(across),
        };
        in_cells * geometry.resolution() * self.heuristic_weight
    }

    /// Turns the settled parent chain into a waypoint path.
    fn assemble<const MAX_CELLS: usize, const MAX_POINTS: usize>(
        &self,
        workspace: &GridSearchWorkspace<MAX_CELLS, T>,
        geometry: &GridGeometry<T>,
        goal_index: usize,
        cost: T,
        expansions: usize,
    ) -> Result<PlanReport<MAX_POINTS, 2, T>, PlanningError> {
        // Count what survives pruning before writing anything, so an overlong plan reports the
        // size it needs rather than failing part-written.
        let mut needed = 0usize;
        walk_pruned_chain(workspace, geometry, goal_index, |_| needed += 1);
        if needed > MAX_POINTS {
            return Err(PlanningError::PathCapacityExceeded { needed });
        }

        // The chain runs goal to start, so it is staged and then pushed in reverse.
        let mut staged = [Vector::<2, T>::zeros(); MAX_POINTS];
        let mut filled = 0usize;
        walk_pruned_chain(workspace, geometry, goal_index, |point| {
            if let Some(slot) = staged.get_mut(filled) {
                *slot = Vector::new(point);
                filled += 1;
            }
        });

        let mut path = PolylinePath::<MAX_POINTS, 2, T>::new();
        for index in (0..filled).rev() {
            let Some(point) = staged.get(index).copied() else {
                continue;
            };
            path.push(point)?;
        }
        Ok(PlanReport::new(path, cost, expansions))
    }
}

impl<T: Numeric + Primal> Default for GridPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Hands each waypoint of the plan to `visit`, in goal-to-start order.
///
/// A cell collinear with its neighbours in the chain is dropped: a straight run of A\* steps says
/// the same thing as its two endpoints, and pruning it keeps the output well inside `MAX_POINTS`.
fn walk_pruned_chain<const MAX_CELLS: usize, T: Numeric + Primal, F: FnMut([T; 2])>(
    workspace: &GridSearchWorkspace<MAX_CELLS, T>,
    geometry: &GridGeometry<T>,
    goal_index: usize,
    mut visit: F,
) {
    let tolerance = T::from_f64(COLLINEAR_TOLERANCE);
    let mut behind: Option<[T; 2]> = None;
    let mut current: Option<[T; 2]> = None;

    let mut index = goal_index;
    // A malformed chain cannot outlast the map.
    for _ in 0..=geometry.cell_count() {
        let Some(point) = centre_of_index(geometry, index) else {
            break;
        };
        match current {
            // The goal is an endpoint, so it is always kept.
            None => visit(point),
            Some(held) => {
                if let Some(back) = behind
                    && !is_collinear(back, held, point, tolerance)
                {
                    visit(held);
                }
                behind = Some(held);
            }
        }
        current = Some(point);

        let parent = read_parent(&workspace.parent, index);
        if parent == NO_PARENT {
            break;
        }
        index = parent as usize;
    }

    // The start is the other endpoint, unless it is also the goal and was already handed over.
    if let Some(held) = current
        && behind.is_some()
    {
        visit(held);
    }
}

/// Whether the middle point sits on the line joining the outer two, by the cross product.
fn is_collinear<T: Numeric>(before: [T; 2], middle: [T; 2], after: [T; 2], tolerance: T) -> bool {
    let first = [middle[0] - before[0], middle[1] - before[1]];
    let second = [after[0] - middle[0], after[1] - middle[1]];
    (first[0] * second[1] - first[1] * second[0]).abs() <= tolerance
}

/// Whether every cell between two world points can be entered.
///
/// Built on the ray walk. Where a ray meets a cell corner exactly, the walk's tie-break crosses the
/// row boundary first, so it enters that neighbour and reports it blocked if it is: a diagonal
/// through two blocked cells meeting at a corner does **not** read through. That is the
/// conservative answer, and it is what keeps an any-angle segment wide enough for a robot.
fn line_of_sight<T: Numeric + Primal, C: TraversalCost<T>>(
    geometry: &GridGeometry<T>,
    cost: &C,
    from: [T; 2],
    into: [T; 2],
) -> bool {
    let separation = [into[0] - from[0], into[1] - from[1]];
    let distance = separation[0].hypot(separation[1]);
    if distance == T::ZERO {
        return true;
    }
    let bearing = separation[1].atan2(separation[0]);
    geometry
        .walk(from, bearing, distance)
        .all(|step| cost.cost_of(step.row, step.column).is_some())
}

/// The cell `(row + row_offset, column + column_offset)`, or `None` off the grid.
fn step_from<T: Numeric + Primal>(
    geometry: &GridGeometry<T>,
    row: usize,
    column: usize,
    row_offset: isize,
    column_offset: isize,
) -> Option<(usize, usize)> {
    let next_row = row.checked_add_signed(row_offset)?;
    let next_column = column.checked_add_signed(column_offset)?;
    geometry
        .contains(next_row, next_column)
        .then_some((next_row, next_column))
}

/// The world middle of the cell a flat index names.
fn centre_of_index<T: Numeric + Primal>(
    geometry: &GridGeometry<T>,
    index: usize,
) -> Option<[T; 2]> {
    let (row, column) = geometry.cell_at(index)?;
    geometry.center_of(row, column)
}

fn read<const MAX_CELLS: usize, T: Numeric>(values: &[T; MAX_CELLS], index: usize) -> T {
    values.get(index).copied().unwrap_or(T::INFINITY)
}

fn read_parent<const MAX_CELLS: usize>(parents: &[u32; MAX_CELLS], index: usize) -> u32 {
    parents.get(index).copied().unwrap_or(NO_PARENT)
}

fn read_visit<const MAX_CELLS: usize>(visits: &[CellVisit; MAX_CELLS], index: usize) -> CellVisit {
    visits.get(index).copied().unwrap_or(CellVisit::Closed)
}

fn set<const MAX_CELLS: usize, V>(values: &mut [V; MAX_CELLS], index: usize, value: V) {
    if let Some(slot) = values.get_mut(index) {
        *slot = value;
    }
}
