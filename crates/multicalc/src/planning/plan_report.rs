#![deny(clippy::indexing_slicing)]

//! A plan, its cost, and what the search spent finding it.

use crate::motion::PolylinePath;
use crate::scalar::Numeric;

/// What a planner found: the path, what it costs, and how much search it took.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanReport<const MAX_POINTS: usize, const DIMENSION: usize, T: Numeric = f64> {
    path: PolylinePath<MAX_POINTS, DIMENSION, T>,
    cost: T,
    iterations: usize,
}

impl<const MAX_POINTS: usize, const DIMENSION: usize, T: Numeric>
    PlanReport<MAX_POINTS, DIMENSION, T>
{
    pub(crate) fn new(
        path: PolylinePath<MAX_POINTS, DIMENSION, T>,
        cost: T,
        iterations: usize,
    ) -> Self {
        PlanReport {
            path,
            cost,
            iterations,
        }
    }

    /// The waypoints, ready for a smoother or a path follower.
    #[inline]
    pub fn path(&self) -> PolylinePath<MAX_POINTS, DIMENSION, T> {
        self.path
    }

    /// What the plan costs under the traversal cost it was planned against.
    ///
    /// For a uniform cost this is the path's length; a costmap scales each step by the cell it
    /// enters, so the cost exceeds the length wherever the path runs near an obstacle.
    #[inline]
    #[must_use]
    pub fn cost(&self) -> T {
        self.cost
    }

    /// Expansions for a grid search, samples drawn for a sampling planner.
    #[inline]
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// How many waypoints the plan carries.
    #[inline]
    #[must_use]
    pub fn waypoint_count(&self) -> usize {
        self.path.len()
    }
}
