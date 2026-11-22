#![deny(clippy::indexing_slicing)]

//! The arrays a grid search runs in, owned by the caller.

use crate::planning::frontier::Frontier;
use crate::scalar::Numeric;

/// A parent slot meaning the cell has none: the start, or a cell never reached.
pub(crate) const NO_PARENT: u32 = u32::MAX;

/// How far a search has got with a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CellVisit {
    /// Never reached.
    Unvisited,
    /// Reached and queued, so its cost may still fall.
    Open,
    /// Expanded, so its cost is settled.
    Closed,
}

/// The per-cell arrays and the frontier one grid search needs.
///
/// `MAX_CELLS` must be at least the map's `rows · columns`.
///
/// Memory is `2·size_of::<T>() + 13` bytes a cell plus padding — about 33 B/cell at `f64` and
/// 21 B/cell at `f32`, so a 128 by 128 map is roughly 540 KB at `f64` and 340 KB at `f32`. It is
/// caller-owned so it can live in a `static` or, with `alloc`, a `Box`, rather than on the stack.
#[derive(Debug, Clone, Copy)]
pub struct GridSearchWorkspace<const MAX_CELLS: usize, T: Numeric = f64> {
    pub(crate) cost_so_far: [T; MAX_CELLS],
    pub(crate) parent: [u32; MAX_CELLS],
    pub(crate) visit: [CellVisit; MAX_CELLS],
    pub(crate) frontier: Frontier<MAX_CELLS, T>,
}

impl<const MAX_CELLS: usize, T: Numeric> GridSearchWorkspace<MAX_CELLS, T> {
    /// A workspace with every cell unreached.
    #[must_use]
    pub fn new() -> Self {
        GridSearchWorkspace {
            cost_so_far: [T::INFINITY; MAX_CELLS],
            parent: [NO_PARENT; MAX_CELLS],
            visit: [CellVisit::Unvisited; MAX_CELLS],
            frontier: Frontier::new(),
        }
    }

    /// The largest map this workspace can search.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        MAX_CELLS
    }

    /// Returns the first `cells` entries to unreached, so a reused workspace costs the map's size
    /// rather than its capacity.
    pub(crate) fn reset(&mut self, cells: usize) {
        let used = cells.min(MAX_CELLS);
        for cost in self.cost_so_far.iter_mut().take(used) {
            *cost = T::INFINITY;
        }
        for parent in self.parent.iter_mut().take(used) {
            *parent = NO_PARENT;
        }
        for visit in self.visit.iter_mut().take(used) {
            *visit = CellVisit::Unvisited;
        }
        self.frontier.clear_prefix(used);
    }
}

impl<const MAX_CELLS: usize, T: Numeric> Default for GridSearchWorkspace<MAX_CELLS, T> {
    fn default() -> Self {
        Self::new()
    }
}
