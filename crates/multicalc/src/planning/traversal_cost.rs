#![deny(clippy::indexing_slicing)]

//! What entering a cell costs a planner, over a plain map or over an inflation costmap.

use core::marker::PhantomData;

use crate::mapping::{CellState, CostGrid, OccupancyMap};
use crate::scalar::{Numeric, Primal};

/// What entering a cell costs a planner.
pub trait TraversalCost<T: Numeric> {
    /// The multiplier for entering a cell. `None` means it may not be entered.
    fn cost_of(&self, row: usize, column: usize) -> Option<T>;
}

/// Every passable cell costs one.
///
/// An [`Unknown`](CellState::Unknown) cell is impassable unless
/// [`with_unknown_passable`](Self::with_unknown_passable) says otherwise — a planner must not route
/// through space no sensor has seen.
///
/// ```
/// use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid};
/// use multicalc::planning::{TraversalCost, UniformCost};
///
/// let mut room: OccupancyGrid<8, 8, 1> = OccupancyGrid::try_new(0.5, [0.0, 0.0])?;
/// room.set_cell(3, 3, true);
///
/// let cost = UniformCost::new(&room);
/// assert_eq!(cost.cost_of(0, 0), Some(1.0));
/// assert_eq!(cost.cost_of(3, 3), None);
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct UniformCost<'map, M, T: Numeric + Primal = f64> {
    map: &'map M,
    unknown_passable: bool,
    scalar: PhantomData<T>,
}

impl<'map, M: OccupancyMap<T>, T: Numeric + Primal> UniformCost<'map, M, T> {
    /// A uniform cost over `map`, with unknown cells impassable.
    #[must_use]
    pub fn new(map: &'map M) -> Self {
        UniformCost {
            map,
            unknown_passable: false,
            scalar: PhantomData,
        }
    }

    /// Whether a cell no sensor has seen may be routed through.
    #[must_use]
    pub fn with_unknown_passable(mut self, passable: bool) -> Self {
        self.unknown_passable = passable;
        self
    }
}

impl<'map, M: OccupancyMap<T>, T: Numeric + Primal> TraversalCost<T> for UniformCost<'map, M, T> {
    fn cost_of(&self, row: usize, column: usize) -> Option<T> {
        match self.map.cell_state(row, column) {
            CellState::Free => Some(T::ONE),
            CellState::Occupied => None,
            CellState::Unknown => self.unknown_passable.then_some(T::ONE),
        }
    }
}

/// Cost read from an inflation costmap, so a path is pushed off the walls rather than scraping
/// along them.
///
/// The multiplier is `1 + weight · cost / 254`, and a
/// [`LETHAL`](CostGrid::LETHAL) cell or one off the grid may not be entered.
///
/// This adapter lives in `planning`, which owns [`TraversalCost`], rather than beside
/// [`CostGrid`] in `mapping`: writing it there would make the two modules mutually recursive.
#[derive(Debug, Clone, Copy)]
pub struct CostmapCost<'grid, const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal = f64> {
    costmap: &'grid CostGrid<ROWS, COLUMNS, T>,
    weight: T,
}

impl<'grid, const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal>
    CostmapCost<'grid, ROWS, COLUMNS, T>
{
    /// A traversal cost reading `costmap`, at weight one.
    #[must_use]
    pub fn new(costmap: &'grid CostGrid<ROWS, COLUMNS, T>) -> Self {
        CostmapCost {
            costmap,
            weight: T::ONE,
        }
    }

    /// How hard the costmap pushes: the multiplier is `1 + weight · cost / 254`.
    #[must_use]
    pub fn with_weight(mut self, weight: T) -> Self {
        self.weight = weight;
        self
    }
}

impl<'grid, const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal> TraversalCost<T>
    for CostmapCost<'grid, ROWS, COLUMNS, T>
{
    fn cost_of(&self, row: usize, column: usize) -> Option<T> {
        let cost = self.costmap.cost_of(row, column)?;
        if cost == CostGrid::<ROWS, COLUMNS, T>::LETHAL {
            return None;
        }
        Some(T::ONE + self.weight * T::from_u64(u64::from(cost)) / T::from_u64(254))
    }
}
