#![deny(clippy::indexing_slicing)]

//! The nav2 inflation formulation: a per-cell cost decaying outward from an obstacle.

use crate::error::MappingError;
use crate::mapping::distance_field::DistanceField;
use crate::mapping::grid_geometry::GridGeometry;
use crate::scalar::{Numeric, Primal};

/// Per-cell traversal cost inflated outward from the obstacles in a distance field.
///
/// From a cell's distance `d` to the nearest obstacle:
///
/// ```text
/// d <= inscribed_radius                    -> LETHAL
/// inscribed_radius < d <= inflation_radius -> 254 · exp(−cost_scaling_factor · (d − inscribed_radius))
/// d > inflation_radius                     -> 0
/// ```
///
/// `inflation_radius` sets how far cost spreads and `cost_scaling_factor` how steeply it decays.
/// `inscribed_radius` is the radius of the largest circle that fits inside the robot's footprint,
/// so a cell within it is one no part of the robot may enter.
///
/// The planner adapter that reads this — `CostmapCost` — lives in `planning`, which owns the
/// traversal-cost trait. Putting it here would make the two modules mutually recursive.
///
/// ```
/// use multicalc::mapping::{
///     CostGrid, DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid,
/// };
///
/// // A 2 m square at 10 cm cells with a pillar in the middle.
/// let mut room: OccupancyGrid<20, 20, 1> = OccupancyGrid::try_new(0.1, [0.0, 0.0])?;
/// room.set_cell(10, 10, true);
///
/// let mut workspace: DistanceTransformWorkspace<21> = DistanceTransformWorkspace::new();
/// let field: DistanceField<20, 20> = DistanceField::try_build(&room, &mut workspace)?;
///
/// let inscribed_radius = 0.2;
/// let inflation_radius = 0.8;
/// let cost_scaling_factor = 3.0;
/// let costmap: CostGrid<20, 20> =
///     CostGrid::try_build(&field, inscribed_radius, inflation_radius, cost_scaling_factor)?;
///
/// // On the pillar and just around it, nothing may enter; far away it costs nothing.
/// assert_eq!(costmap.cost_of(10, 10), Some(CostGrid::<20, 20>::LETHAL));
/// assert_eq!(costmap.cost_of(0, 0), Some(0));
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostGrid<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal = f64> {
    costs: [[u8; COLUMNS]; ROWS],
    geometry: GridGeometry<T>,
}

impl<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal> CostGrid<ROWS, COLUMNS, T> {
    /// A cell no part of the robot may enter.
    pub const LETHAL: u8 = 255;

    /// The highest cost a cell can carry and still be passable.
    const HIGHEST_PASSABLE: u64 = 254;

    /// The inflation of a distance field.
    ///
    /// Returns [`MappingError::NonFinite`] for a non-finite radius or scaling,
    /// [`MappingError::NonPositiveRange`] for a negative `inscribed_radius` or a non-positive
    /// `inflation_radius`, [`MappingError::RadiiNotOrdered`] when the inscribed radius exceeds the
    /// inflation radius, and [`MappingError::NonPositiveScaling`] for a non-positive scaling
    /// factor.
    ///
    /// O(cells), no scratch.
    pub fn try_build(
        field: &DistanceField<ROWS, COLUMNS, T>,
        inscribed_radius: T,
        inflation_radius: T,
        cost_scaling_factor: T,
    ) -> Result<Self, MappingError> {
        if !inscribed_radius.is_finite()
            || !inflation_radius.is_finite()
            || !cost_scaling_factor.is_finite()
        {
            return Err(MappingError::NonFinite);
        }
        if inscribed_radius < T::ZERO || inflation_radius <= T::ZERO {
            return Err(MappingError::NonPositiveRange);
        }
        if inscribed_radius > inflation_radius {
            return Err(MappingError::RadiiNotOrdered);
        }
        if cost_scaling_factor <= T::ZERO {
            return Err(MappingError::NonPositiveScaling);
        }

        let highest = T::from_u64(Self::HIGHEST_PASSABLE);
        let mut costs = [[0u8; COLUMNS]; ROWS];
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let Some(distance) = field.distance_of(row, column) else {
                    continue;
                };
                let cost = if distance <= inscribed_radius {
                    Self::LETHAL
                } else if distance <= inflation_radius {
                    let decayed =
                        highest * (-cost_scaling_factor * (distance - inscribed_radius)).exp();
                    decayed.to_f64() as u8
                } else {
                    0
                };
                if let Some(cell) = costs
                    .get_mut(row)
                    .and_then(|row_costs| row_costs.get_mut(column))
                {
                    *cell = cost;
                }
            }
        }

        Ok(CostGrid {
            costs,
            geometry: field.geometry(),
        })
    }

    /// The grid's placement and index arithmetic.
    #[inline]
    #[must_use]
    pub fn geometry(&self) -> GridGeometry<T> {
        self.geometry
    }

    /// A cell's cost, or `None` off the grid.
    #[inline]
    #[must_use]
    pub fn cost_of(&self, row: usize, column: usize) -> Option<u8> {
        self.costs
            .get(row)
            .and_then(|row_costs| row_costs.get(column))
            .copied()
    }

    /// The cost at a world point, or `None` off the grid.
    #[inline]
    #[must_use]
    pub fn cost_at(&self, point: [T; 2]) -> Option<u8> {
        let (row, column) = self.geometry.cell_of(point)?;
        self.cost_of(row, column)
    }
}
