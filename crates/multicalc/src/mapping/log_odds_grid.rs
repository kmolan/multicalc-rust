#![deny(clippy::indexing_slicing)]

//! Per-cell log-odds `l = log(p / (1 − p))`, integrated by addition and clamped.

use crate::error::MappingError;
use crate::mapping::grid_geometry::GridGeometry;
use crate::mapping::occupancy_grid::{CellState, OccupancyMap};
use crate::mapping::scan_geometry::ScanGeometry;
use crate::scalar::{Numeric, Primal};
use crate::spatial::SE2;

/// A map built from range scans, holding each cell's log-odds of being blocked.
///
/// The Bayesian update collapses to addition in log-odds, which is why it is the standard
/// representation. `l = 0` is exactly "unknown", so a new grid costs nothing to initialise. Clamping
/// to `[clamp_low, clamp_high]` is what lets a cell recover: a person who walks through and leaves
/// does not become a permanent wall.
///
/// Belief is `i8` fixed point, so a 128 by 128 map is 16 KB.
///
/// Unlike a plain occupancy map, a cell off the grid reads
/// [`Unknown`](CellState::Unknown) rather than free — that is what stops a planner routing through
/// unmapped space.
///
/// ```
/// use multicalc::mapping::{CellState, LogOddsGrid, OccupancyMap, ScanGeometry};
/// use multicalc::{SE2, SO2, Vector2D};
///
/// // A 4 m square at 10 cm cells, entirely unobserved to begin with.
/// let mut belief: LogOddsGrid<40, 40> = LogOddsGrid::try_new(0.1, [0.0, 0.0])?;
/// assert_eq!(belief.cell_state(20, 20), CellState::Unknown);
///
/// // A three-beam scan from the middle facing east, reading a wall one metre off.
/// let scan: ScanGeometry<3> = ScanGeometry::try_new(0.2, 4.0)?;
/// let pose = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([2.0, 2.0]));
/// for _ in 0..4 {
///     belief.integrate_scan(pose, &scan, &[1.0; 3]);
/// }
///
/// // The cell the beams stop in is blocked; the ones they crossed to get there are free.
/// assert_eq!(belief.cell_state(20, 30), CellState::Occupied);
/// assert_eq!(belief.cell_state(20, 25), CellState::Free);
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogOddsGrid<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal = f64> {
    belief: [[i8; COLUMNS]; ROWS],
    geometry: GridGeometry<T>,
    free_update: i8,
    occupied_update: i8,
    clamp_low: i8,
    clamp_high: i8,
    free_threshold: i8,
    occupied_threshold: i8,
}

impl<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal> LogOddsGrid<ROWS, COLUMNS, T> {
    /// A grid with every cell unobserved.
    ///
    /// Defaults are `free_update = -2`, `occupied_update = 5`, clamps `±40`, and thresholds `±10`.
    ///
    /// Returns whatever [`GridGeometry::try_new`] rejects the placement with.
    pub fn try_new(resolution: T, origin: [T; 2]) -> Result<Self, MappingError> {
        Ok(LogOddsGrid {
            belief: [[0; COLUMNS]; ROWS],
            geometry: GridGeometry::try_new(ROWS, COLUMNS, resolution, origin)?,
            free_update: -2,
            occupied_update: 5,
            clamp_low: -40,
            clamp_high: 40,
            free_threshold: -10,
            occupied_threshold: 10,
        })
    }

    /// How much one crossing and one hit move a cell's belief.
    ///
    /// Returns [`MappingError::InvalidBeliefSettings`] unless `free < 0 < occupied`.
    pub fn try_with_updates(mut self, free: i8, occupied: i8) -> Result<Self, MappingError> {
        if free >= 0 || occupied <= 0 {
            return Err(MappingError::InvalidBeliefSettings);
        }
        self.free_update = free;
        self.occupied_update = occupied;
        Ok(self)
    }

    /// The range a cell's belief is held inside, which is what lets a cell recover.
    ///
    /// Returns [`MappingError::InvalidBeliefSettings`] unless `low < 0 < high`.
    pub fn try_with_clamps(mut self, low: i8, high: i8) -> Result<Self, MappingError> {
        if low >= 0 || high <= 0 {
            return Err(MappingError::InvalidBeliefSettings);
        }
        self.clamp_low = low;
        self.clamp_high = high;
        Ok(self)
    }

    /// Where belief crosses into reporting free or blocked.
    ///
    /// Returns [`MappingError::InvalidBeliefSettings`] unless
    /// `clamp_low <= free < occupied <= clamp_high`.
    pub fn try_with_thresholds(mut self, free: i8, occupied: i8) -> Result<Self, MappingError> {
        if self.clamp_low > free || free >= occupied || occupied > self.clamp_high {
            return Err(MappingError::InvalidBeliefSettings);
        }
        self.free_threshold = free;
        self.occupied_threshold = occupied;
        Ok(self)
    }

    /// The grid's placement and index arithmetic.
    #[inline]
    #[must_use]
    pub fn geometry(&self) -> GridGeometry<T> {
        self.geometry
    }

    /// A cell's log-odds, or `None` off the grid.
    #[inline]
    #[must_use]
    pub fn belief_at(&self, row: usize, column: usize) -> Option<i8> {
        self.belief
            .get(row)
            .and_then(|row_belief| row_belief.get(column))
            .copied()
    }

    /// Returns every cell to unobserved.
    pub fn reset(&mut self) {
        self.belief = [[0; COLUMNS]; ROWS];
    }

    /// Integrates one scan taken from `pose`.
    ///
    /// Every cell a beam crosses gains `free_update`, the cell it terminates in gains
    /// `occupied_update`, and each is clamped to `[clamp_low, clamp_high]`. A reading at or beyond
    /// `scan.maximum_range()` marks free space only, as does one the scan would not believe.
    ///
    /// Work is bounded by `NUM_BEAMS · maximum_range / resolution` cells.
    pub fn integrate_scan<const NUM_BEAMS: usize>(
        &mut self,
        pose: SE2<T>,
        scan: &ScanGeometry<NUM_BEAMS, T>,
        ranges: &[T; NUM_BEAMS],
    ) {
        let position = pose.translation().into_array();
        let heading = pose.rotation().log();

        for beam in 0..NUM_BEAMS {
            let (Some(offset), Some(&range)) = (scan.beam_angle(beam), ranges.get(beam)) else {
                continue;
            };
            if !scan.range_is_valid(range) {
                continue;
            }
            let bearing = heading + offset;

            // Everything the beam passed through on the way is free.
            for step in self.geometry.walk(position, bearing, range) {
                self.add_to_cell(step.row, step.column, self.free_update);
            }

            // A reading that ran out of range saw nothing to mark blocked.
            if range >= scan.maximum_range() {
                continue;
            }
            let endpoint = [
                position[0] + range * bearing.cos(),
                position[1] + range * bearing.sin(),
            ];
            if let Some((row, column)) = self.geometry.cell_of(endpoint) {
                self.add_to_cell(row, column, self.occupied_update);
            }
        }
    }

    /// Moves one cell's belief and holds it inside the clamps.
    fn add_to_cell(&mut self, row: usize, column: usize, update: i8) {
        let (low, high) = (self.clamp_low, self.clamp_high);
        if let Some(belief) = self
            .belief
            .get_mut(row)
            .and_then(|row_belief| row_belief.get_mut(column))
        {
            *belief = belief.saturating_add(update).clamp(low, high);
        }
    }
}

impl<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal> OccupancyMap<T>
    for LogOddsGrid<ROWS, COLUMNS, T>
{
    fn columns(&self) -> usize {
        COLUMNS
    }

    fn rows(&self) -> usize {
        ROWS
    }

    fn resolution(&self) -> T {
        self.geometry.resolution()
    }

    fn origin(&self) -> [T; 2] {
        self.geometry.origin()
    }

    fn is_occupied(&self, row: usize, column: usize) -> bool {
        self.cell_state(row, column) == CellState::Occupied
    }

    fn geometry(&self) -> GridGeometry<T> {
        self.geometry
    }

    fn cell_state(&self, row: usize, column: usize) -> CellState {
        match self.belief_at(row, column) {
            Some(belief) if belief >= self.occupied_threshold => CellState::Occupied,
            Some(belief) if belief <= self.free_threshold => CellState::Free,
            Some(_) => CellState::Unknown,
            None => CellState::Unknown,
        }
    }
}
