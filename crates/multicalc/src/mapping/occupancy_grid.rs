#![deny(clippy::indexing_slicing)]

//! A map of free and blocked cells, and walking a beam across it.

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};

#[cfg(feature = "alloc")]
use crate::error::MappingError;
use crate::mapping::grid_geometry::GridGeometry;
use crate::mapping::scan_geometry::ScanGeometry;
use crate::scalar::{Numeric, Primal};
use crate::spatial::SE2;

/// Whether a cell is free, blocked, or not yet observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    /// Observed and passable.
    Free,
    /// Observed and blocked.
    Occupied,
    /// Not yet observed. A planner must not route through it.
    Unknown,
}

/// A map made of square cells, each free or blocked, that a beam can be cast across.
///
/// Cells are named row first, as they are stored: cell `(row, column)` covers the world square
/// starting at `origin + [column · resolution, row · resolution]`, so `origin` is the lowest corner
/// of cell `(0, 0)`: row `0` is the lowest `y` and column `0` the lowest `x`.
pub trait OccupancyMap<T: Numeric + Primal = f64> {
    /// How many cells across.
    #[must_use]
    fn columns(&self) -> usize;

    /// How many cells up.
    #[must_use]
    fn rows(&self) -> usize;

    /// The edge length of one cell.
    #[must_use]
    fn resolution(&self) -> T;

    /// The world corner of cell `(0, 0)`: its lowest `x` and lowest `y`.
    #[must_use]
    fn origin(&self) -> [T; 2];

    /// Whether the cell is blocked. A cell outside the map reads as free.
    #[must_use]
    fn is_occupied(&self, row: usize, column: usize) -> bool;

    /// The grid's placement and index arithmetic.
    #[must_use]
    fn geometry(&self) -> GridGeometry<T> {
        GridGeometry::from_parts(
            self.rows(),
            self.columns(),
            self.resolution(),
            self.origin(),
        )
    }

    /// The cell holding `point`, as `(row, column)`, or `None` when the point lies outside the map.
    ///
    /// `point` is a world point in the same coordinates as [`origin`](Self::origin). A point
    /// sitting exactly on the edge between two cells belongs to the higher one, so a cell holds
    /// its lower edge and not its upper.
    ///
    /// ```
    /// use multicalc::mapping::OccupancyMap;
    ///
    /// // A 4 by 4 patch of half-metre cells whose lowest corner sits at (-1, -1).
    /// struct Patch;
    ///
    /// impl OccupancyMap for Patch {
    ///     fn columns(&self) -> usize { 4 }
    ///     fn rows(&self) -> usize { 4 }
    ///     fn resolution(&self) -> f64 { 0.5 }
    ///     fn origin(&self) -> [f64; 2] { [-1.0, -1.0] }
    ///     fn is_occupied(&self, _row: usize, _column: usize) -> bool { false }
    /// }
    ///
    /// let patch = Patch;
    ///
    /// // The lowest corner falls in cell (0, 0), and so does anything within its half metre.
    /// let lowest_corner = [-1.0, -1.0];
    /// let just_inside = [-0.6, -0.9];
    /// assert_eq!(patch.cell_of(lowest_corner), Some((0, 0)));
    /// assert_eq!(patch.cell_of(just_inside), Some((0, 0)));
    ///
    /// // Row comes first: half a metre up is the next row, half a metre across the next column.
    /// let one_up = [-1.0, -0.5];
    /// let one_across = [-0.5, -1.0];
    /// assert_eq!(patch.cell_of(one_up), Some((1, 0)));
    /// assert_eq!(patch.cell_of(one_across), Some((0, 1)));
    ///
    /// // The far corner is past the last cell, and anything below the lowest corner is off the map.
    /// let far_corner = [1.0, 1.0];
    /// let below = [-1.1, -1.0];
    /// assert_eq!(patch.cell_of(far_corner), None);
    /// assert_eq!(patch.cell_of(below), None);
    /// ```
    fn cell_of(&self, point: [T; 2]) -> Option<(usize, usize)> {
        self.geometry().cell_of(point)
    }

    /// How far a beam fired from `start_position` travels before it meets a blocked cell, or
    /// `None` when it meets none within `maximum_range`.
    ///
    /// `start_position` is a world point in the same coordinates as [`origin`](Self::origin), a
    /// place on the map, not a cell index. `bearing` is the direction to fire in, in radians,
    /// measured from the direction of rising `x` and turning toward rising `y`: `0` fires along
    /// the row the beam starts on, and a quarter turn fires up the column. The distance that comes
    /// back is measured from `start_position` and is in the same units as
    /// [`resolution`](Self::resolution).
    ///
    /// The beam is walked one cell at a time. The distance reported is where it crosses into the
    /// blocked cell, so a wall lining up with a cell edge reads exactly rather than rounded to a
    /// cell. A beam that starts on a blocked cell reads zero, and one that starts off the map is
    /// walked from where it first enters.
    ///
    /// ```
    /// use multicalc::mapping::OccupancyMap;
    ///
    /// // Any storage can be a map. This one is a 4 by 4 room of half-metre cells with its lowest
    /// // corner at the world origin.
    /// struct Room {
    ///     cells: [[bool; 4]; 4],
    /// }
    ///
    /// impl OccupancyMap for Room {
    ///     fn columns(&self) -> usize { 4 }
    ///     fn rows(&self) -> usize { 4 }
    ///     fn resolution(&self) -> f64 { 0.5 }
    ///     fn origin(&self) -> [f64; 2] { [0.0, 0.0] }
    ///     fn is_occupied(&self, row: usize, column: usize) -> bool {
    ///         self.cells.get(row).and_then(|row| row.get(column)).copied().unwrap_or(false)
    ///     }
    /// }
    ///
    /// // A wall running up the third column, blocking every row.
    /// let blocked_column = 2;
    /// let cells = core::array::from_fn(|_| core::array::from_fn(|column| column == blocked_column));
    /// let room = Room { cells };
    ///
    /// // Standing in the middle of the lowest-left cell and firing along the row: the wall is met
    /// // where its column begins.
    /// let cell_size = 0.5;
    /// let standing_at = [0.25, 0.25];
    /// let along_the_row = 0.0;
    /// let maximum_range = 4.0;
    /// let wall_begins_at = blocked_column as f64 * cell_size;
    /// let expected = wall_begins_at - standing_at[0];
    /// let distance = room.cast_ray(standing_at, along_the_row, maximum_range);
    /// assert!(distance.is_some_and(|met| (met - expected).abs() < 1e-12));
    ///
    /// // Firing up the column instead, nothing is in the way.
    /// let up_the_column = core::f64::consts::FRAC_PI_2;
    /// assert!(room.cast_ray(standing_at, up_the_column, maximum_range).is_none());
    ///
    /// // A range too short to reach the wall meets nothing either.
    /// let short_range = 0.5;
    /// assert!(room.cast_ray(standing_at, along_the_row, short_range).is_none());
    /// ```
    fn cast_ray(&self, start_position: [T; 2], bearing: T, maximum_range: T) -> Option<T> {
        self.geometry()
            .walk(start_position, bearing, maximum_range)
            .find(|step| self.is_occupied(step.row, step.column))
            .map(|step| step.entry_distance)
    }

    /// Whether the cell is free, blocked, or not yet observed.
    ///
    /// The default answers from [`is_occupied`](Self::is_occupied) and never reports
    /// [`Unknown`](CellState::Unknown); a belief map overrides it.
    #[must_use]
    fn cell_state(&self, row: usize, column: usize) -> CellState {
        if self.is_occupied(row, column) {
            CellState::Occupied
        } else {
            CellState::Free
        }
    }

    /// The range each beam of `scan` reads from `pose`, its maximum range where it meets nothing.
    ///
    /// ```
    /// use multicalc::mapping::{MutableOccupancyMap, OccupancyGrid, OccupancyMap, ScanGeometry};
    /// use multicalc::{SE2, SO2, Vector2D};
    ///
    /// // A 4 m square room at 10 cm cells, with a wall two metres east of the middle.
    /// let mut room: OccupancyGrid<40, 40, 2> = OccupancyGrid::try_new(0.1, [0.0, 0.0])?;
    /// let wall = [[3.0, 0.5], [3.0, 3.5]];
    /// room.occupy_polyline(&wall, false);
    ///
    /// // A three-beam scan facing east from the middle: the centre beam meets the wall.
    /// let scan: ScanGeometry<3> = ScanGeometry::try_new(core::f64::consts::FRAC_PI_2, 4.0)?;
    /// let facing_east = SE2::from_parts(SO2::from_angle(0.0), Vector2D::new([2.0, 2.0]));
    /// let ranges = room.cast_scan(facing_east, &scan);
    /// assert!((ranges[1] - 1.0).abs() <= 0.1);
    /// # Ok::<(), multicalc::CalcError>(())
    /// ```
    #[must_use]
    fn cast_scan<const NUM_BEAMS: usize>(
        &self,
        pose: SE2<T>,
        scan: &ScanGeometry<NUM_BEAMS, T>,
    ) -> [T; NUM_BEAMS] {
        let position = pose.translation().into_array();
        let heading = pose.rotation().log();
        core::array::from_fn(|beam| {
            scan.beam_angle(beam)
                .and_then(|offset| self.cast_ray(position, heading + offset, scan.maximum_range()))
                .unwrap_or(scan.maximum_range())
        })
    }

    /// Where each beam of `scan` ends, in world coordinates.
    #[must_use]
    fn scan_endpoints<const NUM_BEAMS: usize>(
        &self,
        pose: SE2<T>,
        scan: &ScanGeometry<NUM_BEAMS, T>,
    ) -> [[T; 2]; NUM_BEAMS] {
        let position = pose.translation().into_array();
        let heading = pose.rotation().log();
        let ranges = self.cast_scan(pose, scan);
        core::array::from_fn(|beam| {
            let bearing = heading + scan.beam_angle(beam).unwrap_or(T::ZERO);
            let range = ranges.get(beam).copied().unwrap_or(scan.maximum_range());
            [
                position[0] + range * bearing.cos(),
                position[1] + range * bearing.sin(),
            ]
        })
    }
}

/// A map whose cells can be marked, from world geometry rather than cell by cell.
pub trait MutableOccupancyMap<T: Numeric + Primal = f64>: OccupancyMap<T> {
    /// Marks a cell. An index outside the map does nothing.
    fn set_cell(&mut self, row: usize, column: usize, occupied: bool);

    /// Frees every cell.
    fn clear(&mut self);

    /// Blocks the cell holding `point`. A point outside the map does nothing.
    fn occupy_point(&mut self, point: [T; 2]) {
        if let Some((row, column)) = self.cell_of(point) {
            self.set_cell(row, column, true);
        }
    }

    /// Blocks the cells along each edge of a list of points. `closed` joins the last point back to
    /// the first. Each edge is sampled well inside a cell width, so a wall it draws has no gap a
    /// beam could slip through.
    ///
    /// It draws the outline and nothing else: the space a closed shape encloses is left free.
    ///
    /// ```
    /// # use multicalc::mapping::{MutableOccupancyMap, OccupancyMap};
    /// # struct Field { cells: [[bool; 20]; 20] }
    /// # impl Field { fn new() -> Self { Field { cells: [[false; 20]; 20] } } }
    /// # impl OccupancyMap for Field {
    /// #     fn columns(&self) -> usize { 20 }
    /// #     fn rows(&self) -> usize { 20 }
    /// #     fn resolution(&self) -> f64 { 0.1 }
    /// #     fn origin(&self) -> [f64; 2] { [0.0, 0.0] }
    /// #     fn is_occupied(&self, row: usize, column: usize) -> bool {
    /// #         self.cells.get(row).and_then(|row| row.get(column)).copied().unwrap_or(false)
    /// #     }
    /// # }
    /// # impl MutableOccupancyMap for Field {
    /// #     fn set_cell(&mut self, row: usize, column: usize, occupied: bool) {
    /// #         if let Some(cell) = self.cells.get_mut(row).and_then(|row| row.get_mut(column)) {
    /// #             *cell = occupied;
    /// #         }
    /// #     }
    /// #     fn clear(&mut self) { self.cells = [[false; 20]; 20]; }
    /// # }
    /// // A 2 m square field of 10 cm cells, with a 1 m box drawn in the middle of it.
    /// let mut field = Field::new();
    /// let corners = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
    /// let joined_up = true;
    /// field.occupy_polyline(&corners, joined_up);
    ///
    /// // The box is an outline, so the middle of it is still free.
    /// let middle = [1.0, 1.0];
    /// assert_eq!(field.cell_of(middle), Some((10, 10)));
    /// assert!(!field.is_occupied(10, 10));
    ///
    /// // A beam fired from the middle meets a wall whichever way it goes: there is no gap to slip
    /// // through, and every wall is half a metre away.
    /// let cell_size = 0.1;
    /// let half_a_metre = 0.5;
    /// let maximum_range = 2.0;
    /// let straight_at_a_wall = [0.0, core::f64::consts::FRAC_PI_2, core::f64::consts::PI];
    /// for bearing in straight_at_a_wall {
    ///     let distance = field.cast_ray(middle, bearing, maximum_range);
    ///     assert!(distance.is_some_and(|met| (met - half_a_metre).abs() <= cell_size));
    /// }
    ///
    /// // Left open, the edge back to the first corner is never drawn, and a beam fired that way
    /// // runs off the field instead.
    /// let mut open_field = Field::new();
    /// let left_open = false;
    /// open_field.occupy_polyline(&corners, left_open);
    /// let toward_the_missing_edge = core::f64::consts::PI;
    /// assert!(
    ///     open_field
    ///         .cast_ray(middle, toward_the_missing_edge, maximum_range)
    ///         .is_none()
    /// );
    /// ```
    fn occupy_polyline(&mut self, polyline: &[[T; 2]], closed: bool) {
        let step = T::from_f64(0.4) * self.resolution();
        let count = polyline.len();
        let edges = if closed {
            count
        } else {
            count.saturating_sub(1)
        };
        for index in 0..edges {
            let (Some(&start), Some(&end)) = (
                polyline.get(index),
                polyline.get((index + 1) % count.max(1)),
            ) else {
                continue;
            };
            let length = (end[0] - start[0]).hypot(end[1] - start[1]);
            let samples = (length / step).ceil().max(T::ONE).to_f64() as usize;
            for sample in 0..=samples {
                let fraction = T::from_usize(sample) / T::from_usize(samples);
                self.occupy_point([
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                ]);
            }
        }
    }

    /// Blocks the cells around a circle's rim — the outline, not the filled disc.
    ///
    /// The rim is walked in steps small enough that consecutive points land in the same cell or the
    /// next one, so it closes on itself with no gap. A radius smaller than a cell marks one cell or
    /// a handful.
    ///
    /// ```
    /// # use multicalc::mapping::{MutableOccupancyMap, OccupancyMap};
    /// # struct Field { cells: [[bool; 20]; 20] }
    /// # impl Field { fn new() -> Self { Field { cells: [[false; 20]; 20] } } }
    /// # impl OccupancyMap for Field {
    /// #     fn columns(&self) -> usize { 20 }
    /// #     fn rows(&self) -> usize { 20 }
    /// #     fn resolution(&self) -> f64 { 0.1 }
    /// #     fn origin(&self) -> [f64; 2] { [0.0, 0.0] }
    /// #     fn is_occupied(&self, row: usize, column: usize) -> bool {
    /// #         self.cells.get(row).and_then(|row| row.get(column)).copied().unwrap_or(false)
    /// #     }
    /// # }
    /// # impl MutableOccupancyMap for Field {
    /// #     fn set_cell(&mut self, row: usize, column: usize, occupied: bool) {
    /// #         if let Some(cell) = self.cells.get_mut(row).and_then(|row| row.get_mut(column)) {
    /// #             *cell = occupied;
    /// #         }
    /// #     }
    /// #     fn clear(&mut self) { self.cells = [[false; 20]; 20]; }
    /// # }
    /// // A 2 m square field of 10 cm cells, with a pillar drawn in the middle of it.
    /// let mut field = Field::new();
    /// let middle = [1.0, 1.0];
    /// let radius = 0.5;
    /// field.occupy_circle(middle, radius);
    ///
    /// // The pillar is a rim, so the middle of it is still free.
    /// assert!(!field.is_occupied(10, 10));
    ///
    /// // A beam fired from the middle meets the rim at the radius, whichever way it goes — within
    /// // a cell, since that is as finely as a rim can be drawn.
    /// let cell_size = 0.1;
    /// let maximum_range = 2.0;
    /// for step in 0..16 {
    ///     let bearing = core::f64::consts::TAU * step as f64 / 16.0;
    ///     let distance = field.cast_ray(middle, bearing, maximum_range);
    ///     assert!(distance.is_some_and(|met| (met - radius).abs() <= cell_size));
    /// }
    /// ```
    fn occupy_circle(&mut self, center: [T; 2], radius: T) {
        let step = (T::from_f64(0.4) * self.resolution() / radius).max(T::from_f64(1e-3));
        let mut angle = T::ZERO;
        while angle < T::TWO_PI {
            self.occupy_point([
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin(),
            ]);
            angle += step;
        }
    }
}

/// A map of square cells sized when the program runs, each free or blocked — what a map read from a
/// file or sized from a sensor needs.
///
/// Cells are named row first, as they are stored: cell `(row, column)` covers the world square
/// starting at `origin + [column · resolution, row · resolution]`.
///
/// ```
/// use core::f64::consts::{FRAC_PI_2, PI};
///
/// use multicalc::mapping::{DynamicOccupancyGrid, MutableOccupancyMap, OccupancyMap};
///
/// // A 20 m by 15 m warehouse floor at 5 cm cells: 400 across by 300 up, 120,000 cells in all.
/// let floor_width = 20.0_f64;
/// let floor_depth = 15.0;
/// let cell_size = 0.05;
/// let columns = (floor_width / cell_size) as usize;
/// let rows = (floor_depth / cell_size) as usize;
/// let lowest_corner = [0.0, 0.0];
/// let mut floor = DynamicOccupancyGrid::try_new(columns, rows, cell_size, lowest_corner)?;
/// assert_eq!(floor.columns(), 400);
/// assert_eq!(floor.rows(), 300);
///
/// // The outer walls, drawn a decimetre inside the edge so the whole loop lands on the map.
/// let margin = 0.1;
/// let walls = [
///     [margin, margin],
///     [floor_width - margin, margin],
///     [floor_width - margin, floor_depth - margin],
///     [margin, floor_depth - margin],
/// ];
/// let joined_up = true;
/// floor.occupy_polyline(&walls, joined_up);
///
/// // A run of shelving along the south side, and a roof pillar in the middle of the floor.
/// let shelving = [[2.0, 2.0], [8.0, 2.0], [8.0, 3.0], [2.0, 3.0]];
/// floor.occupy_polyline(&shelving, joined_up);
/// let pillar_centre = [10.0, 7.5];
/// let pillar_radius = 0.4;
/// floor.occupy_circle(pillar_centre, pillar_radius);
///
/// // A robot parked halfway down the floor, five metres west of the pillar, taking a scan.
/// let robot = [5.0, 7.5];
/// let maximum_range = 10.0;
///
/// // Anything drawn is a cell thick, so the face a beam meets can sit a cell either side of the
/// // line that drew it. At 5 cm cells that is what a scan against a map can be trusted to.
/// let within_a_cell_or_two = 2.0 * cell_size;
///
/// // East: the near face of the pillar, at its centre less its radius.
/// let east = 0.0;
/// let pillar_face = pillar_centre[0] - pillar_radius - robot[0];
/// let ahead = floor.cast_ray(robot, east, maximum_range);
/// assert!(ahead.is_some_and(|met| (met - pillar_face).abs() <= within_a_cell_or_two));
///
/// // North: nothing until the far wall, seven and a half metres up.
/// let north = FRAC_PI_2;
/// let wall_face = floor_depth - margin - robot[1];
/// let above = floor.cast_ray(robot, north, maximum_range);
/// assert!(above.is_some_and(|met| (met - wall_face).abs() <= within_a_cell_or_two));
///
/// // South: the shelving stops the beam well before the wall behind it.
/// let south = -FRAC_PI_2;
/// let shelving_near_side = 3.0;
/// let shelving_face = robot[1] - shelving_near_side;
/// let below = floor.cast_ray(robot, south, maximum_range);
/// assert!(below.is_some_and(|met| (met - shelving_face).abs() <= within_a_cell_or_two));
///
/// // West: open floor all the way to the wall, which is nearer than the beam can see.
/// let west = PI;
/// let behind = floor.cast_ray(robot, west, maximum_range);
/// assert!(behind.is_some_and(|met| met < robot[0] && met > robot[0] - 2.0 * margin));
///
/// // A shorter-sighted sensor in the same spot sees nothing at all to the west.
/// let short_sighted = 4.0;
/// assert!(floor.cast_ray(robot, west, short_sighted).is_none());
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicOccupancyGrid<T: Numeric + Primal = f64> {
    columns: usize,
    rows: usize,
    resolution: T,
    origin: [T; 2],
    cells: Vec<bool>,
}

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
impl<T: Numeric + Primal> DynamicOccupancyGrid<T> {
    /// A map of the given size with every cell free.
    ///
    /// `resolution` is the edge length of one cell and `origin` is the world corner of cell
    /// `(0, 0)`, its lowest `x` and lowest `y`.
    ///
    /// Returns [`MappingError::EmptyGrid`] with no columns or no rows,
    /// [`MappingError::GridTooLarge`] when the two multiplied out are more cells than can be
    /// counted, [`MappingError::NonFinite`] if the cell size or origin is not finite, and
    /// [`MappingError::NonPositiveResolution`] if the cell size is zero or negative.
    pub fn try_new(
        columns: usize,
        rows: usize,
        resolution: T,
        origin: [T; 2],
    ) -> Result<Self, MappingError> {
        if columns == 0 || rows == 0 {
            return Err(MappingError::EmptyGrid);
        }
        let count = columns
            .checked_mul(rows)
            .ok_or(MappingError::GridTooLarge)?;
        if !resolution.is_finite() || !origin[0].is_finite() || !origin[1].is_finite() {
            return Err(MappingError::NonFinite);
        }
        if resolution <= T::ZERO {
            return Err(MappingError::NonPositiveResolution);
        }
        Ok(DynamicOccupancyGrid {
            columns,
            rows,
            resolution,
            origin,
            cells: vec![false; count],
        })
    }

    /// Where a `(row, column)` pair sits in `cells`, or `None` when the pair is off the map.
    ///
    /// The place is worked out only once both indices are known to be on the map, so the
    /// multiplication can neither overflow nor reach past what was allocated.
    fn index_of(&self, row: usize, column: usize) -> Option<usize> {
        (row < self.rows && column < self.columns).then(|| row * self.columns + column)
    }
}

#[cfg(feature = "alloc")]
impl<T: Numeric + Primal> OccupancyMap<T> for DynamicOccupancyGrid<T> {
    fn columns(&self) -> usize {
        self.columns
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn resolution(&self) -> T {
        self.resolution
    }

    fn origin(&self) -> [T; 2] {
        self.origin
    }

    fn is_occupied(&self, row: usize, column: usize) -> bool {
        self.index_of(row, column)
            .and_then(|index| self.cells.get(index))
            .copied()
            .unwrap_or(false)
    }
}

#[cfg(feature = "alloc")]
impl<T: Numeric + Primal> MutableOccupancyMap<T> for DynamicOccupancyGrid<T> {
    fn set_cell(&mut self, row: usize, column: usize, occupied: bool) {
        if let Some(cell) = self
            .index_of(row, column)
            .and_then(|index| self.cells.get_mut(index))
        {
            *cell = occupied;
        }
    }

    fn clear(&mut self) {
        self.cells.iter_mut().for_each(|cell| *cell = false);
    }
}
