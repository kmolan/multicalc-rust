#![deny(clippy::indexing_slicing)]

//! A map of free and blocked cells, and walking a beam across it.

use crate::scalar::{Numeric, Primal};

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
        let origin = self.origin();
        let column = ((point[0] - origin[0]) / self.resolution())
            .floor()
            .to_f64();
        let row = ((point[1] - origin[1]) / self.resolution())
            .floor()
            .to_f64();
        if column < 0.0 || row < 0.0 {
            return None;
        }
        let (row, column) = (row as usize, column as usize);
        (column < self.columns() && row < self.rows()).then_some((row, column))
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
        if self.columns() == 0 || self.rows() == 0 {
            return None;
        }
        let direction = [bearing.cos(), bearing.sin()];

        // Where the beam meets the map. A beam starting inside enters at zero.
        let entry = entry_distance(self, start_position, direction)?;
        if entry > maximum_range {
            return None;
        }
        let point = [
            start_position[0] + entry * direction[0],
            start_position[1] + entry * direction[1],
        ];

        // The cell the beam is in as it enters. Rounding at an edge can push the index just
        // outside, so it is forced back on.
        let lowest_corner = self.origin();
        let resolution = self.resolution();
        let mut column = clamp_index((point[0] - lowest_corner[0]) / resolution, self.columns());
        let mut row = clamp_index((point[1] - lowest_corner[1]) / resolution, self.rows());

        // Which way each index moves, and how far along the beam one whole cell is. A beam running
        // parallel to an axis never crosses that axis's edges, so its stride is infinite and the
        // direction it would have stepped never comes up.
        let step_column = axis_step(direction[0]);
        let step_row = axis_step(direction[1]);
        let stride_column = if direction[0] == T::ZERO {
            T::INFINITY
        } else {
            resolution / direction[0].abs()
        };
        let stride_row = if direction[1] == T::ZERO {
            T::INFINITY
        } else {
            resolution / direction[1].abs()
        };

        // How far along the beam the next edge on each axis is.
        let mut next_column = boundary_distance(
            start_position[0],
            direction[0],
            column,
            step_column,
            lowest_corner[0],
            resolution,
        );
        let mut next_row = boundary_distance(
            start_position[1],
            direction[1],
            row,
            step_row,
            lowest_corner[1],
            resolution,
        );

        // How far along the beam the current cell was entered.
        let mut entered = entry;
        loop {
            if entered > maximum_range {
                return None;
            }
            if column < 0 || row < 0 {
                return None;
            }
            let (column_index, row_index) = (column as usize, row as usize);
            if column_index >= self.columns() || row_index >= self.rows() {
                return None;
            }
            if self.is_occupied(row_index, column_index) {
                return Some(entered);
            }
            // Cross whichever edge comes first.
            if next_column < next_row {
                column += step_column;
                entered = next_column;
                next_column += stride_column;
            } else {
                row += step_row;
                entered = next_row;
                next_row += stride_row;
            }
        }
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

/// How far along the beam `start_position + distance · direction` the map is entered, or `None` if
/// it never is. A beam already inside returns zero.
fn entry_distance<T: Numeric + Primal, M: OccupancyMap<T> + ?Sized>(
    map: &M,
    start_position: [T; 2],
    direction: [T; 2],
) -> Option<T> {
    let lowest_corner = map.origin();
    let highest_corner = [
        lowest_corner[0] + T::from_usize(map.columns()) * map.resolution(),
        lowest_corner[1] + T::from_usize(map.rows()) * map.resolution(),
    ];
    let mut near = T::NEG_INFINITY;
    let mut far = T::INFINITY;
    for (start, step, low, high) in [
        (
            start_position[0],
            direction[0],
            lowest_corner[0],
            highest_corner[0],
        ),
        (
            start_position[1],
            direction[1],
            lowest_corner[1],
            highest_corner[1],
        ),
    ] {
        if step == T::ZERO {
            // Running parallel to this pair of edges: only a beam already between them can enter.
            if start < low || start > high {
                return None;
            }
        } else {
            let inverse = T::ONE / step;
            let mut first = (low - start) * inverse;
            let mut second = (high - start) * inverse;
            if first > second {
                core::mem::swap(&mut first, &mut second);
            }
            near = near.max(first);
            far = far.min(second);
        }
    }
    if near > far || far < T::ZERO {
        return None;
    }
    Some(near.max(T::ZERO))
}

/// Which way an index moves as the beam advances along one axis. A component of zero never steps,
/// because the stride along that axis is infinite, so either answer serves.
fn axis_step<T: Numeric>(component: T) -> isize {
    if component > T::ZERO { 1 } else { -1 }
}

/// How far along the beam the far edge of the current cell on one axis is.
fn boundary_distance<T: Numeric>(
    start_axis: T,
    direction_axis: T,
    index: isize,
    step: isize,
    lowest_corner_axis: T,
    resolution: T,
) -> T {
    if direction_axis == T::ZERO {
        return T::INFINITY;
    }
    let next_index = if step > 0 { index + 1 } else { index };
    let boundary = lowest_corner_axis + T::from_f64(next_index as f64) * resolution;
    (boundary - start_axis) / direction_axis
}

/// Floors a coordinate to a cell index, held inside `[0, length)` so rounding at an edge cannot
/// land outside the map.
fn clamp_index<T: Numeric + Primal>(value: T, length: usize) -> isize {
    let floored = value.floor().to_f64();
    if floored < 0.0 {
        0
    } else if floored as usize >= length {
        length as isize - 1
    } else {
        floored as isize
    }
}
