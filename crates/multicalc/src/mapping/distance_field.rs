#![deny(clippy::indexing_slicing)]

//! The exact Euclidean distance transform, by Felzenszwalb–Huttenlocher: two separable
//! one-dimensional lower-envelope passes, O(cells), no priority queue and no approximation.

use crate::error::MappingError;
use crate::mapping::grid_geometry::GridGeometry;
use crate::mapping::occupancy_grid::{CellState, OccupancyMap};
use crate::scalar::{Numeric, Primal};

/// Stands in for "no obstacle on this line", in squared cell units.
///
/// Large enough that no real squared cell distance reaches it, and finite so that
/// `unreachable − unreachable` is zero rather than NaN — an infinity here would make the envelope's
/// crossing formula indeterminate on an empty line.
fn unreachable<T: Numeric>() -> T {
    T::from_f64(1e20)
}

/// Scratch for one one-dimensional pass of the transform.
///
/// `MAX_SPAN` must be at least `ROWS.max(COLUMNS) + 1`: the envelope carries one more breakpoint
/// than it has parabolas.
#[derive(Debug, Clone, Copy)]
pub struct DistanceTransformWorkspace<const MAX_SPAN: usize, T: Numeric = f64> {
    /// Where each parabola of the lower envelope has its vertex.
    vertices: [usize; MAX_SPAN],
    /// Where consecutive parabolas of the envelope cross.
    intersections: [T; MAX_SPAN],
    /// The line being transformed.
    row_scratch: [T; MAX_SPAN],
    /// Its transform, staged separately because a parabola's vertex can lie right of the sample
    /// being written.
    transformed: [T; MAX_SPAN],
}

impl<const MAX_SPAN: usize, T: Numeric> DistanceTransformWorkspace<MAX_SPAN, T> {
    /// A zeroed workspace.
    #[must_use]
    pub fn new() -> Self {
        DistanceTransformWorkspace {
            vertices: [0; MAX_SPAN],
            intersections: [T::ZERO; MAX_SPAN],
            row_scratch: [T::ZERO; MAX_SPAN],
            transformed: [T::ZERO; MAX_SPAN],
        }
    }

    /// The longest line this workspace can transform, one more than the grid span it serves.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        MAX_SPAN
    }
}

impl<const MAX_SPAN: usize, T: Numeric> Default for DistanceTransformWorkspace<MAX_SPAN, T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Each cell's distance in metres to the nearest blocked cell.
///
/// [`Unknown`](CellState::Unknown) seeds as free, so the distance is to the nearest *known*
/// obstacle. A map with no obstacles at all gives an infinite distance everywhere.
///
/// Building is a design-time or low-rate operation over every cell, never per-tick; the queries are
/// what a loop calls.
///
/// ```
/// use multicalc::mapping::{
///     DistanceField, DistanceTransformWorkspace, MutableOccupancyMap, OccupancyGrid,
/// };
///
/// // A 1 m square at 10 cm cells with one blocked cell in the middle.
/// let mut room: OccupancyGrid<10, 10, 1> = OccupancyGrid::try_new(0.1, [0.0, 0.0])?;
/// room.set_cell(5, 5, true);
///
/// let mut workspace: DistanceTransformWorkspace<11> = DistanceTransformWorkspace::new();
/// let field: DistanceField<10, 10> = DistanceField::try_build(&room, &mut workspace)?;
///
/// // Zero on the obstacle, and the hypotenuse away from it.
/// assert_eq!(field.distance_of(5, 5), Some(0.0));
/// let three_across_and_four_up = 3.0f64.hypot(4.0) * 0.1;
/// assert!(
///     field.distance_of(9, 8).is_some_and(|d| (d - three_across_and_four_up).abs() < 1e-12)
/// );
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceField<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal = f64> {
    distances: [[T; COLUMNS]; ROWS],
    geometry: GridGeometry<T>,
}

impl<const ROWS: usize, const COLUMNS: usize, T: Numeric + Primal> DistanceField<ROWS, COLUMNS, T> {
    /// The distance field of `map`.
    ///
    /// Returns [`MappingError::GridShapeMismatch`] unless the map is `ROWS` by `COLUMNS`,
    /// [`MappingError::WorkspaceTooSmall`] unless `MAX_SPAN >= ROWS.max(COLUMNS) + 1`, and whatever
    /// [`GridGeometry::try_new`] rejects the map's placement with.
    pub fn try_build<const MAX_SPAN: usize, M: OccupancyMap<T>>(
        map: &M,
        workspace: &mut DistanceTransformWorkspace<MAX_SPAN, T>,
    ) -> Result<Self, MappingError> {
        if map.rows() != ROWS || map.columns() != COLUMNS {
            return Err(MappingError::GridShapeMismatch);
        }
        if MAX_SPAN < ROWS.max(COLUMNS) + 1 {
            return Err(MappingError::WorkspaceTooSmall);
        }
        let geometry = GridGeometry::try_new(ROWS, COLUMNS, map.resolution(), map.origin())?;

        // Seed in squared cell units: zero on an obstacle, unreachable elsewhere.
        let mut squared = [[unreachable::<T>(); COLUMNS]; ROWS];
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                if map.cell_state(row, column) == CellState::Occupied {
                    write_cell(&mut squared, row, column, T::ZERO);
                }
            }
        }

        // Along every row, then along every column of that result.
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                write(
                    &mut workspace.row_scratch,
                    column,
                    read(&squared, row, column),
                );
            }
            lower_envelope(workspace, COLUMNS);
            for column in 0..COLUMNS {
                let value = read_line(&workspace.transformed, column);
                write_cell(&mut squared, row, column, value);
            }
        }
        for column in 0..COLUMNS {
            for row in 0..ROWS {
                write(&mut workspace.row_scratch, row, read(&squared, row, column));
            }
            lower_envelope(workspace, ROWS);
            for row in 0..ROWS {
                let value = read_line(&workspace.transformed, row);
                write_cell(&mut squared, row, column, value);
            }
        }

        // Squared cell units to metres, with an untouched line reading infinity.
        let resolution = geometry.resolution();
        let limit = unreachable::<T>();
        let mut distances = [[T::ZERO; COLUMNS]; ROWS];
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                let value = read(&squared, row, column);
                let metres = if value >= limit {
                    T::INFINITY
                } else {
                    value.sqrt() * resolution
                };
                write_cell(&mut distances, row, column, metres);
            }
        }

        Ok(DistanceField {
            distances,
            geometry,
        })
    }

    /// The grid's placement and index arithmetic.
    #[inline]
    #[must_use]
    pub fn geometry(&self) -> GridGeometry<T> {
        self.geometry
    }

    /// A cell's distance to the nearest blocked cell, or `None` off the grid.
    #[inline]
    #[must_use]
    pub fn distance_of(&self, row: usize, column: usize) -> Option<T> {
        self.distances
            .get(row)
            .and_then(|row_distances| row_distances.get(column))
            .copied()
    }

    /// The distance at a world point, bilinear over the four surrounding cell centres.
    ///
    /// `None` where any of the four falls outside, so the outer half-cell rim of the grid has no
    /// interpolated value. The blend is exactly differentiable by swapping `T` for an autodiff
    /// scalar, which is what a gradient-based planner needs of it.
    #[must_use]
    pub fn distance_at(&self, point: [T; 2]) -> Option<T> {
        let (row, column, row_fraction, column_fraction) = self.fractional_cell(point)?;
        let lower_left = self.distance_of(row, column)?;
        let lower_right = self.distance_of(row, column + 1)?;
        let upper_left = self.distance_of(row + 1, column)?;
        let upper_right = self.distance_of(row + 1, column + 1)?;

        let lower = lower_left + (lower_right - lower_left) * column_fraction;
        let upper = upper_left + (upper_right - upper_left) * column_fraction;
        Some(lower + (upper - lower) * row_fraction)
    }

    /// The field's gradient at a world point, by central differences over one cell.
    ///
    /// `None` where a probe falls outside the interpolatable region.
    #[must_use]
    pub fn gradient_at(&self, point: [T; 2]) -> Option<[T; 2]> {
        let step = self.geometry.resolution();
        let twice = step + step;
        let ahead_x = self.distance_at([point[0] + step, point[1]])?;
        let behind_x = self.distance_at([point[0] - step, point[1]])?;
        let ahead_y = self.distance_at([point[0], point[1] + step])?;
        let behind_y = self.distance_at([point[0], point[1] - step])?;
        Some([(ahead_x - behind_x) / twice, (ahead_y - behind_y) / twice])
    }

    /// The lower-left cell of the four surrounding `point`, and how far between them it sits.
    fn fractional_cell(&self, point: [T; 2]) -> Option<(usize, usize, T, T)> {
        if !point[0].is_finite() || !point[1].is_finite() {
            return None;
        }
        let origin = self.geometry.origin();
        let resolution = self.geometry.resolution();
        // Cell centres sit half a cell in from the corners.
        let column_axis = (point[0] - origin[0]) / resolution - T::HALF;
        let row_axis = (point[1] - origin[1]) / resolution - T::HALF;
        let column_floor = column_axis.floor();
        let row_floor = row_axis.floor();
        if column_floor.to_f64() < 0.0 || row_floor.to_f64() < 0.0 {
            return None;
        }
        Some((
            row_floor.to_f64() as usize,
            column_floor.to_f64() as usize,
            row_axis - row_floor,
            column_axis - column_floor,
        ))
    }
}

/// One cell of a grid, or the unreachable sentinel if the index is somehow off it.
fn read<const ROWS: usize, const COLUMNS: usize, T: Numeric>(
    grid: &[[T; COLUMNS]; ROWS],
    row: usize,
    column: usize,
) -> T {
    grid.get(row)
        .and_then(|row_cells| row_cells.get(column))
        .copied()
        .unwrap_or_else(unreachable)
}

fn write_cell<const ROWS: usize, const COLUMNS: usize, T: Numeric>(
    grid: &mut [[T; COLUMNS]; ROWS],
    row: usize,
    column: usize,
    value: T,
) {
    if let Some(cell) = grid
        .get_mut(row)
        .and_then(|row_cells| row_cells.get_mut(column))
    {
        *cell = value;
    }
}

fn read_line<const MAX_SPAN: usize, T: Numeric>(line: &[T; MAX_SPAN], index: usize) -> T {
    line.get(index).copied().unwrap_or_else(unreachable)
}

fn write<const MAX_SPAN: usize, T>(line: &mut [T; MAX_SPAN], index: usize, value: T) {
    if let Some(slot) = line.get_mut(index) {
        *slot = value;
    }
}

/// The one-dimensional distance transform of `row_scratch[..length]`, into `transformed[..length]`.
///
/// The lower envelope of the parabolas `f(q) + (x − q)²`: `vertices` holds which parabolas are on
/// the envelope and `intersections` where consecutive ones cross, which is why the workspace needs
/// one slot more than the span.
fn lower_envelope<const MAX_SPAN: usize, T: Numeric>(
    workspace: &mut DistanceTransformWorkspace<MAX_SPAN, T>,
    length: usize,
) {
    if length == 0 {
        return;
    }
    let mut envelope_top = 0usize;
    write(&mut workspace.vertices, 0, 0);
    write(&mut workspace.intersections, 0, T::NEG_INFINITY);
    write(&mut workspace.intersections, 1, T::INFINITY);

    for sample_index in 1..length {
        let sample_value = T::from_usize(sample_index);
        let at_sample = read_line(&workspace.row_scratch, sample_index);
        let mut crossing;
        loop {
            let vertex = workspace.vertices.get(envelope_top).copied().unwrap_or(0);
            let vertex_value = T::from_usize(vertex);
            let at_vertex = read_line(&workspace.row_scratch, vertex);
            crossing = ((at_sample + sample_value * sample_value)
                - (at_vertex + vertex_value * vertex_value))
                / (sample_value + sample_value - vertex_value - vertex_value);
            let boundary = workspace
                .intersections
                .get(envelope_top)
                .copied()
                .unwrap_or(T::NEG_INFINITY);
            // The `envelope_top == 0` arm is what keeps the pop from underflowing; the sentinel
            // at `intersections[0]` means it is never reached on well-formed input.
            if crossing > boundary || envelope_top == 0 {
                break;
            }
            envelope_top -= 1;
        }
        envelope_top += 1;
        write(&mut workspace.vertices, envelope_top, sample_index);
        write(&mut workspace.intersections, envelope_top, crossing);
        write(&mut workspace.intersections, envelope_top + 1, T::INFINITY);
    }

    let mut envelope_index = 0usize;
    for sample_index in 0..length {
        let sample_value = T::from_usize(sample_index);
        while workspace
            .intersections
            .get(envelope_index + 1)
            .copied()
            .unwrap_or(T::INFINITY)
            < sample_value
        {
            envelope_index += 1;
        }
        let vertex = workspace.vertices.get(envelope_index).copied().unwrap_or(0);
        let separation = sample_value - T::from_usize(vertex);
        let value = separation * separation + read_line(&workspace.row_scratch, vertex);
        write(&mut workspace.transformed, sample_index, value);
    }
}
