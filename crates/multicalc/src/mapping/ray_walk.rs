#![deny(clippy::indexing_slicing)]

//! The cells a beam passes through, by the Amanatides–Woo grid traversal.

use crate::mapping::grid_geometry::GridGeometry;
use crate::scalar::{Numeric, Primal};

/// One cell a beam entered, and how far along the beam it entered it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayStep<T: Numeric = f64> {
    /// Row of the cell entered.
    pub row: usize,
    /// Column of the cell entered.
    pub column: usize,
    /// Distance from the ray's start to where it entered this cell.
    pub entry_distance: T,
}

/// The cells a beam passes through, in order, stopping when it leaves the grid or runs past its
/// maximum range.
///
/// A slab test finds where the beam meets the grid, then each step crosses whichever axis boundary
/// comes first. A beam starting outside is walked from where it enters; one that never enters
/// yields nothing.
///
/// ```
/// use multicalc::mapping::GridGeometry;
///
/// // Straight along the middle row of a 5 by 5 grid of unit cells.
/// let geometry: GridGeometry = GridGeometry::try_new(5, 5, 1.0, [0.0, 0.0])?;
/// let middle_of_the_lowest_left_cell_of_that_row = [0.5, 2.5];
/// let along_the_row = 0.0;
/// let maximum_range = 10.0;
///
/// let walk = geometry.walk(middle_of_the_lowest_left_cell_of_that_row, along_the_row, maximum_range);
/// let expected: [(usize, usize); 5] = core::array::from_fn(|column| (2, column));
/// for (step, expected) in walk.zip(expected) {
///     assert_eq!((step.row, step.column), expected);
/// }
///
/// // Five cells across, and the first is entered at zero because the beam starts inside it.
/// let mut walk = geometry.walk(middle_of_the_lowest_left_cell_of_that_row, along_the_row, maximum_range);
/// assert!(walk.next().is_some_and(|step| step.entry_distance == 0.0));
/// assert_eq!(walk.count(), 4);
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RayWalk<T: Numeric + Primal = f64> {
    row: isize,
    column: isize,
    step_row: isize,
    step_column: isize,
    stride_row: T,
    stride_column: T,
    next_row: T,
    next_column: T,
    entered: T,
    maximum_range: T,
    rows: usize,
    columns: usize,
    finished: bool,
}

impl<T: Numeric + Primal> RayWalk<T> {
    /// The walk a beam from `start` on `bearing` makes across `geometry`.
    pub(crate) fn new(
        geometry: &GridGeometry<T>,
        start: [T; 2],
        bearing: T,
        maximum_range: T,
    ) -> Self {
        let empty = RayWalk {
            row: 0,
            column: 0,
            step_row: 1,
            step_column: 1,
            stride_row: T::INFINITY,
            stride_column: T::INFINITY,
            next_row: T::INFINITY,
            next_column: T::INFINITY,
            entered: T::ZERO,
            maximum_range,
            rows: geometry.rows(),
            columns: geometry.columns(),
            finished: true,
        };
        if geometry.rows() == 0 || geometry.columns() == 0 {
            return empty;
        }
        let direction = [bearing.cos(), bearing.sin()];

        // Where the beam meets the grid. A beam starting inside enters at zero.
        let Some(entry) = entry_distance(geometry, start, direction) else {
            return empty;
        };
        if entry > maximum_range {
            return empty;
        }
        let point = [
            start[0] + entry * direction[0],
            start[1] + entry * direction[1],
        ];

        // The cell the beam is in as it enters. Rounding at an edge can push the index just
        // outside, so it is forced back on.
        let lowest_corner = geometry.origin();
        let resolution = geometry.resolution();
        let column = clamp_index(
            (point[0] - lowest_corner[0]) / resolution,
            geometry.columns(),
        );
        let row = clamp_index((point[1] - lowest_corner[1]) / resolution, geometry.rows());

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
        let next_column = boundary_distance(
            start[0],
            direction[0],
            column,
            step_column,
            lowest_corner[0],
            resolution,
        );
        let next_row = boundary_distance(
            start[1],
            direction[1],
            row,
            step_row,
            lowest_corner[1],
            resolution,
        );

        RayWalk {
            row,
            column,
            step_row,
            step_column,
            stride_row,
            stride_column,
            next_row,
            next_column,
            entered: entry,
            maximum_range,
            rows: geometry.rows(),
            columns: geometry.columns(),
            finished: false,
        }
    }
}

impl<T: Numeric + Primal> Iterator for RayWalk<T> {
    type Item = RayStep<T>;

    fn next(&mut self) -> Option<RayStep<T>> {
        if self.finished || self.entered > self.maximum_range {
            return None;
        }
        if self.column < 0 || self.row < 0 {
            return None;
        }
        let (column, row) = (self.column as usize, self.row as usize);
        if column >= self.columns || row >= self.rows {
            return None;
        }
        let step = RayStep {
            row,
            column,
            entry_distance: self.entered,
        };
        // Cross whichever edge comes first.
        if self.next_column < self.next_row {
            self.column += self.step_column;
            self.entered = self.next_column;
            self.next_column += self.stride_column;
        } else {
            self.row += self.step_row;
            self.entered = self.next_row;
            self.next_row += self.stride_row;
        }
        Some(step)
    }
}

/// How far along the beam `start + distance · direction` the grid is entered, or `None` if it never
/// is. A beam already inside returns zero.
fn entry_distance<T: Numeric + Primal>(
    geometry: &GridGeometry<T>,
    start: [T; 2],
    direction: [T; 2],
) -> Option<T> {
    let lowest_corner = geometry.origin();
    let highest_corner = [
        lowest_corner[0] + T::from_usize(geometry.columns()) * geometry.resolution(),
        lowest_corner[1] + T::from_usize(geometry.rows()) * geometry.resolution(),
    ];
    let mut near = T::NEG_INFINITY;
    let mut far = T::INFINITY;
    for (start, step, low, high) in [
        (start[0], direction[0], lowest_corner[0], highest_corner[0]),
        (start[1], direction[1], lowest_corner[1], highest_corner[1]),
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
/// land outside the grid.
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
