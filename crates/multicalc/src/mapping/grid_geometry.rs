#![deny(clippy::indexing_slicing)]

//! Where a grid sits in the world, and the index arithmetic every grid in the module shares.

use crate::error::MappingError;
use crate::mapping::ray_walk::RayWalk;
use crate::scalar::{Numeric, Primal};

/// The placement of a grid of square cells, and the world-to-cell arithmetic over it.
///
/// Cell `(row, column)` covers the world square starting at
/// `origin + [column · resolution, row · resolution]`, so `origin` is the lowest corner of cell
/// `(0, 0)`: row `0` is the lowest `y` and column `0` the lowest `x`. Cells flatten row-major, so
/// index `row · columns + column`.
///
/// ```
/// use multicalc::mapping::GridGeometry;
///
/// // A 4 by 4 patch of half-metre cells whose lowest corner sits at (-1, -1).
/// let geometry: GridGeometry = GridGeometry::try_new(4, 4, 0.5, [-1.0, -1.0])?;
///
/// assert_eq!(geometry.cell_of([-0.6, -0.9]), Some((0, 0)));
/// assert_eq!(geometry.center_of(0, 0), Some([-0.75, -0.75]));
/// assert_eq!(geometry.index_of(1, 2), Some(6));
/// assert_eq!(geometry.cell_at(6), Some((1, 2)));
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridGeometry<T: Numeric + Primal = f64> {
    origin: [T; 2],
    resolution: T,
    rows: usize,
    columns: usize,
}

impl<T: Numeric + Primal> GridGeometry<T> {
    /// A grid of `rows` by `columns` cells of edge length `resolution`, with cell `(0, 0)`'s lowest
    /// corner at `origin`.
    ///
    /// Returns [`MappingError::NonFinite`] if the cell size or either origin coordinate is infinite
    /// or NaN, [`MappingError::NonPositiveResolution`] if the cell size is zero or negative,
    /// [`MappingError::EmptyGrid`] with no rows or no columns, and [`MappingError::GridTooLarge`]
    /// when the two multiplied out are more cells than an index can name.
    pub fn try_new(
        rows: usize,
        columns: usize,
        resolution: T,
        origin: [T; 2],
    ) -> Result<Self, MappingError> {
        if !resolution.is_finite() || !origin[0].is_finite() || !origin[1].is_finite() {
            return Err(MappingError::NonFinite);
        }
        if resolution <= T::ZERO {
            return Err(MappingError::NonPositiveResolution);
        }
        if rows == 0 || columns == 0 {
            return Err(MappingError::EmptyGrid);
        }
        let count = rows
            .checked_mul(columns)
            .ok_or(MappingError::GridTooLarge)?;
        if count >= u32::MAX as usize {
            return Err(MappingError::GridTooLarge);
        }
        Ok(GridGeometry {
            origin,
            resolution,
            rows,
            columns,
        })
    }

    /// The four parts, unvalidated, for the [`OccupancyMap::geometry`] default body.
    ///
    /// A degenerate geometry built this way answers `None` to every query rather than panicking.
    ///
    /// [`OccupancyMap::geometry`]: crate::mapping::OccupancyMap::geometry
    pub(crate) fn from_parts(rows: usize, columns: usize, resolution: T, origin: [T; 2]) -> Self {
        GridGeometry {
            origin,
            resolution,
            rows,
            columns,
        }
    }

    /// How many cells up.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// How many cells across.
    #[inline]
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The edge length of one cell.
    #[inline]
    #[must_use]
    pub fn resolution(&self) -> T {
        self.resolution
    }

    /// The world corner of cell `(0, 0)`: its lowest `x` and lowest `y`.
    #[inline]
    #[must_use]
    pub fn origin(&self) -> [T; 2] {
        self.origin
    }

    /// `rows · columns`, saturating.
    #[inline]
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.rows.saturating_mul(self.columns)
    }

    /// Whether the pair names a cell on the grid.
    #[inline]
    #[must_use]
    pub fn contains(&self, row: usize, column: usize) -> bool {
        row < self.rows && column < self.columns
    }

    /// The cell holding `point`, as `(row, column)`, or `None` when the point lies outside.
    ///
    /// A point sitting exactly on the edge between two cells belongs to the higher one, so a cell
    /// holds its lower edge and not its upper.
    #[must_use]
    pub fn cell_of(&self, point: [T; 2]) -> Option<(usize, usize)> {
        let column = ((point[0] - self.origin[0]) / self.resolution)
            .floor()
            .to_f64();
        let row = ((point[1] - self.origin[1]) / self.resolution)
            .floor()
            .to_f64();
        if column < 0.0 || row < 0.0 {
            return None;
        }
        let (row, column) = (row as usize, column as usize);
        self.contains(row, column).then_some((row, column))
    }

    /// The world point at the middle of a cell:
    /// `origin + [(column + ½)·resolution, (row + ½)·resolution]`.
    #[must_use]
    pub fn center_of(&self, row: usize, column: usize) -> Option<[T; 2]> {
        self.contains(row, column).then(|| {
            [
                self.origin[0] + (T::from_usize(column) + T::HALF) * self.resolution,
                self.origin[1] + (T::from_usize(row) + T::HALF) * self.resolution,
            ]
        })
    }

    /// The row-major flattening `row · columns + column`, or `None` off the grid.
    #[inline]
    #[must_use]
    pub fn index_of(&self, row: usize, column: usize) -> Option<usize> {
        self.contains(row, column)
            .then(|| row * self.columns + column)
    }

    /// The inverse of [`index_of`](Self::index_of).
    #[inline]
    #[must_use]
    pub fn cell_at(&self, index: usize) -> Option<(usize, usize)> {
        (index < self.cell_count() && self.columns > 0)
            .then(|| (index / self.columns, index % self.columns))
    }

    /// The cells a beam from `start` on `bearing` passes through, in order.
    ///
    /// `bearing` is measured from the direction of rising `x`, turning toward rising `y`. A beam
    /// starting outside is walked from where it enters; one that never enters yields nothing.
    #[must_use]
    pub fn walk(&self, start: [T; 2], bearing: T, maximum_range: T) -> RayWalk<T> {
        RayWalk::new(self, start, bearing, maximum_range)
    }
}
