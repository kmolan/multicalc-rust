#![deny(clippy::indexing_slicing)]

//! One bit per cell, sized at compile time.

use crate::error::MappingError;
use crate::mapping::grid_geometry::GridGeometry;
use crate::mapping::occupancy_grid::{MutableOccupancyMap, OccupancyMap};
use crate::scalar::{Numeric, Primal};

/// How many cells one word holds.
const CELLS_PER_WORD: usize = 32;

/// A map of `ROWS` by `COLUMNS` square cells, one bit each, sized at compile time.
///
/// `WORDS_PER_ROW` must be at least `COLUMNS.div_ceil(32)`, which
/// [`WORDS_NEEDED`](Self::WORDS_NEEDED) states and [`try_new`](Self::try_new) checks. It appears in
/// every user's type signature, so it is a parameter rather than a computed dimension: stable Rust
/// has no `generic_const_exprs`.
///
/// Memory is `ROWS · WORDS_PER_ROW · 4` bytes, so a 128 by 128 map is 2 KB against
/// [`DynamicOccupancyGrid`](crate::mapping::DynamicOccupancyGrid)'s 16 KB, and it needs no heap.
///
/// ```
/// use multicalc::mapping::{MutableOccupancyMap, OccupancyMap, OccupancyGrid};
///
/// // A 6.4 m square at 5 cm cells: 128 by 128, and 128 / 32 = 4 words to a row.
/// let cell_size = 0.05;
/// let mut room: OccupancyGrid<128, 128, 4> = OccupancyGrid::try_new(cell_size, [0.0, 0.0])?;
/// assert_eq!(room.words_per_row(), 4);
///
/// // 128 rows of 4 words of 4 bytes: 2 KB of cells.
/// assert_eq!(128 * room.words_per_row() * 4, 2048);
///
/// // A wall up the middle, and a beam fired at it from the left.
/// let wall = [[3.2, 1.0], [3.2, 5.0]];
/// let joined_up = false;
/// room.occupy_polyline(&wall, joined_up);
///
/// let standing_at = [1.0, 3.0];
/// let east = 0.0;
/// let maximum_range = 6.0;
/// let distance = room.cast_ray(standing_at, east, maximum_range);
/// assert!(distance.is_some_and(|met| (met - 2.2).abs() <= cell_size));
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OccupancyGrid<
    const ROWS: usize,
    const COLUMNS: usize,
    const WORDS_PER_ROW: usize,
    T: Numeric + Primal = f64,
> {
    words: [[u32; WORDS_PER_ROW]; ROWS],
    geometry: GridGeometry<T>,
}

impl<const ROWS: usize, const COLUMNS: usize, const WORDS_PER_ROW: usize, T: Numeric + Primal>
    OccupancyGrid<ROWS, COLUMNS, WORDS_PER_ROW, T>
{
    /// The smallest `WORDS_PER_ROW` that holds `COLUMNS` bits: `COLUMNS.div_ceil(32)`.
    pub const WORDS_NEEDED: usize = COLUMNS.div_ceil(CELLS_PER_WORD);

    /// A map with every cell free.
    ///
    /// `resolution` is the edge length of one cell and `origin` is the world corner of cell
    /// `(0, 0)`, its lowest `x` and lowest `y`.
    ///
    /// Returns [`MappingError::WordsPerRowTooSmall`] when `WORDS_PER_ROW` is below
    /// [`WORDS_NEEDED`](Self::WORDS_NEEDED), and otherwise whatever
    /// [`GridGeometry::try_new`] rejects the placement with.
    pub fn try_new(resolution: T, origin: [T; 2]) -> Result<Self, MappingError> {
        if WORDS_PER_ROW < Self::WORDS_NEEDED {
            return Err(MappingError::WordsPerRowTooSmall);
        }
        Ok(OccupancyGrid {
            words: [[0; WORDS_PER_ROW]; ROWS],
            geometry: GridGeometry::try_new(ROWS, COLUMNS, resolution, origin)?,
        })
    }

    /// How many words each row of cells is packed into.
    #[inline]
    #[must_use]
    pub fn words_per_row(&self) -> usize {
        WORDS_PER_ROW
    }

    /// The grid's placement and index arithmetic.
    #[inline]
    #[must_use]
    pub fn geometry(&self) -> GridGeometry<T> {
        self.geometry
    }
}

impl<const ROWS: usize, const COLUMNS: usize, const WORDS_PER_ROW: usize, T: Numeric + Primal>
    OccupancyMap<T> for OccupancyGrid<ROWS, COLUMNS, WORDS_PER_ROW, T>
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
        if column >= COLUMNS {
            return false;
        }
        self.words
            .get(row)
            .and_then(|row_words| row_words.get(column / CELLS_PER_WORD))
            .is_some_and(|word| (word >> (column % CELLS_PER_WORD)) & 1 == 1)
    }

    fn geometry(&self) -> GridGeometry<T> {
        self.geometry
    }
}

impl<const ROWS: usize, const COLUMNS: usize, const WORDS_PER_ROW: usize, T: Numeric + Primal>
    MutableOccupancyMap<T> for OccupancyGrid<ROWS, COLUMNS, WORDS_PER_ROW, T>
{
    fn set_cell(&mut self, row: usize, column: usize, occupied: bool) {
        if column >= COLUMNS {
            return;
        }
        let bit = 1u32 << (column % CELLS_PER_WORD);
        if let Some(word) = self
            .words
            .get_mut(row)
            .and_then(|row_words| row_words.get_mut(column / CELLS_PER_WORD))
        {
            if occupied {
                *word |= bit;
            } else {
                *word &= !bit;
            }
        }
    }

    fn clear(&mut self) {
        self.words = [[0; WORDS_PER_ROW]; ROWS];
    }
}
