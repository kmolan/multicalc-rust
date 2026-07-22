//! An occupancy grid and ray casting across it.
//!
//! The world is a rectangular array of square cells, each free or occupied. A grid can be built in
//! code — mark cells by index, or drop a world point into the cell that holds it — or loaded from a
//! CSV of `0`s and `1`s. Casting a ray walks it cell by cell and reports the distance to the first
//! occupied cell it enters.

use std::fmt;
use std::fs;
use std::path::Path;

/// A rectangular grid of square cells, each free or occupied.
///
/// Cell `(column, row)` covers the world square with `x` in
/// `[origin.x + column·resolution, origin.x + (column + 1)·resolution]` and `y` likewise. Column 0
/// is the lowest `x`, row 0 the lowest `y`, so `origin` is the lower-left corner of cell `(0, 0)`.
#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyGrid {
    columns: usize,
    rows: usize,
    resolution: f64,
    origin: [f64; 2],
    cells: Vec<bool>,
}

/// What can go wrong while loading a grid from a CSV file.
#[derive(Debug)]
pub enum GridError {
    /// The file could not be read.
    Io(std::io::Error),
    /// A row had a different width than the first, at the given 1-based line number.
    Ragged { line: usize },
    /// A token was neither `0` nor `1`, at the given 1-based line number.
    BadToken { line: usize },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridError::Io(error) => write!(f, "could not read the grid file: {error}"),
            GridError::Ragged { line } => {
                write!(f, "row on line {line} has a different width than the first row")
            }
            GridError::BadToken { line } => {
                write!(f, "line {line} has a value that is not 0 or 1")
            }
        }
    }
}

impl std::error::Error for GridError {}

impl From<std::io::Error> for GridError {
    fn from(error: std::io::Error) -> Self {
        GridError::Io(error)
    }
}

impl OccupancyGrid {
    /// A grid of the given size with every cell free.
    #[must_use]
    pub fn new(columns: usize, rows: usize, resolution: f64, origin: [f64; 2]) -> Self {
        OccupancyGrid {
            columns,
            rows,
            resolution,
            origin,
            cells: vec![false; columns * rows],
        }
    }

    /// Marks a cell by its column and row. An index outside the grid does nothing.
    pub fn set_cell(&mut self, column: usize, row: usize, occupied: bool) {
        if column < self.columns && row < self.rows {
            self.cells[row * self.columns + column] = occupied;
        }
    }

    /// Marks the cell that holds `point` as occupied. A point outside the grid does nothing.
    pub fn occupy_point(&mut self, point: [f64; 2]) {
        if let Some((column, row)) = self.cell_of(point) {
            self.set_cell(column, row, true);
        }
    }

    /// The number of columns.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The cell edge length, in metres.
    #[must_use]
    pub fn resolution(&self) -> f64 {
        self.resolution
    }

    /// The world coordinate of the lower-left corner of cell `(0, 0)`.
    #[must_use]
    pub fn origin(&self) -> [f64; 2] {
        self.origin
    }

    /// Whether the cell at `(column, row)` is occupied. Cells outside the grid read as free.
    #[must_use]
    pub fn is_occupied(&self, column: usize, row: usize) -> bool {
        column < self.columns
            && row < self.rows
            && self.cells[row * self.columns + column]
    }

    /// Loads a grid from a CSV of `0`s and `1`s, with the given cell size and origin.
    ///
    /// Each line is one row of cells; tokens are separated by spaces and/or commas. The first line
    /// is the top of the grid (highest `y`) and the last is the bottom (the origin row), so the
    /// file reads the way the world looks on screen. Blank lines and lines starting with `#` are
    /// ignored. Every row must have the same width.
    pub fn from_csv(
        path: impl AsRef<Path>,
        resolution: f64,
        origin: [f64; 2],
    ) -> Result<Self, GridError> {
        let text = fs::read_to_string(path)?;

        // Parse each non-empty, non-comment line into a row of cells, remembering its source line
        // number for error messages.
        let mut top_down: Vec<(usize, Vec<bool>)> = Vec::new();
        for (index, raw) in text.lines().enumerate() {
            let line = index + 1;
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let mut row = Vec::new();
            for token in trimmed.split([',', ' ', '\t']).filter(|t| !t.is_empty()) {
                match token {
                    "0" => row.push(false),
                    "1" => row.push(true),
                    _ => return Err(GridError::BadToken { line }),
                }
            }
            top_down.push((line, row));
        }

        let columns = top_down.first().map_or(0, |(_, row)| row.len());
        let rows = top_down.len();
        let mut grid = OccupancyGrid::new(columns, rows, resolution, origin);

        // The file is top-down but row 0 is the bottom, so fill from the last file line upward.
        for (grid_row, (line, row)) in top_down.iter().rev().enumerate() {
            if row.len() != columns {
                return Err(GridError::Ragged { line: *line });
            }
            for (column, &occupied) in row.iter().enumerate() {
                grid.set_cell(column, grid_row, occupied);
            }
        }
        Ok(grid)
    }

    /// The cell holding `point`, or `None` if the point lies outside the grid.
    fn cell_of(&self, point: [f64; 2]) -> Option<(usize, usize)> {
        let column = ((point[0] - self.origin[0]) / self.resolution).floor();
        let row = ((point[1] - self.origin[1]) / self.resolution).floor();
        if column < 0.0 || row < 0.0 {
            return None;
        }
        let (column, row) = (column as usize, row as usize);
        (column < self.columns && row < self.rows).then_some((column, row))
    }

    /// The distance from `origin` to the first occupied cell along `bearing`, or `None` when no
    /// occupied cell is met within `maximum_range`.
    ///
    /// The ray is walked cell by cell (Amanatides–Woo grid traversal). The reported distance is the
    /// exact point where the ray crosses into the first occupied cell, so a ray meeting a wall whose
    /// face lines up with a cell boundary reads an exact distance rather than one rounded to a cell.
    #[must_use]
    pub fn cast_ray(&self, origin: [f64; 2], bearing: f64, maximum_range: f64) -> Option<f64> {
        if self.columns == 0 || self.rows == 0 {
            return None;
        }
        let direction = [bearing.cos(), bearing.sin()];

        // Find where the ray enters the grid box. A ray that starts inside enters at t = 0.
        let entry = self.entry_distance(origin, direction)?;
        if entry > maximum_range {
            return None;
        }
        let point = [
            origin[0] + entry * direction[0],
            origin[1] + entry * direction[1],
        ];

        // The cell the ray is in as it enters the grid. Floating error at a boundary can push the
        // index just outside, so clamp it back onto the grid.
        let mut column = clamp_index((point[0] - self.origin[0]) / self.resolution, self.columns);
        let mut row = clamp_index((point[1] - self.origin[1]) / self.resolution, self.rows);

        // The step direction on each axis and how far along the ray one full cell width is. A ray
        // parallel to an axis never crosses that axis's boundaries, so its step is infinite.
        let step_column = direction[0].signum() as isize;
        let step_row = direction[1].signum() as isize;
        let delta_column = if direction[0] == 0.0 {
            f64::INFINITY
        } else {
            self.resolution / direction[0].abs()
        };
        let delta_row = if direction[1] == 0.0 {
            f64::INFINITY
        } else {
            self.resolution / direction[1].abs()
        };

        // The ray parameter at which the next boundary on each axis is crossed.
        let mut next_column = self.boundary_distance(origin[0], direction[0], column, step_column, self.origin[0]);
        let mut next_row = self.boundary_distance(origin[1], direction[1], row, step_row, self.origin[1]);

        // The ray parameter at which the ray entered the current cell.
        let mut entered = entry;
        loop {
            if entered > maximum_range {
                return None;
            }
            if column < 0 || row < 0 || column as usize >= self.columns || row as usize >= self.rows
            {
                return None;
            }
            if self.is_occupied(column as usize, row as usize) {
                return Some(entered);
            }
            // Step across whichever axis boundary comes first.
            if next_column < next_row {
                column += step_column;
                entered = next_column;
                next_column += delta_column;
            } else {
                row += step_row;
                entered = next_row;
                next_row += delta_row;
            }
        }
    }

    /// The ray parameter at which `origin + t·direction` enters the grid box, or `None` if it never
    /// does. A ray already inside returns `0`.
    fn entry_distance(&self, origin: [f64; 2], direction: [f64; 2]) -> Option<f64> {
        let minimum = self.origin;
        let maximum = [
            self.origin[0] + self.columns as f64 * self.resolution,
            self.origin[1] + self.rows as f64 * self.resolution,
        ];
        let mut near = f64::NEG_INFINITY;
        let mut far = f64::INFINITY;
        for axis in 0..2 {
            if direction[axis] == 0.0 {
                // Parallel to this pair of faces: a hit is only possible from inside the slab.
                if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] {
                    return None;
                }
            } else {
                let inverse = 1.0 / direction[axis];
                let mut t1 = (minimum[axis] - origin[axis]) * inverse;
                let mut t2 = (maximum[axis] - origin[axis]) * inverse;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                near = near.max(t1);
                far = far.min(t2);
            }
        }
        if near > far || far < 0.0 {
            return None;
        }
        Some(near.max(0.0))
    }

    /// The ray parameter at which `origin_axis + t·direction_axis` reaches the far boundary of the
    /// current cell along one axis.
    fn boundary_distance(
        &self,
        origin_axis: f64,
        direction_axis: f64,
        index: isize,
        step: isize,
        grid_minimum: f64,
    ) -> f64 {
        if direction_axis == 0.0 {
            return f64::INFINITY;
        }
        let next_index = if step > 0 { index + 1 } else { index };
        let boundary = grid_minimum + next_index as f64 * self.resolution;
        (boundary - origin_axis) / direction_axis
    }
}

/// Floors `value` to a cell index, clamped to `[0, length)`, so floating error at a boundary cannot
/// land outside the grid.
fn clamp_index(value: f64, length: usize) -> isize {
    let floored = value.floor();
    if floored < 0.0 {
        0
    } else if floored as usize >= length {
        length as isize - 1
    } else {
        floored as isize
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use std::f64::consts::PI;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Writes `contents` to a fresh file under the system temp directory and returns its path. Each
    // call gets a unique name so tests running at once do not clash.
    fn temp_csv(contents: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "multicalc_grid_{}_{unique}.csv",
            std::process::id()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    // A grid with a single occupied column whose left face sits exactly at x = 2, spanning enough y
    // that the oblique rays below meet it away from any row boundary.
    fn wall() -> OccupancyGrid {
        let mut grid = OccupancyGrid::new(15, 12, 1.0, [-5.0, -5.25]);
        for row in 0..grid.rows() {
            grid.set_cell(7, row, true);
        }
        grid
    }

    #[test]
    fn ray_hits_a_wall_head_on_at_the_exact_face() {
        let hit = wall().cast_ray([0.0, 0.0], 0.0, 10.0).unwrap();
        assert!((hit - 2.0).abs() < 1e-12, "hit {hit}");
    }

    #[test]
    fn oblique_ray_hits_the_wall_at_the_exact_face() {
        // A ray at 45° reaches the face at x = 2 after travelling 2 / cos(45°).
        let hit = wall().cast_ray([0.0, 0.0], PI / 4.0, 10.0).unwrap();
        assert!((hit - 2.0 / (PI / 4.0).cos()).abs() < 1e-12, "hit {hit}");
    }

    #[test]
    fn ray_into_empty_space_reads_nothing() {
        let grid = OccupancyGrid::new(15, 12, 1.0, [-5.0, -5.25]);
        assert!(grid.cast_ray([0.0, 0.0], 0.0, 10.0).is_none());
    }

    #[test]
    fn ray_pointing_away_from_the_wall_misses() {
        assert!(wall().cast_ray([0.0, 0.0], PI, 10.0).is_none());
    }

    #[test]
    fn range_is_respected() {
        assert!(wall().cast_ray([0.0, 0.0], 0.0, 1.0).is_none());
        assert!(wall().cast_ray([0.0, 0.0], 0.0, 2.5).is_some());
    }

    #[test]
    fn a_ray_starting_in_an_occupied_cell_reads_zero() {
        let mut grid = OccupancyGrid::new(4, 4, 1.0, [0.0, 0.0]);
        grid.set_cell(1, 1, true);
        let hit = grid.cast_ray([1.5, 1.5], 0.0, 10.0).unwrap();
        assert!(hit.abs() < 1e-12, "hit {hit}");
    }

    #[test]
    fn occupy_point_marks_the_containing_cell() {
        let mut grid = OccupancyGrid::new(4, 4, 0.5, [-1.0, -1.0]);
        // (0.1, 0.1) sits in column 2, row 2 with a 0.5 m cell and origin (-1, -1).
        grid.occupy_point([0.1, 0.1]);
        assert!(grid.is_occupied(2, 2));
        assert!(!grid.is_occupied(1, 2));
        // A point outside the grid is ignored.
        grid.occupy_point([100.0, 100.0]);
    }

    #[test]
    fn csv_round_trips_with_the_top_line_at_the_highest_y() {
        // Top line is the highest y; only the top-left cell is occupied.
        let path = temp_csv("1 0 0\n0 0 0\n");
        let grid = OccupancyGrid::from_csv(&path, 1.0, [0.0, 0.0]).unwrap();
        assert_eq!(grid.columns(), 3);
        assert_eq!(grid.rows(), 2);
        // The top line became the top row (row 1, highest y), leftmost column.
        assert!(grid.is_occupied(0, 1));
        assert!(!grid.is_occupied(0, 0));
    }

    #[test]
    fn csv_accepts_commas_and_skips_comments() {
        let path = temp_csv("# a corridor\n1,1\n\n0,0\n");
        let grid = OccupancyGrid::from_csv(&path, 1.0, [0.0, 0.0]).unwrap();
        assert_eq!(grid.columns(), 2);
        assert_eq!(grid.rows(), 2);
    }

    #[test]
    fn a_ragged_csv_is_rejected() {
        let path = temp_csv("1 1 1\n1 1\n");
        assert!(matches!(
            OccupancyGrid::from_csv(&path, 1.0, [0.0, 0.0]),
            Err(GridError::Ragged { .. })
        ));
    }

    #[test]
    fn a_bad_token_is_rejected() {
        let path = temp_csv("1 2 0\n");
        assert!(matches!(
            OccupancyGrid::from_csv(&path, 1.0, [0.0, 0.0]),
            Err(GridError::BadToken { .. })
        ));
    }
}
