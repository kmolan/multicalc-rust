//! Reading an occupancy grid out of a CSV of `0`s and `1`s.

use std::fmt;
use std::path::Path;

use multicalc::error::MappingError;
use multicalc::mapping::{DynamicOccupancyGrid, MutableOccupancyMap};

/// What can go wrong while reading a grid file.
#[derive(Debug)]
pub enum GridFileError {
    /// The file could not be read.
    IoError(std::io::Error),
    /// A row had a different width than the first, at the given 1-based line number.
    Ragged { line: usize },
    /// A value was neither `0` nor `1`, at the given 1-based line number.
    BadToken { line: usize },
    /// The file described a grid the mapping module will not build.
    Grid(MappingError),
}

impl fmt::Display for GridFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridFileError::IoError(error) => write!(f, "could not read the grid file: {error}"),
            GridFileError::Ragged { line } => {
                write!(
                    f,
                    "row on line {line} has a different width than the first row"
                )
            }
            GridFileError::BadToken { line } => {
                write!(f, "line {line} has a value that is not 0 or 1")
            }
            GridFileError::Grid(error) => {
                write!(f, "the file does not describe a usable grid: {error}")
            }
        }
    }
}

impl std::error::Error for GridFileError {}

impl From<std::io::Error> for GridFileError {
    fn from(error: std::io::Error) -> Self {
        GridFileError::IoError(error)
    }
}

impl From<MappingError> for GridFileError {
    fn from(error: MappingError) -> Self {
        GridFileError::Grid(error)
    }
}

/// Loads a grid from a CSV of `0`s and `1`s, with the given cell size and origin.
///
/// Each line is one row of cells; values are separated by spaces and/or commas. The first line is
/// the top of the grid (highest `y`) and the last is the bottom (the origin row), so the file reads
/// the way the world looks on screen. Blank lines and lines starting with `#` are skipped. Every row
/// must have the same width.
pub fn load_occupancy_grid_csv(
    path: impl AsRef<Path>,
    resolution: f64,
    origin: [f64; 2],
) -> Result<DynamicOccupancyGrid, GridFileError> {
    let text = std::fs::read_to_string(path)?;

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
        for token in trimmed
            .split([',', ' ', '\t'])
            .filter(|tok| !tok.is_empty())
        {
            match token {
                "0" => row.push(false),
                "1" => row.push(true),
                _ => return Err(GridFileError::BadToken { line }),
            }
        }
        top_down.push((line, row));
    }

    let columns = top_down.first().map_or(0, |(_, row)| row.len());
    let rows = top_down.len();
    let mut grid = DynamicOccupancyGrid::try_new(columns, rows, resolution, origin)?;

    // The file is top-down but row 0 is the bottom, so fill from the last file line upward.
    for (grid_row, (line, row)) in top_down.iter().rev().enumerate() {
        if row.len() != columns {
            return Err(GridFileError::Ragged { line: *line });
        }
        for (column, &occupied) in row.iter().enumerate() {
            grid.set_cell(grid_row, column, occupied);
        }
    }
    Ok(grid)
}
