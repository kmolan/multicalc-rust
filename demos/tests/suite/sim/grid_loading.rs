use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use multicalc::mapping::OccupancyMap;
use multicalc_demos::sim::grid_loading::{GridFileError, load_occupancy_grid_csv};

// Writes `contents` to a fresh file under the system temp directory and returns its path. Each
// call gets a unique name so tests running at once do not clash.
#[must_use]
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

#[test]
fn csv_round_trips_with_the_top_line_at_the_highest_y() {
    // Top line is the highest y; only the top-left cell is occupied.
    let path = temp_csv("1 0 0\n0 0 0\n");
    let grid = load_occupancy_grid_csv(&path, 1.0, [0.0, 0.0]).unwrap();
    assert_eq!(grid.columns(), 3);
    assert_eq!(grid.rows(), 2);
    // The top line became the top row (row 1, highest y), leftmost column.
    assert!(grid.is_occupied(1, 0));
    assert!(!grid.is_occupied(0, 0));
}

#[test]
fn csv_accepts_commas_and_skips_comments() {
    let path = temp_csv("# a corridor\n1,1\n\n0,0\n");
    let grid = load_occupancy_grid_csv(&path, 1.0, [0.0, 0.0]).unwrap();
    assert_eq!(grid.columns(), 2);
    assert_eq!(grid.rows(), 2);
}

#[test]
fn a_ragged_csv_is_rejected() {
    let path = temp_csv("1 1 1\n1 1\n");
    assert!(matches!(
        load_occupancy_grid_csv(&path, 1.0, [0.0, 0.0]),
        Err(GridFileError::Ragged { .. })
    ));
}

#[test]
fn a_bad_token_is_rejected() {
    let path = temp_csv("1 2 0\n");
    assert!(matches!(
        load_occupancy_grid_csv(&path, 1.0, [0.0, 0.0]),
        Err(GridFileError::BadToken { .. })
    ));
}

#[test]
fn an_empty_file_is_rejected_rather_than_making_a_grid_with_no_cells() {
    let path = temp_csv("# nothing but a comment\n");
    assert!(matches!(
        load_occupancy_grid_csv(&path, 1.0, [0.0, 0.0]),
        Err(GridFileError::Grid(_))
    ));
}
