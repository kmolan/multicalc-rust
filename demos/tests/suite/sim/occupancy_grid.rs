use multicalc_demos::sim::occupancy_grid::{GridError, OccupancyGrid};
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
fn occupy_polyline_draws_a_gap_free_wall() {
    let mut grid = OccupancyGrid::new(20, 20, 0.1, [0.0, 0.0]);
    // A vertical wall at x = 1.0; a ray crossing it head-on is stopped at the wall.
    grid.occupy_polyline(&[[1.0, 0.2], [1.0, 1.5]], false);
    let hit = grid.cast_ray([0.0, 0.8], 0.0, 5.0).unwrap();
    assert!((hit - 1.0).abs() < grid.resolution() + 1e-9, "hit {hit}");
}

#[test]
fn occupy_circle_marks_the_rim() {
    let mut grid = OccupancyGrid::new(40, 40, 0.05, [0.0, 0.0]);
    grid.occupy_circle([1.0, 1.0], 0.3);
    // A ray through the centre meets the near rim at about radius 0.3, so 0.7 m away.
    let hit = grid.cast_ray([0.0, 1.0], 0.0, 5.0).unwrap();
    assert!(hit > 0.6 && hit < 0.72, "hit {hit}");
    // Only the rim is marked: the cell at the centre stays free.
    assert!(!grid.is_occupied(20, 20));
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
