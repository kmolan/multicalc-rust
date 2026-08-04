use multicalc::mapping::{DynamicOccupancyGrid, OccupancyMap};
use multicalc_demos::sim::localization_obstacle_avoidance_2d::lap_track_2d::lap_track_2d;
use std::f64::consts::FRAC_PI_2;

#[test]
fn the_corridor_is_nine_hundred_millimetres_wide() {
    let track = lap_track_2d().unwrap();
    // A clear point on the bottom-straight centreline (outer wall at y = 0, inner at y = 0.9).
    let centre = [2.3, 0.45];
    let cell = track.grid.resolution();
    let down = track
        .grid
        .cast_ray(centre, -FRAC_PI_2, 2.0)
        .expect("a wall below the centreline");
    let up = track
        .grid
        .cast_ray(centre, FRAC_PI_2, 2.0)
        .expect("a wall above the centreline");
    assert!(
        (down - 0.45).abs() <= cell + 1e-9,
        "distance down to the outer wall: {down}"
    );
    assert!(
        (up - 0.45).abs() <= cell + 1e-9,
        "distance up to the inner island: {up}"
    );
}

#[test]
fn the_start_pose_is_clear() {
    let track = lap_track_2d().unwrap();
    let start = [track.start_pose[0], track.start_pose[1]];
    let nearest = nearest_occupied_distance(&track.grid, start);
    assert!(
        nearest > 0.34,
        "the start pose has only {nearest} m to the nearest wall"
    );
}

#[test]
fn inside_slip_zone_matches_the_rectangle() {
    let track = lap_track_2d().unwrap();
    assert!(track.inside_slip_zone([2.3, 0.45]));
    assert!(!track.inside_slip_zone([1.0, 0.45]));
    assert!(!track.inside_slip_zone([2.3, 1.5]));
}

#[test]
fn every_passage_admits_the_chassis_with_margin() {
    // Each obstacle leaves a gap whose centre clears the chassis half-width (0.17 m) plus the
    // 0.06 m clearance bar, so a geometry defect fails here rather than as a contact deep in a run.
    let track = lap_track_2d().unwrap();
    let bar = 0.17 + 0.06;
    let passages = [
        ("box", [1.55, 0.51]),
        ("pillar", [3.0, 0.51]),
        ("gate", [5.55, 2.0]),
        ("chicane low gap", [2.35, 3.51]),
        ("chicane high gap", [3.15, 3.58]),
        ("slalom post one", [3.9, 3.58]),
        ("slalom post two", [4.4, 3.5]),
        ("angled barrier", [0.55, 1.30]),
    ];
    for (label, point) in passages {
        let clearance = nearest_occupied_distance(&track.grid, point);
        assert!(
            clearance > bar,
            "{label} passage too tight: {clearance:.3} m at {point:?}"
        );
    }
}

/// The distance from `point` to the centre of the nearest occupied cell, or infinity if the grid
/// is empty.
#[must_use]
fn nearest_occupied_distance(grid: &DynamicOccupancyGrid, point: [f64; 2]) -> f64 {
    let origin = grid.origin();
    let cell = grid.resolution();
    let mut nearest = f64::INFINITY;
    for row in 0..grid.rows() {
        for column in 0..grid.columns() {
            if grid.is_occupied(row, column) {
                let centre = [
                    origin[0] + (column as f64 + 0.5) * cell,
                    origin[1] + (row as f64 + 0.5) * cell,
                ];
                let distance = (centre[0] - point[0]).hypot(centre[1] - point[1]);
                nearest = nearest.min(distance);
            }
        }
    }
    nearest
}
