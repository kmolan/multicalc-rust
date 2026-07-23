//! The marked 2D lap track: a rounded-rectangle corridor rasterized into an occupancy grid, with a
//! wheel-slip patch and the start pose.

use std::f64::consts::FRAC_PI_2;

use super::occupancy_grid::OccupancyGrid;

/// Folds an angle into the range (-π, π].
#[must_use]
pub fn wrap_angle(angle: f64) -> f64 {
    use std::f64::consts::TAU;
    angle - TAU * (angle / TAU).round()
}

/// A closed rounded-rectangle outline, counter-clockwise. Returns `4 * segments_per_corner` points.
///
/// The straights carry no intermediate points because rasterizing joins consecutive points with a
/// wall; only the corners need sampling to trace their quarter-turns.
#[must_use]
pub fn rounded_rectangle(
    center: [f64; 2],
    half_extent: [f64; 2],
    corner_radius: f64,
    segments_per_corner: usize,
) -> Vec<[f64; 2]> {
    let corner_radius = corner_radius
        .min(half_extent[0])
        .min(half_extent[1])
        .max(0.0);
    let segments_per_corner = segments_per_corner.max(1);
    // Each corner: the centre of its arc (relative to `center`) and the angle the arc starts at.
    let corners = [
        (
            [
                half_extent[0] - corner_radius,
                -(half_extent[1] - corner_radius),
            ],
            -FRAC_PI_2,
        ),
        (
            [
                half_extent[0] - corner_radius,
                half_extent[1] - corner_radius,
            ],
            0.0,
        ),
        (
            [
                -(half_extent[0] - corner_radius),
                half_extent[1] - corner_radius,
            ],
            FRAC_PI_2,
        ),
        (
            [
                -(half_extent[0] - corner_radius),
                -(half_extent[1] - corner_radius),
            ],
            2.0 * FRAC_PI_2,
        ),
    ];
    let mut points = Vec::with_capacity(4 * segments_per_corner);
    for (offset, start_angle) in corners {
        for index in 0..segments_per_corner {
            let fraction = index as f64 / (segments_per_corner - 1).max(1) as f64;
            let angle = start_angle + FRAC_PI_2 * fraction;
            points.push([
                center[0] + offset[0] + corner_radius * angle.cos(),
                center[1] + offset[1] + corner_radius * angle.sin(),
            ]);
        }
    }
    points
}

/// A closed 2D lap track drawn into an occupancy grid, with the extra facts the run driver needs.
#[derive(Debug, Clone, PartialEq)]
pub struct LapTrack2D {
    /// The walls: the map the localizer matches against and the surface the lidar sees.
    pub grid: OccupancyGrid,
    /// Where the robot truly starts, as [x, y, heading].
    pub start_pose: [f64; 3],
    /// A rough first guess for the localizer: position near the truth, heading unknown.
    pub localization_hint: [f64; 3],
    /// The rectangle where a wheel slips, as [[x_min, y_min], [x_max, y_max]].
    pub slip_zone: [[f64; 2]; 2],
    /// The dead-end pocket, where the clearance bar is not applied.
    pub pocket_bounds: [[f64; 2]; 2],
    /// The middle of the inner island, used to count laps.
    pub island_center: [f64; 2],
}

impl LapTrack2D {
    /// Whether the point is in the wheel-slip rectangle.
    #[must_use]
    pub fn inside_slip_zone(&self, point: [f64; 2]) -> bool {
        inside(self.slip_zone, point)
    }

    /// Whether the point is in the dead-end pocket.
    #[must_use]
    pub fn inside_pocket(&self, point: [f64; 2]) -> bool {
        inside(self.pocket_bounds, point)
    }
}

/// The pinned 2D lap track: a rounded-rectangle corridor with the staged obstacle gauntlet.
///
/// The one grid is the prior map the localizer matches against and the surface the lidar casts
/// against, so the two can never disagree.
#[must_use]
pub fn lap_track_2d() -> LapTrack2D {
    // A grid spanning the outer boundary with a one-cell margin, 0.05 m cells, origin at the corner.
    let resolution = 0.05_f64;
    let columns = ((6.0 + 0.4) / resolution).ceil() as usize; // outer x-extent 0..6 plus margin
    let rows = ((4.0 + 0.4) / resolution).ceil() as usize;
    let mut grid = OccupancyGrid::new(columns, rows, resolution, [-0.2, -0.2]);

    // Boundary loops (concentric corners keep the corridor a clean 0.9 m).
    grid.occupy_polyline(&rounded_rectangle([3.0, 2.0], [3.0, 2.0], 1.2, 8), true);
    grid.occupy_polyline(&rounded_rectangle([3.0, 2.0], [2.1, 1.1], 0.3, 8), true);

    // Bottom straight: a box against the outer wall, then a one-way-past pillar. The slip patch
    // between them (x in [2.0, 2.6]) is left clear.
    grid.occupy_polyline(&box_outline([1.4, 0.0], [1.7, 0.30]), true);
    grid.occupy_circle([3.0, 0.20], 0.18);

    // Right straight (corridor x in [5.1, 6.0]): a gate leaving a 0.70 m slot centred at y = 2.0.
    grid.occupy_polyline(&box_outline([5.1, 1.85], [5.30, 2.15]), true);
    grid.occupy_polyline(&box_outline([5.80, 1.85], [6.0, 2.15]), true);

    // Top straight (corridor y in [3.1, 4.0]): a chicane forcing an S, then a two-post slalom.
    grid.occupy_polyline(&box_outline([2.15, 3.55], [2.55, 4.0]), true);
    grid.occupy_polyline(&box_outline([2.95, 3.1], [3.35, 3.55]), true);
    grid.occupy_circle([3.9, 3.35], 0.15);
    grid.occupy_circle([4.4, 3.75], 0.15);

    // Left straight (corridor x in [0.0, 0.9]): a slanted barrier, then a dead-end pocket in the wall.
    let barrier = box_outline([0.15, 1.15], [0.55, 1.35]);
    grid.occupy_polyline(&rotate_points(&barrier, [0.35, 1.25], 0.52), true); // ~30° slant
    stamp_alcove(&mut grid, [[-0.2, 1.7], [0.5, 2.3]]);

    LapTrack2D {
        grid,
        start_pose: [1.0, 0.45, 0.0],
        localization_hint: [1.0, 0.45, 0.0],
        slip_zone: [[2.0, 0.0], [2.6, 0.9]],
        pocket_bounds: [[-0.2, 1.6], [0.6, 2.4]],
        island_center: [3.0, 2.0],
    }
}

/// Whether the point sits inside the axis-aligned rectangle `[[x_min, y_min], [x_max, y_max]]`.
fn inside(bounds: [[f64; 2]; 2], point: [f64; 2]) -> bool {
    let [[x_min, y_min], [x_max, y_max]] = bounds;
    point[0] >= x_min && point[0] <= x_max && point[1] >= y_min && point[1] <= y_max
}

/// The four corners of an axis-aligned box, from opposite corners.
fn box_outline(min: [f64; 2], max: [f64; 2]) -> Vec<[f64; 2]> {
    vec![
        [min[0], min[1]],
        [max[0], min[1]],
        [max[0], max[1]],
        [min[0], max[1]],
    ]
}

/// Turns each point about `center` by `angle`, so a box can be laid down at a slant.
fn rotate_points(points: &[[f64; 2]], center: [f64; 2], angle: f64) -> Vec<[f64; 2]> {
    let (sin, cos) = angle.sin_cos();
    points
        .iter()
        .map(|p| {
            let (dx, dy) = (p[0] - center[0], p[1] - center[1]);
            [
                center[0] + cos * dx - sin * dy,
                center[1] + sin * dx + cos * dy,
            ]
        })
        .collect()
}

/// Marks three sides of a box, leaving the +x side open — a dead-end pocket in the outer wall.
fn stamp_alcove(grid: &mut OccupancyGrid, bounds: [[f64; 2]; 2]) {
    let [[x_min, y_min], [x_max, y_max]] = bounds;
    grid.occupy_polyline(
        &[
            [x_max, y_min],
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
        ],
        false,
    );
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn rounded_rectangle_has_four_arcs_of_points_within_the_extent() {
        let center = [3.0, 2.0];
        let half_extent = [3.0, 2.0];
        let points = rounded_rectangle(center, half_extent, 1.2, 8);
        assert_eq!(points.len(), 4 * 8);
        for point in &points {
            assert!(
                (point[0] - center[0]).abs() <= half_extent[0] + 1e-9,
                "point {point:?} outside the x-extent"
            );
            assert!(
                (point[1] - center[1]).abs() <= half_extent[1] + 1e-9,
                "point {point:?} outside the y-extent"
            );
        }
    }

    #[test]
    fn the_corridor_is_nine_hundred_millimetres_wide() {
        let track = lap_track_2d();
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
        let track = lap_track_2d();
        let start = [track.start_pose[0], track.start_pose[1]];
        let nearest = nearest_occupied_distance(&track.grid, start);
        assert!(
            nearest > 0.34,
            "the start pose has only {nearest} m to the nearest wall"
        );
    }

    #[test]
    fn inside_slip_zone_matches_the_rectangle() {
        let track = lap_track_2d();
        assert!(track.inside_slip_zone([2.3, 0.45]));
        assert!(!track.inside_slip_zone([1.0, 0.45]));
        assert!(!track.inside_slip_zone([2.3, 1.5]));
    }

    #[test]
    fn wrap_angle_folds_full_turns_away() {
        // Interior values pass through untouched.
        assert!((wrap_angle(0.5) - 0.5).abs() < 1e-12);
        assert!((wrap_angle(-1.2) + 1.2).abs() < 1e-12);
        // Whole turns are removed.
        assert!((wrap_angle(2.0 * PI + 0.3) - 0.3).abs() < 1e-12);
        assert!((wrap_angle(-2.0 * PI - 0.3) + 0.3).abs() < 1e-12);
        // An odd multiple of π lands on the ±π boundary.
        assert!((wrap_angle(3.0 * PI).abs() - PI).abs() < 1e-12);
        assert!((wrap_angle(-3.0 * PI).abs() - PI).abs() < 1e-12);
    }

    /// The distance from `point` to the centre of the nearest occupied cell, or infinity if the grid
    /// is empty.
    fn nearest_occupied_distance(grid: &OccupancyGrid, point: [f64; 2]) -> f64 {
        let origin = grid.origin();
        let cell = grid.resolution();
        let mut nearest = f64::INFINITY;
        for row in 0..grid.rows() {
            for column in 0..grid.columns() {
                if grid.is_occupied(column, row) {
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
}
