//! The marked 2D lap track: a rounded-rectangle corridor rasterized into an occupancy grid, with a
//! wheel-slip patch and the start pose.

use crate::sim::geometry::{box_outline, rotate_points, rounded_rectangle};
use crate::sim::occupancy_grid::OccupancyGrid;

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

    // Every obstacle leaves a 0.70 m passage in the 0.9 m corridor: chassis 0.34 m plus the 0.06 m
    // clearance bar each side needs 0.46 m, so 0.70 m clears it with margin.

    // Bottom straight (corridor y in [0.0, 0.9]): a box against the outer wall, then a one-way-past
    // pillar low against the wall. The slip patch between them (x in [2.0, 2.6]) is left clear.
    grid.occupy_polyline(&box_outline([1.4, 0.0], [1.7, 0.12]), true); // 0.78 m above
    grid.occupy_circle([3.0, -0.06], 0.18); // top at 0.12: reject the low side, take the 0.78 m above

    // Right straight (corridor x in [5.1, 6.0]): a gate leaving a 0.80 m slot centred at x = 5.55.
    grid.occupy_polyline(&box_outline([5.1, 1.85], [5.15, 2.15]), true);
    grid.occupy_polyline(&box_outline([5.95, 1.85], [6.0, 2.15]), true);

    // Top straight (corridor y in [3.1, 4.0]): a chicane, then two staggered slalom posts, each
    // leaving about 0.80 m past it.
    grid.occupy_polyline(&box_outline([2.15, 3.93], [2.55, 4.0]), true); // from the top wall
    grid.occupy_polyline(&box_outline([2.95, 3.1], [3.35, 3.17]), true); // from the bottom wall
    grid.occupy_circle([3.9, 3.0], 0.15); // low post: 0.85 m above
    grid.occupy_circle([4.4, 4.05], 0.15); // high post: 0.80 m below

    // Left straight (corridor x in [0.0, 0.9]): a slanted barrier near the outer wall (0.70 m beside
    // it), then a dead-end pocket cut into the outer wall.
    let barrier = box_outline([0.0, 1.15], [0.15, 1.45]);
    grid.occupy_polyline(&rotate_points(&barrier, [0.075, 1.30], 0.45), true); // ~26° slant
    // The pocket is a cavity in the outer wall (mouth flush at x = 0), so it never narrows the
    // corridor; it is a false opening the reactive planner can be tempted into.
    stamp_alcove(&mut grid, [[-0.2, 1.7], [0.0, 2.3]]);

    LapTrack2D {
        grid,
        start_pose: [1.0, 0.45, 0.0],
        localization_hint: [1.0, 0.45, 0.0],
        slip_zone: [[2.0, 0.0], [2.6, 0.9]],
        pocket_bounds: [[-0.3, 1.6], [0.25, 2.4]],
        island_center: [3.0, 2.0],
    }
}

/// Whether the point sits inside the axis-aligned rectangle `[[x_min, y_min], [x_max, y_max]]`.
fn inside(bounds: [[f64; 2]; 2], point: [f64; 2]) -> bool {
    let [[x_min, y_min], [x_max, y_max]] = bounds;
    point[0] >= x_min && point[0] <= x_max && point[1] >= y_min && point[1] <= y_max
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
