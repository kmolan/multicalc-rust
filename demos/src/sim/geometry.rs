//! Point lists tracing 2D shapes.

use std::f64::consts::{FRAC_PI_2, TAU};

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

/// A closed circle outline, for drawing a footprint or an uncertainty ellipse.
///
/// The first point is repeated at the end, so the list closes on itself.
#[must_use]
pub fn circle_outline(center: [f64; 2], radius: f64, segments: usize) -> Vec<[f64; 2]> {
    let segments = segments.max(1);
    (0..=segments)
        .map(|index| {
            let angle = TAU * index as f64 / segments as f64;
            [
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin(),
            ]
        })
        .collect()
}

/// The four corners of an axis-aligned box, from opposite corners.
#[must_use]
pub fn box_outline(min: [f64; 2], max: [f64; 2]) -> Vec<[f64; 2]> {
    vec![
        [min[0], min[1]],
        [max[0], min[1]],
        [max[0], max[1]],
        [min[0], max[1]],
    ]
}

/// Turns each point about `center` by `angle`, so a shape can be laid down at a slant.
#[must_use]
pub fn rotate_points(points: &[[f64; 2]], center: [f64; 2], angle: f64) -> Vec<[f64; 2]> {
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
