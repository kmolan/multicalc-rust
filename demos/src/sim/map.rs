//! Map primitives and closed-form ray casting.

/// A line segment between two points, in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub start: [f64; 2],
    pub end: [f64; 2],
}

/// A circular obstacle, in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: [f64; 2],
    pub radius: f64,
}

/// One obstacle in a map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Obstacle {
    Segment(Segment),
    Circle(Circle),
}

/// A hardcoded world: a list of obstacles a ray can hit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Map {
    obstacles: Vec<Obstacle>,
}

impl Map {
    /// An empty map.
    #[must_use]
    pub fn new() -> Self {
        Map {
            obstacles: Vec::new(),
        }
    }

    /// Adds a segment between two points.
    #[must_use]
    pub fn with_segment(mut self, start: [f64; 2], end: [f64; 2]) -> Self {
        self.obstacles
            .push(Obstacle::Segment(Segment { start, end }));
        self
    }

    /// Adds a circle.
    #[must_use]
    pub fn with_circle(mut self, center: [f64; 2], radius: f64) -> Self {
        self.obstacles
            .push(Obstacle::Circle(Circle { center, radius }));
        self
    }

    /// Adds a segment between each consecutive pair of points and one from the last back to the
    /// first, so a polygon boundary is one call. Fewer than two points adds nothing.
    #[must_use]
    pub fn with_closed_loop(mut self, points: &[[f64; 2]]) -> Self {
        if points.len() < 2 {
            return self;
        }
        for pair in points.windows(2) {
            if let [start, end] = pair {
                self = self.with_segment(*start, *end);
            }
        }
        if let (Some(last), Some(first)) = (points.last(), points.first()) {
            self = self.with_segment(*last, *first);
        }
        self
    }

    /// The obstacles in the map.
    #[must_use]
    pub fn obstacles(&self) -> &[Obstacle] {
        &self.obstacles
    }

    /// The distance from `origin` to the nearest obstacle along `bearing`, or `None` when nothing
    /// is hit within `maximum_range`.
    #[must_use]
    pub fn cast_ray(&self, origin: [f64; 2], bearing: f64, maximum_range: f64) -> Option<f64> {
        let direction = [bearing.cos(), bearing.sin()];
        let mut nearest: Option<f64> = None;
        for obstacle in &self.obstacles {
            let hit = match obstacle {
                Obstacle::Segment(segment) => ray_segment(origin, direction, segment),
                Obstacle::Circle(circle) => ray_circle(origin, direction, circle),
            };
            if let Some(distance) = hit
                && nearest.is_none_or(|current| distance < current)
            {
                nearest = Some(distance);
            }
        }
        nearest.filter(|&distance| distance <= maximum_range)
    }
}

/// Where the ray `origin + t·direction` meets a segment, with `direction` a unit vector.
///
/// Writing `cross(a, b) = a.x·b.y − a.y·b.x`, `s` for the segment vector and `q` for the offset
/// from the ray origin to the segment start, the ray parameter is `cross(q, s) / cross(direction,
/// s)` and the segment parameter is `cross(q, direction) / cross(direction, s)`. A hit needs a
/// non-parallel pair, a non-negative ray parameter, and a segment parameter within `[0, 1]`.
fn ray_segment(origin: [f64; 2], direction: [f64; 2], segment: &Segment) -> Option<f64> {
    let s = [
        segment.end[0] - segment.start[0],
        segment.end[1] - segment.start[1],
    ];
    let q = [segment.start[0] - origin[0], segment.start[1] - origin[1]];
    let denominator = direction[0] * s[1] - direction[1] * s[0];
    if denominator.abs() < 1e-12 {
        return None; // parallel
    }
    let t = (q[0] * s[1] - q[1] * s[0]) / denominator;
    let u = (q[0] * direction[1] - q[1] * direction[0]) / denominator;
    (t >= 0.0 && (0.0..=1.0).contains(&u)).then_some(t)
}

/// Where the ray `origin + t·direction` meets a circle, with `direction` a unit vector.
///
/// With `f` the offset from the circle centre to the ray origin, the unit direction makes the
/// quadratic coefficient one, so the roots are `−b ∓ √(b² − c)`. The near root is the entry point;
/// a negative near root with a non-negative far root means the ray started inside the circle.
fn ray_circle(origin: [f64; 2], direction: [f64; 2], circle: &Circle) -> Option<f64> {
    let f = [origin[0] - circle.center[0], origin[1] - circle.center[1]];
    let b = f[0] * direction[0] + f[1] * direction[1];
    let c = f[0] * f[0] + f[1] * f[1] - circle.radius * circle.radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let near = -b - root;
    let far = -b + root;
    if near >= 0.0 {
        Some(near)
    } else if far >= 0.0 {
        Some(far)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn ray_hits_axis_aligned_segment() {
        let map = Map::new().with_segment([2.0, -1.0], [2.0, 1.0]);
        let hit = map.cast_ray([0.0, 0.0], 0.0, 10.0).unwrap();
        assert!((hit - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ray_misses_segment_behind_it() {
        let map = Map::new().with_segment([2.0, -1.0], [2.0, 1.0]);
        assert!(map.cast_ray([0.0, 0.0], PI, 10.0).is_none());
    }

    #[test]
    fn ray_parallel_to_segment_misses() {
        let map = Map::new().with_segment([0.0, 1.0], [5.0, 1.0]);
        assert!(map.cast_ray([0.0, 0.0], 0.0, 10.0).is_none());
    }

    #[test]
    fn ray_misses_past_the_segment_end() {
        // The segment spans y in [1, 3], so a ray along y = 0 passes below it.
        let map = Map::new().with_segment([2.0, 1.0], [2.0, 3.0]);
        assert!(map.cast_ray([0.0, 0.0], 0.0, 10.0).is_none());
    }

    #[test]
    fn ray_hits_circle_at_the_near_intersection() {
        let map = Map::new().with_circle([3.0, 0.0], 1.0);
        let hit = map.cast_ray([0.0, 0.0], 0.0, 10.0).unwrap();
        assert!((hit - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ray_tangent_to_circle_grazes_it() {
        // The discriminant is zero here, so the square root loses precision and a looser
        // tolerance is correct rather than a fudge.
        let map = Map::new().with_circle([3.0, 1.0], 1.0);
        let hit = map.cast_ray([0.0, 0.0], 0.0, 10.0).unwrap();
        assert!((hit - 3.0).abs() < 1e-6, "hit {hit}");
    }

    #[test]
    fn ray_from_inside_circle_returns_the_far_intersection() {
        let map = Map::new().with_circle([0.0, 0.0], 1.0);
        let hit = map.cast_ray([0.0, 0.0], 0.0, 10.0).unwrap();
        assert!((hit - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nearest_of_several_obstacles_wins_and_range_is_respected() {
        let map = Map::new()
            .with_segment([2.0, -1.0], [2.0, 1.0])
            .with_circle([5.0, 0.0], 1.0)
            .with_segment([8.0, -1.0], [8.0, 1.0]);
        let hit = map.cast_ray([0.0, 0.0], 0.0, 10.0).unwrap();
        assert!((hit - 2.0).abs() < 1e-12);
        assert!(map.cast_ray([0.0, 0.0], 0.0, 1.0).is_none());
    }

    #[test]
    fn closed_loop_makes_a_polygon_boundary() {
        let map = Map::new().with_closed_loop(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert_eq!(map.obstacles().len(), 4);
        let hit = map.cast_ray([0.5, 0.5], 0.0, 10.0).unwrap();
        assert!((hit - 0.5).abs() < 1e-12);
    }
}
