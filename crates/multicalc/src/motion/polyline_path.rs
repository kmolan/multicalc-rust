//! A polyline path: an ordered set of waypoints joined by straight segments.
#![deny(clippy::indexing_slicing)]

use crate::error::MotionError;
use crate::linear_algebra::Vector;
use crate::scalar::Numeric;

/// What a lookahead query does once it runs past the end of the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndOfPath {
    /// Clamp to the final waypoint.
    Stop,
    /// Wrap around to the start.
    Loop,
}

/// The result of projecting a query point onto a path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathProjection<const DIMENSION: usize, T: Numeric> {
    point: Vector<DIMENSION, T>,
    segment_index: usize,
    arc_length: T,
    distance: T,
}

impl<const DIMENSION: usize, T: Numeric> PathProjection<DIMENSION, T> {
    /// The closest point found on the path.
    #[inline]
    pub fn point(&self) -> Vector<DIMENSION, T> {
        self.point
    }

    /// The index of the segment that contains the point (0 for a single-point path).
    #[inline]
    #[must_use]
    pub fn segment_index(&self) -> usize {
        self.segment_index
    }

    /// The arc length from the start of the path to the point.
    #[inline]
    #[must_use]
    pub fn arc_length(&self) -> T {
        self.arc_length
    }

    /// The distance from the query point to the point on the path.
    #[inline]
    #[must_use]
    pub fn distance(&self) -> T {
        self.distance
    }
}

/// A capacity-and-length waypoint path in `DIMENSION`-dimensional space.
///
/// Storage is a fixed array of `MAX_POINTS` waypoints with a runtime length, so the path is
/// stack-allocated and needs no heap. Duplicate consecutive waypoints are accepted; every query
/// treats a zero-length segment as contributing no arc length.
///
/// ```
/// use multicalc::motion::PolylinePath;
/// use multicalc::linear_algebra::Vector;
///
/// // An L-shaped path: three units east, then four units north.
/// let path: PolylinePath<3, 2, f64> = PolylinePath::try_from_points(&[
///     Vector::new([0.0, 0.0]),
///     Vector::new([3.0, 0.0]),
///     Vector::new([3.0, 4.0]),
/// ])
/// .unwrap();
/// assert!((path.total_arc_length() - 7.0).abs() < 1e-12);
///
/// // Two units along from the start sits on the first leg.
/// let [x, y] = path.lookahead_point(0.0, 2.0).unwrap().into_array();
/// assert!((x - 2.0).abs() < 1e-12 && y.abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolylinePath<const MAX_POINTS: usize, const DIMENSION: usize, T: Numeric> {
    points: [Vector<DIMENSION, T>; MAX_POINTS],
    length: usize,
    end_of_path: EndOfPath,
}

impl<const MAX_POINTS: usize, const DIMENSION: usize, T: Numeric> Default
    for PolylinePath<MAX_POINTS, DIMENSION, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_POINTS: usize, const DIMENSION: usize, T: Numeric>
    PolylinePath<MAX_POINTS, DIMENSION, T>
{
    /// An empty path that stops at its end.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            points: [Vector::zeros(); MAX_POINTS],
            length: 0,
            end_of_path: EndOfPath::Stop,
        }
    }

    /// Builds a path from a slice of waypoints.
    ///
    /// Returns [`MotionError::CapacityExceeded`] if more than `MAX_POINTS` waypoints are supplied, or
    /// [`MotionError::NonFinite`] if any coordinate is not finite.
    pub fn try_from_points(points: &[Vector<DIMENSION, T>]) -> Result<Self, MotionError> {
        if points.len() > MAX_POINTS {
            return Err(MotionError::CapacityExceeded);
        }
        if points.iter().any(|point| !point.is_finite()) {
            return Err(MotionError::NonFinite);
        }
        let mut path = Self::new();
        for (slot, point) in path.points.iter_mut().zip(points.iter()) {
            *slot = *point;
        }
        path.length = points.len();
        Ok(path)
    }

    /// Appends a waypoint.
    ///
    /// Returns [`MotionError::CapacityExceeded`] if the path is already full, or
    /// [`MotionError::NonFinite`] if any coordinate is not finite.
    pub fn push(&mut self, point: Vector<DIMENSION, T>) -> Result<(), MotionError> {
        if self.length == MAX_POINTS {
            return Err(MotionError::CapacityExceeded);
        }
        if !point.is_finite() {
            return Err(MotionError::NonFinite);
        }
        match self.points.get_mut(self.length) {
            Some(slot) => {
                *slot = point;
                self.length += 1;
                Ok(())
            }
            None => Err(MotionError::CapacityExceeded),
        }
    }

    /// Sets the end-of-path behaviour.
    #[inline]
    #[must_use]
    pub fn with_end_of_path(mut self, mode: EndOfPath) -> Self {
        self.end_of_path = mode;
        self
    }

    /// The number of waypoints.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether the path has no waypoints.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The waypoints as a slice.
    #[inline]
    pub fn waypoints(&self) -> &[Vector<DIMENSION, T>] {
        self.points.get(..self.length).unwrap_or(&[])
    }

    /// The total arc length along the path, zero for a path with fewer than two waypoints.
    #[must_use]
    pub fn total_arc_length(&self) -> T {
        let mut total = T::ZERO;
        for window in self.waypoints().windows(2) {
            if let [a, b] = window {
                total += (*b - *a).norm();
            }
        }
        total
    }

    /// The closest point on the path to a query point.
    ///
    /// Returns [`MotionError::PathTooShort`] if the path has no waypoints.
    pub fn closest_point(
        &self,
        query: Vector<DIMENSION, T>,
    ) -> Result<PathProjection<DIMENSION, T>, MotionError> {
        let waypoints = self.waypoints();
        let first = match waypoints.first() {
            Some(point) => *point,
            None => return Err(MotionError::PathTooShort),
        };
        if self.length == 1 {
            return Ok(PathProjection {
                point: first,
                segment_index: 0,
                arc_length: T::ZERO,
                distance: (query - first).norm(),
            });
        }

        let mut best: Option<PathProjection<DIMENSION, T>> = None;
        let mut arc_at_start = T::ZERO;
        for (segment_index, window) in waypoints.windows(2).enumerate() {
            let (a, b) = match window {
                [a, b] => (*a, *b),
                _ => continue,
            };
            let direction = b - a;
            let denominator = direction.norm_squared();
            let segment_length = direction.norm();
            let (candidate, parameter) = if denominator == T::ZERO {
                (a, T::ZERO)
            } else {
                let parameter = ((query - a).dot(direction) / denominator)
                    .max(T::ZERO)
                    .min(T::ONE);
                (a + direction.scale(parameter), parameter)
            };
            let distance = (query - candidate).norm();
            let improved = match &best {
                Some(current) => distance < current.distance,
                None => true,
            };
            if improved {
                best = Some(PathProjection {
                    point: candidate,
                    segment_index,
                    arc_length: arc_at_start + segment_length * parameter,
                    distance,
                });
            }
            arc_at_start += segment_length;
        }
        best.ok_or(MotionError::PathTooShort)
    }

    /// The point a given arc length ahead of a starting arc length along the path.
    ///
    /// The end-of-path mode decides what happens once the target runs past the end: [`EndOfPath::Stop`]
    /// clamps to the last waypoint and [`EndOfPath::Loop`] wraps around. Returns
    /// [`MotionError::PathTooShort`] if the path has no waypoints.
    pub fn lookahead_point(
        &self,
        from_arc_length: T,
        lookahead: T,
    ) -> Result<Vector<DIMENSION, T>, MotionError> {
        let waypoints = self.waypoints();
        let first = match waypoints.first() {
            Some(point) => *point,
            None => return Err(MotionError::PathTooShort),
        };
        if self.length == 1 {
            return Ok(first);
        }
        let last = waypoints.last().copied().unwrap_or(first);

        let total = self.total_arc_length();
        let mut target = from_arc_length + lookahead;
        match self.end_of_path {
            EndOfPath::Stop => {
                if target >= total {
                    return Ok(last);
                }
            }
            EndOfPath::Loop => {
                if total > T::ZERO {
                    target = target - total * (target / total).floor();
                } else {
                    return Ok(first);
                }
            }
        }

        let mut arc_at_start = T::ZERO;
        for window in waypoints.windows(2) {
            let (a, b) = match window {
                [a, b] => (*a, *b),
                _ => continue,
            };
            let direction = b - a;
            let segment_length = direction.norm();
            if segment_length == T::ZERO {
                continue;
            }
            if arc_at_start + segment_length >= target {
                let parameter = (target - arc_at_start) / segment_length;
                return Ok(a + direction.scale(parameter));
            }
            arc_at_start += segment_length;
        }
        Ok(last)
    }
}
