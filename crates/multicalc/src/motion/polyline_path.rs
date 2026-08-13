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
pub struct PathProjection<const DIMENSION: usize, T: Numeric = f64> {
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
/// let arc_length_so_far = 0.0;
/// let lookahead_distance = 2.0;
/// let [x, y] = path
///     .lookahead_point(arc_length_so_far, lookahead_distance)
///     .unwrap()
///     .into_array();
/// assert!((x - 2.0).abs() < 1e-12 && y.abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolylinePath<const MAX_POINTS: usize, const DIMENSION: usize, T: Numeric = f64> {
    points: [Vector<DIMENSION, T>; MAX_POINTS],
    cumulative_lengths: [T; MAX_POINTS],
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
            cumulative_lengths: [T::ZERO; MAX_POINTS],
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

        // Return empty `path` if `points.len` is 0.
        let Some(&first_point) = points.first() else {
            return Ok(path);
        };

        let mut acc = T::ZERO;

        path.points[0] = first_point;
        path.cumulative_lengths[0] = acc;

        let slots = path
            .points
            .iter_mut()
            .zip(path.cumulative_lengths.iter_mut())
            .skip(1);
        let point_pairs = points.iter().zip(points.iter().skip(1));

        for ((slot_point, slot_cumulative_length), (&a, &b)) in slots.zip(point_pairs) {
            acc += (b - a).norm();
            *slot_point = b;
            *slot_cumulative_length = acc;
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

        let acc = if self.length == 0 {
            T::ZERO
        } else {
            let last_index = self.length - 1;
            let last_point = self
                .points
                .get_mut(last_index)
                .ok_or(MotionError::CapacityExceeded)?;
            let last_cumulative = self
                .cumulative_lengths
                .get_mut(last_index)
                .ok_or(MotionError::CapacityExceeded)?;

            *last_cumulative + (point - *last_point).norm()
        };

        let slot_point = self
            .points
            .get_mut(self.length)
            .ok_or(MotionError::CapacityExceeded)?;
        let slot_cumulative = self
            .cumulative_lengths
            .get_mut(self.length)
            .ok_or(MotionError::CapacityExceeded)?;

        *slot_point = point;
        *slot_cumulative = acc;
        self.length += 1;

        Ok(())
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
        if self.length == 0 {
            return T::ZERO;
        }

        self.cumulative_lengths
            .get(self.length - 1)
            .copied()
            .unwrap_or(T::ZERO)
    }

    /// The closest point on the path to a query point.
    ///
    /// Returns [`MotionError::PathTooShort`] if the path has no waypoints, or
    /// [`MotionError::OutOfSync`] if the path cannot get the cumulative arc length.
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
                    arc_length: self
                        .cumulative_lengths
                        .get(segment_index)
                        .copied()
                        .ok_or(MotionError::OutOfSync)?
                        + segment_length * parameter,
                    distance,
                });
            }
        }
        best.ok_or(MotionError::PathTooShort)
    }

    /// The point a given arc length ahead of a starting arc length along the path.
    ///
    /// The end-of-path mode decides what happens once the target runs past the end: [`EndOfPath::Stop`]
    /// clamps to the last waypoint and [`EndOfPath::Loop`] wraps around. Returns
    /// [`MotionError::PathTooShort`] if the path has no waypoints, or
    /// [`MotionError::OutOfSync`] if the path cannot get the cumulative arc length.
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

        for (segment_index, window) in waypoints.windows(2).enumerate() {
            let (a, b) = match window {
                [a, b] => (*a, *b),
                _ => continue,
            };
            let direction = b - a;
            let segment_length = direction.norm();
            if segment_length == T::ZERO {
                continue;
            }

            let arc_length = self
                .cumulative_lengths
                .get(segment_index)
                .copied()
                .ok_or(MotionError::OutOfSync)?;
            if arc_length + segment_length >= target {
                let parameter = (target - arc_length) / segment_length;
                return Ok(a + direction.scale(parameter));
            }
        }
        Ok(last)
    }
}
