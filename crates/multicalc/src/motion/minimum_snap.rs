//! Planning the smoothest path through a set of waypoints.
#![deny(clippy::indexing_slicing)]

use crate::error::MotionError;
use crate::linear_algebra::{Matrix, Vector};
use crate::polynomial::{PiecewisePolynomial, Polynomial, endpoint_mapping_inverse};
use crate::scalar::Numeric;

/// Each segment is a degree-7 curve, which is what smoothing the fourth derivative asks for.
const COEFFICIENTS_PER_SEGMENT: usize = 8;
/// Position, velocity, acceleration and jerk are pinned at each end of a segment.
const DERIVATIVES_PER_WAYPOINT: usize = 4;
/// Of those, position is always known; the other three are what the solve chooses.
const FREE_DERIVATIVES_PER_WAYPOINT: usize = 3;

/// How the path is moving where it starts or finishes.
///
/// [`Default`] is all zeros, which means starting or finishing at a standstill.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryDerivatives<const DIMENSION: usize, T: Numeric = f64> {
    /// How fast it is moving.
    pub velocity: Vector<DIMENSION, T>,
    /// How fast that speed is changing.
    pub acceleration: Vector<DIMENSION, T>,
    /// How fast that change is itself changing.
    pub jerk: Vector<DIMENSION, T>,
}

impl<const DIMENSION: usize, T: Numeric> Default for BoundaryDerivatives<DIMENSION, T> {
    fn default() -> Self {
        Self {
            velocity: Vector::zeros(),
            acceleration: Vector::zeros(),
            jerk: Vector::zeros(),
        }
    }
}

impl<const DIMENSION: usize, T: Numeric> BoundaryDerivatives<DIMENSION, T> {
    /// Whether every value is finite.
    fn all_finite(&self) -> bool {
        self.velocity.is_finite() && self.acceleration.is_finite() && self.jerk.is_finite()
    }

    /// The value for one order, counting velocity as 1 and jerk as 3.
    fn at_order(&self, order: usize) -> Vector<DIMENSION, T> {
        match order {
            1 => self.velocity,
            2 => self.acceleration,
            _ => self.jerk,
        }
    }
}

/// Whether one end of a segment is already known, or is for the solve to choose.
enum EndpointDerivative<const DIMENSION: usize, T: Numeric> {
    /// Known up front: every position, and the motion at the very start and finish.
    Fixed(Vector<DIMENSION, T>),
    /// For the solve to choose, at this row of the reduced system.
    Free(usize),
}

/// `index · (index - 1) · (index - 2) · (index - 3)`, which is how many ways a term survives being
/// differentiated four times.
fn ways_after_four_derivatives<T: Numeric>(index: usize) -> T {
    let mut product = T::ONE;
    for step in 0..4 {
        product *= T::from_usize(index.saturating_sub(step));
    }
    product
}

/// The 8×8 that turns a segment's coefficients into its total snap, on the segment's own 0-to-1
/// clock.
///
/// Only terms from the fourth power up survive four differentiations, so everything below is zero.
fn snap_cost<T: Numeric>() -> Matrix<COEFFICIENTS_PER_SEGMENT, COEFFICIENTS_PER_SEGMENT, T> {
    let mut cost = Matrix::zeros();
    for row in 4..COEFFICIENTS_PER_SEGMENT {
        for column in 4..COEFFICIENTS_PER_SEGMENT {
            let value = ways_after_four_derivatives::<T>(row)
                * ways_after_four_derivatives::<T>(column)
                / T::from_usize(row + column - 7);
            if let Some(slot) = cost.get_mut(row, column) {
                *slot = value;
            }
        }
    }
    cost
}

/// One axis of a waypoint's four values.
fn along_one_axis<const DIMENSION: usize, T: Numeric>(
    block: &[Vector<DIMENSION, T>; DERIVATIVES_PER_WAYPOINT],
    axis: usize,
) -> [T; DERIVATIVES_PER_WAYPOINT] {
    let mut values = [T::ZERO; DERIVATIVES_PER_WAYPOINT];
    for (slot, vector) in values.iter_mut().zip(block.iter()) {
        *slot = vector.get(axis).copied().unwrap_or(T::ZERO);
    }
    values
}

/// Plans the smoothest path through a set of waypoints, as one polynomial per segment.
///
/// Smoothest here means the fourth derivative of position is kept as small as possible over the
/// whole path — the quantity a multirotor's motors have to produce, which is why this is the usual
/// choice for one. The path passes through every waypoint exactly and is smooth in position,
/// velocity, acceleration and jerk everywhere, including across the joins.
///
/// **Planning does not belong in a control loop.** The work grows with the number of waypoints and
/// includes a matrix factorization, so it is not bounded per tick. Plan once, off the loop; the
/// [`PiecewisePolynomial`] it returns is what the loop evaluates, and that *is* bounded.
///
/// `MAX_FREE_DERIVATIVES` must be at least `3 × (MAX_SEGMENTS - 1)` — three values are chosen at
/// each waypoint between the first and last. Stable Rust cannot work that out from `MAX_SEGMENTS`,
/// so it is given separately and checked at runtime:
///
/// | Segments | `MAX_FREE_DERIVATIVES` needed |
/// |---|---|
/// | 4 | 9 |
/// | 8 | 21 |
/// | 12 | 33 |
///
/// Too small gives [`MotionError::WorkspaceTooSmall`] rather than a panic.
///
/// ```
/// use multicalc::motion::MinimumSnapPlanner;
/// use multicalc::linear_algebra::Vector;
///
/// // Three segments in two dimensions, from a standstill to a standstill.
/// let planner = MinimumSnapPlanner::<4, 9, 2, f64>::new();
/// let waypoints = [
///     Vector::new([0.0, 0.0]),
///     Vector::new([1.0, 2.0]),
///     Vector::new([3.0, 1.0]),
///     Vector::new([4.0, 3.0]),
/// ];
/// let trajectory = planner.plan(&waypoints, &[1.0, 1.5, 1.2]).unwrap();
///
/// // It arrives at the second waypoint at the end of the first segment.
/// let [x, y] = trajectory.evaluate(1.0).unwrap().into_array();
/// assert!((x - 1.0).abs() < 1e-9 && (y - 2.0).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimumSnapPlanner<
    const MAX_SEGMENTS: usize,
    const MAX_FREE_DERIVATIVES: usize,
    const DIMENSION: usize,
    T: Numeric = f64,
> {
    start: BoundaryDerivatives<DIMENSION, T>,
    end: BoundaryDerivatives<DIMENSION, T>,
}

impl<
    const MAX_SEGMENTS: usize,
    const MAX_FREE_DERIVATIVES: usize,
    const DIMENSION: usize,
    T: Numeric,
> Default for MinimumSnapPlanner<MAX_SEGMENTS, MAX_FREE_DERIVATIVES, DIMENSION, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    const MAX_SEGMENTS: usize,
    const MAX_FREE_DERIVATIVES: usize,
    const DIMENSION: usize,
    T: Numeric,
> MinimumSnapPlanner<MAX_SEGMENTS, MAX_FREE_DERIVATIVES, DIMENSION, T>
{
    /// A planner starting and finishing at a standstill.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: BoundaryDerivatives::default(),
            end: BoundaryDerivatives::default(),
        }
    }

    /// Sets how the path is moving where it starts.
    #[inline]
    #[must_use]
    pub fn with_start(mut self, boundary: BoundaryDerivatives<DIMENSION, T>) -> Self {
        self.start = boundary;
        self
    }

    /// Sets how the path is moving where it finishes.
    #[inline]
    #[must_use]
    pub fn with_end(mut self, boundary: BoundaryDerivatives<DIMENSION, T>) -> Self {
        self.end = boundary;
        self
    }

    /// Whether one end of a segment is already known, and if not, which row of the solve chooses it.
    fn classify(
        &self,
        waypoint: usize,
        order: usize,
        waypoints: &[Vector<DIMENSION, T>],
        segments: usize,
    ) -> EndpointDerivative<DIMENSION, T> {
        // Where the path goes is given, at every waypoint.
        if order == 0 {
            return EndpointDerivative::Fixed(
                waypoints.get(waypoint).copied().unwrap_or(Vector::zeros()),
            );
        }
        // How it is moving is given at the two ends of the path, and chosen everywhere between.
        if waypoint == 0 {
            return EndpointDerivative::Fixed(self.start.at_order(order));
        }
        if waypoint == segments {
            return EndpointDerivative::Fixed(self.end.at_order(order));
        }
        EndpointDerivative::Free(FREE_DERIVATIVES_PER_WAYPOINT * (waypoint - 1) + (order - 1))
    }

    /// Plans a trajectory through the waypoints, taking each segment the time given for it.
    ///
    /// There must be one duration for each pair of waypoints;
    /// [`durations_from_average_speed`] gives a reasonable first set.
    ///
    /// Returns [`MotionError::PathTooShort`] for fewer than two waypoints,
    /// [`MotionError::CapacityExceeded`] for more segments than fit,
    /// [`MotionError::SegmentCountMismatch`] when the duration count does not match,
    /// [`MotionError::DurationNotPositive`] for a duration that is zero, negative or not a number,
    /// [`MotionError::NonFinite`] for a waypoint or boundary value that is not finite,
    /// [`MotionError::WorkspaceTooSmall`] when `MAX_FREE_DERIVATIVES` is below
    /// `3 × (segments - 1)`, and [`MotionError::Linalg`] when the system cannot be solved.
    pub fn plan(
        &self,
        waypoints: &[Vector<DIMENSION, T>],
        durations: &[T],
    ) -> Result<
        PiecewisePolynomial<MAX_SEGMENTS, COEFFICIENTS_PER_SEGMENT, DIMENSION, T>,
        MotionError,
    > {
        if waypoints.len() < 2 {
            return Err(MotionError::PathTooShort);
        }
        let segments = waypoints.len() - 1;
        if segments > MAX_SEGMENTS {
            return Err(MotionError::CapacityExceeded);
        }
        if durations.len() != segments {
            return Err(MotionError::SegmentCountMismatch);
        }
        for duration in durations {
            if !duration.is_finite() || *duration <= T::ZERO {
                return Err(MotionError::DurationNotPositive);
            }
        }
        if waypoints.iter().any(|waypoint| !waypoint.is_finite())
            || !self.start.all_finite()
            || !self.end.all_finite()
        {
            return Err(MotionError::NonFinite);
        }
        let free_count = FREE_DERIVATIVES_PER_WAYPOINT * (segments - 1);
        if free_count > MAX_FREE_DERIVATIVES {
            return Err(MotionError::WorkspaceTooSmall);
        }

        // Build the system that chooses the free values. Everything already known moves to the other
        // side, so the global system never has to exist.
        let cost = snap_cost::<T>();
        let mut reduced = Matrix::<MAX_FREE_DERIVATIVES, MAX_FREE_DERIVATIVES, T>::zeros();
        let mut known_side = Matrix::<MAX_FREE_DERIVATIVES, DIMENSION, T>::zeros();

        for (segment, duration) in durations.iter().enumerate() {
            // The segment's cost is written against its coefficients; this restates it against the
            // values at its two ends, which is what neighbouring segments share.
            let inverse_mapping = endpoint_mapping_inverse(*duration)?;
            let in_endpoint_terms =
                inverse_mapping.transpose() * cost.scale(duration.powi(-7)) * inverse_mapping;

            for local_row in 0..COEFFICIENTS_PER_SEGMENT {
                let row_waypoint = segment + local_row / DERIVATIVES_PER_WAYPOINT;
                let row_order = local_row % DERIVATIVES_PER_WAYPOINT;
                // A row for something already known contributes nothing to choose.
                let EndpointDerivative::Free(free_row) =
                    self.classify(row_waypoint, row_order, waypoints, segments)
                else {
                    continue;
                };

                for local_column in 0..COEFFICIENTS_PER_SEGMENT {
                    let column_waypoint = segment + local_column / DERIVATIVES_PER_WAYPOINT;
                    let column_order = local_column % DERIVATIVES_PER_WAYPOINT;
                    let entry = in_endpoint_terms
                        .get(local_row, local_column)
                        .copied()
                        .unwrap_or(T::ZERO);

                    match self.classify(column_waypoint, column_order, waypoints, segments) {
                        EndpointDerivative::Free(free_column) => {
                            // Adjacent segments write the same entry, which is what makes the path
                            // continuous without any condition saying so.
                            if let Some(slot) = reduced.get_mut(free_row, free_column) {
                                *slot += entry;
                            }
                        }
                        EndpointDerivative::Fixed(value) => {
                            for axis in 0..DIMENSION {
                                let known = value.get(axis).copied().unwrap_or(T::ZERO);
                                if let Some(slot) = known_side.get_mut(free_row, axis) {
                                    *slot -= entry * known;
                                }
                            }
                        }
                    }
                }
            }
        }

        // The system is meant to read the same across the diagonal; rounding can leave it slightly
        // off, which the factorization would rather not see.
        let mut system = (reduced + reduced.transpose()).scale(T::HALF);
        // Rows past what this many segments needs are left empty, so give them a 1 on the diagonal
        // and they solve to zero rather than being singular.
        for row in free_count..MAX_FREE_DERIVATIVES {
            if let Some(slot) = system.get_mut(row, row) {
                *slot = T::ONE;
            }
        }

        // The durations alone decide the system, so one factorization covers every axis.
        let solved = if free_count == 0 {
            Matrix::<MAX_FREE_DERIVATIVES, DIMENSION, T>::zeros()
        } else {
            system.lu()?.solve_matrix::<DIMENSION>(known_side)
        };

        // Gather each waypoint's four values, taking the known ones directly and the chosen ones
        // from the solve. One more waypoint than segments, and the last is held on its own because
        // an array of `MAX_SEGMENTS + 1` cannot be written.
        let mut blocks =
            [[Vector::<DIMENSION, T>::zeros(); DERIVATIVES_PER_WAYPOINT]; MAX_SEGMENTS];
        let mut last_block = [Vector::<DIMENSION, T>::zeros(); DERIVATIVES_PER_WAYPOINT];
        for waypoint in 0..=segments {
            let mut block = [Vector::<DIMENSION, T>::zeros(); DERIVATIVES_PER_WAYPOINT];
            for (order, slot) in block.iter_mut().enumerate() {
                *slot = match self.classify(waypoint, order, waypoints, segments) {
                    EndpointDerivative::Fixed(value) => value,
                    EndpointDerivative::Free(row) => {
                        Vector::from_fn(|axis| solved.get(row, axis).copied().unwrap_or(T::ZERO))
                    }
                };
            }
            if waypoint == segments {
                last_block = block;
            } else if let Some(slot) = blocks.get_mut(waypoint) {
                *slot = block;
            }
        }

        // Each segment is then the one curve matching those four values at each of its ends.
        let mut pieces =
            [[Polynomial::<COEFFICIENTS_PER_SEGMENT, T>::zeros(); DIMENSION]; MAX_SEGMENTS];
        let mut spans = [T::ZERO; MAX_SEGMENTS];
        let empty = [Vector::<DIMENSION, T>::zeros(); DERIVATIVES_PER_WAYPOINT];

        for segment in 0..segments {
            let duration = durations.get(segment).copied().unwrap_or(T::ZERO);
            let start_block = blocks.get(segment).copied().unwrap_or(empty);
            let end_block = if segment + 1 == segments {
                last_block
            } else {
                blocks.get(segment + 1).copied().unwrap_or(empty)
            };

            for axis in 0..DIMENSION {
                let piece = Polynomial::<COEFFICIENTS_PER_SEGMENT, T>::from_endpoint_derivatives(
                    &along_one_axis(&start_block, axis),
                    &along_one_axis(&end_block, axis),
                    duration,
                )?;
                if let Some(slot) = pieces.get_mut(segment).and_then(|row| row.get_mut(axis)) {
                    *slot = piece;
                }
            }
            if let Some(slot) = spans.get_mut(segment) {
                *slot = duration;
            }
        }

        Ok(PiecewisePolynomial::try_from_pieces(
            pieces.get(..segments).unwrap_or(&[]),
            spans.get(..segments).unwrap_or(&[]),
        )?)
    }
}

/// Fills `durations` with a time for each pair of waypoints, so the whole path is covered at
/// roughly the given speed.
///
/// A starting point for tuning, not the fastest possible timing: a sharp corner needs more time than
/// its straight-line distance suggests, because the path has to slow down to turn.
///
/// Returns [`MotionError::PathTooShort`] for fewer than two waypoints,
/// [`MotionError::SegmentCountMismatch`] when `durations` is not one shorter than `waypoints`,
/// [`MotionError::NonFinite`] for a waypoint that is not finite, and
/// [`MotionError::DurationNotPositive`] when the speed is not above zero or when two waypoints in a
/// row sit in the same place — a segment covering no distance has no sensible duration.
///
/// ```
/// use multicalc::linear_algebra::Vector;
/// use multicalc::motion::durations_from_average_speed;
///
/// // Three units then four, at two units a second.
/// let waypoints = [
///     Vector::new([0.0, 0.0]),
///     Vector::new([3.0, 0.0]),
///     Vector::new([3.0, 4.0]),
/// ];
/// let mut durations = [0.0_f64; 2];
/// durations_from_average_speed(&waypoints, 2.0, &mut durations).unwrap();
/// assert!((durations[0] - 1.5).abs() < 1e-12 && (durations[1] - 2.0).abs() < 1e-12);
/// ```
pub fn durations_from_average_speed<const DIMENSION: usize, T: Numeric>(
    waypoints: &[Vector<DIMENSION, T>],
    average_speed: T,
    durations: &mut [T],
) -> Result<(), MotionError> {
    if waypoints.len() < 2 {
        return Err(MotionError::PathTooShort);
    }
    if durations.len() != waypoints.len() - 1 {
        return Err(MotionError::SegmentCountMismatch);
    }
    if !average_speed.is_finite() || average_speed <= T::ZERO {
        return Err(MotionError::DurationNotPositive);
    }
    if waypoints.iter().any(|waypoint| !waypoint.is_finite()) {
        return Err(MotionError::NonFinite);
    }

    for (slot, pair) in durations.iter_mut().zip(waypoints.windows(2)) {
        let distance = match (pair.first(), pair.get(1)) {
            (Some(from), Some(to)) => (*to - *from).norm(),
            _ => T::ZERO,
        };
        if distance <= T::ZERO {
            return Err(MotionError::DurationNotPositive);
        }
        *slot = distance / average_speed;
    }
    Ok(())
}
