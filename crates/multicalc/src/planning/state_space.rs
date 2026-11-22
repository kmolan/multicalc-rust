#![deny(clippy::indexing_slicing)]

//! Where a sampling planner's states live, and which of them are free.

use crate::error::PlanningError;
use crate::linear_algebra::Vector;
use crate::random::{RandomScalar, RandomSource};
use crate::scalar::Numeric;

/// The space a sampling planner explores: where states live, how far apart they are, and how to
/// move part-way between two of them.
///
/// This is a trait on correctness grounds rather than for flexibility. A continuous joint at `+179°`
/// and `−179°` is `2°` apart, but raw coordinate subtraction makes it `358°` and interpolates the
/// long way round, so a planner assuming Euclidean coordinates returns wrong nearest neighbours on
/// exactly the arm configurations this crate exists for.
pub trait StateSpace<const DIMENSION: usize, T: Numeric> {
    /// A uniform draw from the whole space.
    fn sample<R: RandomSource<T>>(&self, source: &mut R) -> Vector<DIMENSION, T>
    where
        T: RandomScalar;

    /// Distance between two states. Must be a metric on this space.
    fn distance(&self, from: &Vector<DIMENSION, T>, into: &Vector<DIMENSION, T>) -> T;

    /// The state `amount` of the way from `from` to `to`, `amount` in zero to one.
    fn interpolate(
        &self,
        from: &Vector<DIMENSION, T>,
        into: &Vector<DIMENSION, T>,
        amount: T,
    ) -> Vector<DIMENSION, T>;

    /// Whether a state lies inside the space's bounds. Obstacles are [`StateValidity`]'s job.
    fn contains(&self, state: &Vector<DIMENSION, T>) -> bool {
        let _ = state;
        true
    }
}

/// A box in `DIMENSION`-dimensional Euclidean space.
///
/// ```
/// use multicalc::planning::{BoxSpace, StateSpace};
/// use multicalc::{Pcg32, Vector};
///
/// let space: BoxSpace<2> = BoxSpace::try_new(Vector::new([0.0, 0.0]), Vector::new([4.0, 3.0]))?;
/// let mut source = Pcg32::new(20260830);
///
/// let drawn = space.sample(&mut source);
/// assert!(space.contains(&drawn));
///
/// // The metric and the interpolation are the ordinary Euclidean ones.
/// let corner_to_corner = space.distance(&Vector::new([0.0, 0.0]), &Vector::new([3.0, 4.0]));
/// assert!((corner_to_corner - 5.0).abs() < 1e-12);
/// # Ok::<(), multicalc::CalcError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxSpace<const DIMENSION: usize, T: Numeric = f64> {
    lower: Vector<DIMENSION, T>,
    upper: Vector<DIMENSION, T>,
}

impl<const DIMENSION: usize, T: Numeric> BoxSpace<DIMENSION, T> {
    /// The box between two corners.
    ///
    /// Returns [`PlanningError::NonFinite`] for a non-finite bound and
    /// [`PlanningError::BoundsReversed`] where a lower bound sits above its upper.
    pub fn try_new(
        lower: Vector<DIMENSION, T>,
        upper: Vector<DIMENSION, T>,
    ) -> Result<Self, PlanningError> {
        for axis in 0..DIMENSION {
            let (Some(&low), Some(&high)) = (lower.get(axis), upper.get(axis)) else {
                continue;
            };
            if !low.is_finite() || !high.is_finite() {
                return Err(PlanningError::NonFinite);
            }
            if low > high {
                return Err(PlanningError::BoundsReversed);
            }
        }
        Ok(BoxSpace { lower, upper })
    }

    /// The box's lower corner.
    #[inline]
    pub fn lower(&self) -> Vector<DIMENSION, T> {
        self.lower
    }

    /// The box's upper corner.
    #[inline]
    pub fn upper(&self) -> Vector<DIMENSION, T> {
        self.upper
    }

    /// One axis's bounds.
    fn bounds_of(&self, axis: usize) -> (T, T) {
        let low = self.lower.get(axis).copied().unwrap_or(T::ZERO);
        let high = self.upper.get(axis).copied().unwrap_or(T::ZERO);
        (low, high)
    }
}

impl<const DIMENSION: usize, T: Numeric> StateSpace<DIMENSION, T> for BoxSpace<DIMENSION, T> {
    fn sample<R: RandomSource<T>>(&self, source: &mut R) -> Vector<DIMENSION, T>
    where
        T: RandomScalar,
    {
        Vector::from_fn(|axis| {
            let (low, high) = self.bounds_of(axis);
            low + source.next_unit() * (high - low)
        })
    }

    fn distance(&self, from: &Vector<DIMENSION, T>, into: &Vector<DIMENSION, T>) -> T {
        let mut sum_of_squares = T::ZERO;
        for axis in 0..DIMENSION {
            let (Some(&start), Some(&end)) = (from.get(axis), into.get(axis)) else {
                continue;
            };
            let separation = end - start;
            sum_of_squares += separation * separation;
        }
        sum_of_squares.sqrt()
    }

    fn interpolate(
        &self,
        from: &Vector<DIMENSION, T>,
        into: &Vector<DIMENSION, T>,
        amount: T,
    ) -> Vector<DIMENSION, T> {
        Vector::from_fn(|axis| {
            let start = from.get(axis).copied().unwrap_or(T::ZERO);
            let end = into.get(axis).copied().unwrap_or(T::ZERO);
            start + (end - start) * amount
        })
    }

    fn contains(&self, state: &Vector<DIMENSION, T>) -> bool {
        (0..DIMENSION).all(|axis| {
            let (low, high) = self.bounds_of(axis);
            state
                .get(axis)
                .is_some_and(|&value| value.is_finite() && value >= low && value <= high)
        })
    }
}

/// Whether a state is free of obstacles.
///
/// A check that cannot be evaluated must report the state invalid. The answer is a `bool` rather
/// than a `Result` deliberately: the fallible parts of a real oracle — `forward_kinematics`,
/// `CollisionQuery::check` — fail only on construction bugs, never per state, and a generic error
/// would add a type parameter to every `try_plan` while giving the planner no useful recovery.
pub trait StateValidity<const DIMENSION: usize, T: Numeric> {
    /// Whether the state is free.
    fn is_state_valid(&self, state: &Vector<DIMENSION, T>) -> bool;
}

/// So the everyday case is a one-line closure.
impl<const DIMENSION: usize, T: Numeric, F> StateValidity<DIMENSION, T> for F
where
    F: Fn(&Vector<DIMENSION, T>) -> bool,
{
    fn is_state_valid(&self, state: &Vector<DIMENSION, T>) -> bool {
        self(state)
    }
}
