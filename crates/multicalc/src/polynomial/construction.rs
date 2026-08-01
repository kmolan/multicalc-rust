//! Building a polynomial from roots, from sampled points, or from a series expansion.
//!
//! [`Polynomial::from_points`] and [`Polynomial::fit_least_squares`] both shift and stretch the
//! sample positions into the range -1 to 1 before doing any work, and undo that at the end. Without
//! it, points spread over a wide range raise the numbers in the solve to high powers and it
//! degenerates. With it, fits stay well behaved to about the eighth power in `f64`, which is past
//! anything worth fitting with a single polynomial — beyond that, use pieces.
//!
//! One thing that helps with is worth being plain about. Undoing the shift at the end puts the
//! answer back into powers of the caller's own variable, and when the samples sit far from zero
//! that step gives up digits no matter how the fit was done: a small value near zero comes out of
//! adding and subtracting much larger ones. Samples spread over 0 to 1000 with values reaching
//! `1e11` reproduce those values to rounding against `1e11`, so the small ones near zero are right
//! to about six digits rather than all sixteen. Keeping the samples near zero, or asking for the
//! value rather than reading the coefficients, avoids it.
#![deny(clippy::indexing_slicing)]

use crate::error::{LinalgError, PolynomialError};
use crate::linear_algebra::{Matrix, PivotedQr, Vector};
use crate::polynomial::Polynomial;
use crate::scalar::{Jet, Numeric};

/// Multiplies the polynomial held in `coefficients` by `(x - root)`, in place.
///
/// `raised_degree` is the degree the answer reaches, one more than what is there now.
fn multiply_in_a_root<T: Numeric>(coefficients: &mut [T], raised_degree: usize, root: T) {
    // Each coefficient becomes the one below it, less the root times itself, so a single value
    // carried along is enough to work upward without a second copy.
    let mut carried = T::ZERO;
    for power in 0..=raised_degree {
        let current = coefficients.get(power).copied().unwrap_or(T::ZERO);
        if let Some(slot) = coefficients.get_mut(power) {
            *slot = carried - root * current;
        }
        carried = current;
    }
}

/// The middle of the sample positions and half their spread, which together move any position into
/// the range -1 to 1.
///
/// A spread of zero means every position is the same, which no polynomial can be fitted through.
fn sample_range<T: Numeric>(nodes: &[T]) -> Result<(T, T), PolynomialError> {
    let mut smallest = nodes.first().copied().ok_or(PolynomialError::Empty)?;
    let mut largest = smallest;
    for node in nodes {
        smallest = smallest.min(*node);
        largest = largest.max(*node);
    }
    let half_width = (largest - smallest) * T::HALF;
    if half_width == T::ZERO {
        return Err(PolynomialError::DuplicateNode);
    }
    Ok(((smallest + largest) * T::HALF, half_width))
}

impl<const COEFFICIENT_COUNT: usize, T: Numeric> Polynomial<COEFFICIENT_COUNT, T> {
    /// The polynomial with exactly these roots, and 1 as its highest coefficient.
    ///
    /// Returns [`PolynomialError::DegreeOverflow`] when there are more roots than the polynomial
    /// has room for, and [`PolynomialError::NonFinite`] when a root is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // (x - 1)(x - 2)(x - 3)
    /// let p = Polynomial::<4>::from_roots(&[1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(p.coefficients(), &[-6.0, 11.0, -6.0, 1.0]);
    /// ```
    pub fn from_roots(roots: &[T]) -> Result<Self, PolynomialError> {
        if roots.iter().any(|root| !root.is_finite()) {
            return Err(PolynomialError::NonFinite);
        }
        if roots.len() >= COEFFICIENT_COUNT {
            return Err(PolynomialError::DegreeOverflow);
        }

        let mut coefficients = [T::ZERO; COEFFICIENT_COUNT];
        match coefficients.get_mut(0) {
            Some(slot) => *slot = T::ONE,
            None => return Err(PolynomialError::DegreeOverflow),
        }
        for (index, root) in roots.iter().enumerate() {
            multiply_in_a_root(&mut coefficients, index + 1, *root);
        }
        Ok(Self::new(coefficients))
    }

    /// The series a [`Jet`] carries, as a polynomial.
    ///
    /// A jet's coefficients are already those of the series, so this copies them across. The result
    /// describes the function near the point the jet was expanded about, so
    /// [`shift_argument`](Self::shift_argument) is what moves that point somewhere else.
    ///
    /// Returns [`PolynomialError::DegreeOverflow`] when the jet holds more coefficients than fit.
    ///
    /// ```
    /// use multicalc::{Jet, Polynomial};
    ///
    /// // Squaring near x = 0 gives back exactly x².
    /// let squared = Jet::<f64, 3>::variable(0.0) * Jet::<f64, 3>::variable(0.0);
    /// let p = Polynomial::<3>::from_jet(&squared).unwrap();
    /// assert_eq!(p.coefficients(), &[0.0, 0.0, 1.0]);
    /// ```
    pub fn from_jet<const JET_ORDER: usize>(
        jet: &Jet<T, JET_ORDER>,
    ) -> Result<Self, PolynomialError> {
        if JET_ORDER > COEFFICIENT_COUNT {
            return Err(PolynomialError::DegreeOverflow);
        }
        let mut coefficients = [T::ZERO; COEFFICIENT_COUNT];
        for (slot, coefficient) in coefficients.iter_mut().zip(jet.coeffs.iter()) {
            *slot = *coefficient;
        }
        Ok(Self::new(coefficients))
    }

    /// Sample positions that bunch up toward the ends of the range, in increasing order.
    ///
    /// Sampling a function at these instead of at evenly spaced positions keeps a fit through them
    /// from swinging wildly near the ends, which is what evenly spaced points cause once there are
    /// more than a handful.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// let nodes = Polynomial::<4>::chebyshev_nodes(-1.0, 1.0);
    /// assert!(nodes[0] > -1.0 && nodes[3] < 1.0);
    /// assert!(nodes[0] < nodes[1] && nodes[1] < nodes[2] && nodes[2] < nodes[3]);
    /// ```
    #[must_use]
    pub fn chebyshev_nodes(lower: T, upper: T) -> [T; COEFFICIENT_COUNT] {
        let centre = (lower + upper) * T::HALF;
        let half_width = (upper - lower) * T::HALF;
        let count = T::from_usize(COEFFICIENT_COUNT);

        let mut nodes = [T::ZERO; COEFFICIENT_COUNT];
        for (index, slot) in nodes.iter_mut().enumerate() {
            // The formula runs from the top of the range downward, so count backward to hand them
            // back in increasing order.
            let step = COEFFICIENT_COUNT - 1 - index;
            let angle = (T::TWO * T::from_usize(step) + T::ONE) * T::PI / (T::TWO * count);
            *slot = centre + half_width * angle.cos();
        }
        nodes
    }

    /// The one polynomial passing through every given point.
    ///
    /// There must be exactly as many points as the polynomial has coefficients, since that is how
    /// many it takes to pin one down. Returns [`PolynomialError::DuplicateNode`] when two points
    /// share a position, and [`PolynomialError::NonFinite`] when any value is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // Three points off 1 - 2x + 3x².
    /// let p = Polynomial::<3>::from_points(&[0.0, 1.0, 2.0], &[1.0, 2.0, 9.0]).unwrap();
    /// assert!((p.evaluate(3.0) - 22.0).abs() < 1e-10);
    /// ```
    pub fn from_points(
        nodes: &[T; COEFFICIENT_COUNT],
        values: &[T; COEFFICIENT_COUNT],
    ) -> Result<Self, PolynomialError> {
        if nodes
            .iter()
            .chain(values.iter())
            .any(|value| !value.is_finite())
        {
            return Err(PolynomialError::NonFinite);
        }
        // A single point is just a constant, and has no spread to normalize against.
        if COEFFICIENT_COUNT <= 1 {
            return Ok(Self::new(*values));
        }
        let (centre, half_width) = sample_range(nodes)?;

        let mut normalized = [T::ZERO; COEFFICIENT_COUNT];
        for (slot, node) in normalized.iter_mut().zip(nodes.iter()) {
            *slot = (*node - centre) / half_width;
        }

        // Work out how much each extra point bends the answer away from the points before it. Each
        // pass over the table replaces neighbouring differences with their own differences, and the
        // first entry of each pass is what the polynomial needs.
        let mut table = *values;
        let mut bends = [T::ZERO; COEFFICIENT_COUNT];
        if let Some(slot) = bends.get_mut(0) {
            *slot = table.first().copied().unwrap_or(T::ZERO);
        }
        for order in 1..COEFFICIENT_COUNT {
            for index in 0..COEFFICIENT_COUNT - order {
                let current = table.get(index).copied().unwrap_or(T::ZERO);
                let next = table.get(index + 1).copied().unwrap_or(T::ZERO);
                let near = normalized.get(index).copied().unwrap_or(T::ZERO);
                let far = normalized.get(index + order).copied().unwrap_or(T::ZERO);
                let gap = far - near;
                if gap == T::ZERO {
                    return Err(PolynomialError::DuplicateNode);
                }
                if let Some(slot) = table.get_mut(index) {
                    *slot = (next - current) / gap;
                }
            }
            if let Some(slot) = bends.get_mut(order) {
                *slot = table.first().copied().unwrap_or(T::ZERO);
            }
        }

        // Fold those back into ordinary coefficients, starting from the last and working down: each
        // step multiplies by the matching point's position and adds the next bend in.
        let mut coefficients = [T::ZERO; COEFFICIENT_COUNT];
        if let Some(slot) = coefficients.get_mut(0) {
            *slot = bends.last().copied().unwrap_or(T::ZERO);
        }
        for step in (0..COEFFICIENT_COUNT - 1).rev() {
            let node = normalized.get(step).copied().unwrap_or(T::ZERO);
            multiply_in_a_root(&mut coefficients, COEFFICIENT_COUNT - 1 - step, node);
            if let Some(slot) = coefficients.get_mut(0) {
                *slot += bends.get(step).copied().unwrap_or(T::ZERO);
            }
        }

        // Undo the shift and stretch that moved the positions into the range -1 to 1.
        Ok(Self::new(coefficients)
            .scale_argument(T::ONE / half_width)
            .shift_argument(-centre))
    }

    /// The polynomial of this size that comes closest to more points than it can pass through.
    ///
    /// Closest means the squared misses summed over every point are as small as they can be.
    /// Returns [`PolynomialError::TooFewSamples`] when there are fewer points than coefficients,
    /// [`PolynomialError::DuplicateNode`] when every point shares one position, and
    /// [`PolynomialError::NonFinite`] when any value is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // Five points off 1 + 2x, fitted with a straight line.
    /// let nodes = [0.0, 1.0, 2.0, 3.0, 4.0];
    /// let values = [1.0, 3.0, 5.0, 7.0, 9.0];
    /// let p = Polynomial::<2>::fit_least_squares(&nodes, &values).unwrap();
    /// assert!((p.evaluate(10.0) - 21.0).abs() < 1e-10);
    /// ```
    pub fn fit_least_squares<const SAMPLE_COUNT: usize>(
        nodes: &[T; SAMPLE_COUNT],
        values: &[T; SAMPLE_COUNT],
    ) -> Result<Self, PolynomialError> {
        if SAMPLE_COUNT < COEFFICIENT_COUNT {
            return Err(PolynomialError::TooFewSamples);
        }
        if nodes
            .iter()
            .chain(values.iter())
            .any(|value| !value.is_finite())
        {
            return Err(PolynomialError::NonFinite);
        }
        let (centre, half_width) = sample_range(nodes)?;

        // One row per point, holding that point's position raised to each power in turn.
        let mut design = Matrix::<SAMPLE_COUNT, COEFFICIENT_COUNT, T>::zeros();
        for (row, node) in nodes.iter().enumerate() {
            let normalized = (*node - centre) / half_width;
            let mut raised = T::ONE;
            for column in 0..COEFFICIENT_COUNT {
                if let Some(slot) = design.get_mut(row, column) {
                    *slot = raised;
                }
                raised *= normalized;
            }
        }

        let solved = PivotedQr::decompose(design)?.solve_least_squares(Vector::new(*values))?;
        Ok(Self::new(solved.into_array())
            .scale_argument(T::ONE / half_width)
            .shift_argument(-centre))
    }
}

/// Turns the eight coefficients of a degree-7 piece into its value and first three derivatives at
/// each end, measured against the outer parameter rather than the piece's own 0-to-1 clock.
///
/// `span` is how much of the outer parameter the piece covers, which is what converts between the
/// two.
pub(crate) fn endpoint_mapping<T: Numeric>(span: T) -> Matrix<8, 8, T> {
    let mut mapping = Matrix::<8, 8, T>::zeros();

    // At the start of the piece only the lowest coefficients survive: the value is the first one,
    // and each derivative picks out one more.
    if let Some(slot) = mapping.get_mut(0, 0) {
        *slot = T::ONE;
    }
    let mut ways = T::ONE;
    let mut span_raised = T::ONE;
    for order in 1..4 {
        ways *= T::from_usize(order);
        span_raised *= span;
        if let Some(slot) = mapping.get_mut(order, order) {
            *slot = ways / span_raised;
        }
    }

    // At the end of the piece every coefficient contributes, since the clock reads one there.
    for column in 0..8 {
        if let Some(slot) = mapping.get_mut(4, column) {
            *slot = T::ONE;
        }
    }
    let mut span_raised = T::ONE;
    for order in 1..4 {
        span_raised *= span;
        for column in order..8 {
            // Differentiating `order` times leaves this many copies of the term behind.
            let mut ways = T::ONE;
            for step in 0..order {
                ways *= T::from_usize(column - step);
            }
            if let Some(slot) = mapping.get_mut(4 + order, column) {
                *slot = ways / span_raised;
            }
        }
    }
    mapping
}

/// The other direction: the eight coefficients that produce a wanted value and first three
/// derivatives at each end.
pub(crate) fn endpoint_mapping_inverse<T: Numeric>(
    span: T,
) -> Result<Matrix<8, 8, T>, LinalgError> {
    endpoint_mapping(span).inverse()
}

impl<T: Numeric> Polynomial<4, T> {
    /// The smooth curve matching a value and a slope at each end.
    ///
    /// The result runs on the piece's own clock, from 0 at the start to 1 at the end, and `span` is
    /// how much of the outer parameter that covers — it is what converts the caller's slopes, which
    /// are measured against the outer parameter.
    ///
    /// Returns [`PolynomialError::SpanNotPositive`] when `span` is zero or negative, and
    /// [`PolynomialError::NonFinite`] when any input is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // From 0 to 10 over two units of the outer parameter, at rest at both ends.
    /// let p = Polynomial::<4>::from_endpoint_derivatives(0.0, 0.0, 10.0, 0.0, 2.0).unwrap();
    /// assert!(p.evaluate(0.0).abs() < 1e-12);
    /// assert!((p.evaluate(1.0) - 10.0).abs() < 1e-12);
    /// ```
    pub fn from_endpoint_derivatives(
        start_value: T,
        start_slope: T,
        end_value: T,
        end_slope: T,
        span: T,
    ) -> Result<Self, PolynomialError> {
        for value in [start_value, start_slope, end_value, end_slope, span] {
            if !value.is_finite() {
                return Err(PolynomialError::NonFinite);
            }
        }
        if span <= T::ZERO {
            return Err(PolynomialError::SpanNotPositive);
        }

        // Slopes against the outer parameter become slopes against the piece's own clock.
        let start_rise = start_slope * span;
        let end_rise = end_slope * span;
        Ok(Self::new([
            start_value,
            start_rise,
            -T::THREE * start_value - T::TWO * start_rise + T::THREE * end_value - end_rise,
            T::TWO * start_value + start_rise - T::TWO * end_value + end_rise,
        ]))
    }
}

impl<T: Numeric> Polynomial<8, T> {
    /// The curve matching a value and its first three derivatives at each end.
    ///
    /// `start` and `end` each hold `[value, velocity, acceleration, jerk]` measured against the
    /// outer parameter. As with the cubic, the result runs on the piece's own 0-to-1 clock and
    /// `span` is how much of the outer parameter that covers.
    ///
    /// Returns [`PolynomialError::SpanNotPositive`] when `span` is zero or negative,
    /// [`PolynomialError::NonFinite`] when any input is infinite or NaN, and
    /// [`PolynomialError::Linalg`] when the eight conditions cannot be solved.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // From 0 to 10 over two units, starting and finishing at a standstill.
    /// let still = [0.0; 4];
    /// let arrived = [10.0, 0.0, 0.0, 0.0];
    /// let p = Polynomial::<8>::from_endpoint_derivatives(&still, &arrived, 2.0).unwrap();
    /// assert!(p.evaluate(0.0).abs() < 1e-10);
    /// assert!((p.evaluate(1.0) - 10.0).abs() < 1e-10);
    /// ```
    pub fn from_endpoint_derivatives(
        start: &[T; 4],
        end: &[T; 4],
        span: T,
    ) -> Result<Self, PolynomialError> {
        if !span.is_finite()
            || start
                .iter()
                .chain(end.iter())
                .any(|value| !value.is_finite())
        {
            return Err(PolynomialError::NonFinite);
        }
        if span <= T::ZERO {
            return Err(PolynomialError::SpanNotPositive);
        }

        let mut wanted = [T::ZERO; 8];
        for (slot, value) in wanted.iter_mut().zip(start.iter().chain(end.iter())) {
            *slot = *value;
        }
        let coefficients = endpoint_mapping_inverse(span)? * Vector::new(wanted);
        Ok(Self::new(coefficients.into_array()))
    }
}
