//! A curve made of polynomial pieces, one after another.
#![deny(clippy::indexing_slicing)]

use crate::error::PolynomialError;
use crate::linear_algebra::Vector;
use crate::polynomial::Polynomial;
use crate::scalar::Numeric;

/// A curve in `DIMENSION` dimensions made of polynomial pieces laid end to end.
///
/// Each piece runs on its own clock, from 0 at its start to 1 at its end, and carries a **span**
/// saying how much of the shared parameter it covers. Keeping every piece on the same 0-to-1 clock
/// is what stops the numbers in any solve from spanning many powers of the piece width, which is
/// what makes a high-degree piece well behaved. Derivatives against the shared parameter come from
/// dividing by the span once per order.
///
/// The word **span** is used rather than duration because the shared parameter need not be time. In
/// [`motion`](crate::motion), where it is, the same number is called a duration.
///
/// Evaluation is fixed work with no allocation, so it is safe in a tight loop. A parameter before
/// the start or past the end clamps to that end rather than running off the curve.
///
/// ```
/// use multicalc::{PiecewisePolynomial, Polynomial};
///
/// // One axis: climbing to 1 over two units of the parameter, then on to 3 over one more.
/// let first = [Polynomial::<2>::new([0.0, 1.0])];
/// let second = [Polynomial::<2>::new([1.0, 2.0])];
/// let curve =
///     PiecewisePolynomial::<2, 2, 1>::try_from_pieces(&[first, second], &[2.0, 1.0]).unwrap();
///
/// assert!((curve.total_span() - 3.0).abs() < 1e-12);
///
/// let [handover] = curve.evaluate(2.0).unwrap().into_array();
/// assert!((handover - 1.0).abs() < 1e-12);
///
/// // Past the end it holds at the last value.
/// let [beyond] = curve.evaluate(9.0).unwrap().into_array();
/// assert!((beyond - 3.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PiecewisePolynomial<
    const MAX_PIECES: usize,
    const COEFFICIENTS_PER_PIECE: usize,
    const DIMENSION: usize,
    T: Numeric = f64,
> {
    pieces: [[Polynomial<COEFFICIENTS_PER_PIECE, T>; DIMENSION]; MAX_PIECES],
    spans: [T; MAX_PIECES],
    length: usize,
}

impl<
    const MAX_PIECES: usize,
    const COEFFICIENTS_PER_PIECE: usize,
    const DIMENSION: usize,
    T: Numeric,
> PiecewisePolynomial<MAX_PIECES, COEFFICIENTS_PER_PIECE, DIMENSION, T>
{
    /// Builds a curve from its pieces and how much of the parameter each one covers.
    ///
    /// There must be one span per piece. Returns [`PolynomialError::Empty`] when either list is
    /// empty, [`PolynomialError::CapacityExceeded`] when the lists disagree in length or there are
    /// more pieces than fit, and [`PolynomialError::SpanNotPositive`] when a span is zero, negative,
    /// or not a number.
    pub fn try_from_pieces(
        pieces: &[[Polynomial<COEFFICIENTS_PER_PIECE, T>; DIMENSION]],
        spans: &[T],
    ) -> Result<Self, PolynomialError> {
        if pieces.is_empty() || spans.is_empty() {
            return Err(PolynomialError::Empty);
        }
        if pieces.len() != spans.len() || pieces.len() > MAX_PIECES {
            return Err(PolynomialError::CapacityExceeded);
        }
        for span in spans {
            if !span.is_finite() || *span <= T::ZERO {
                return Err(PolynomialError::SpanNotPositive);
            }
        }

        let mut curve = Self {
            pieces: [[Polynomial::zeros(); DIMENSION]; MAX_PIECES],
            spans: [T::ZERO; MAX_PIECES],
            length: pieces.len(),
        };
        for (slot, piece) in curve.pieces.iter_mut().zip(pieces.iter()) {
            *slot = *piece;
        }
        for (slot, span) in curve.spans.iter_mut().zip(spans.iter()) {
            *slot = *span;
        }
        Ok(curve)
    }

    /// How many pieces the curve has.
    #[inline]
    #[must_use]
    pub fn piece_count(&self) -> usize {
        self.length
    }

    /// Whether the curve has no pieces at all.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// How much of the parameter the whole curve covers.
    #[inline]
    #[must_use]
    pub fn total_span(&self) -> T {
        let mut total = T::ZERO;
        for span in self.spans.iter().take(self.length) {
            total += *span;
        }
        total
    }

    /// How much of the parameter one piece covers, or `None` past the last piece.
    #[inline]
    #[must_use]
    pub fn span(&self, piece: usize) -> Option<T> {
        if piece >= self.length {
            return None;
        }
        self.spans.get(piece).copied()
    }

    /// One axis of one piece, or `None` past the last piece or axis.
    #[inline]
    #[must_use]
    pub fn piece_polynomial(
        &self,
        piece: usize,
        axis: usize,
    ) -> Option<&Polynomial<COEFFICIENTS_PER_PIECE, T>> {
        if piece >= self.length {
            return None;
        }
        self.pieces.get(piece)?.get(axis)
    }

    /// Which piece a parameter falls in, and how far along that piece it sits as a number from 0
    /// to 1.
    ///
    /// Before the start it gives the very beginning, and at or past the end the very end, so the
    /// curve holds rather than running off. `None` only when there are no pieces.
    fn locate(&self, parameter: T) -> Option<(usize, T)> {
        if self.length == 0 {
            return None;
        }
        if parameter <= T::ZERO {
            return Some((0, T::ZERO));
        }
        // Walk the pieces, keeping track of how much of the parameter is already behind.
        let mut covered = T::ZERO;
        for (index, span) in self.spans.iter().take(self.length).enumerate() {
            if parameter < covered + *span {
                return Some((index, (parameter - covered) / *span));
            }
            covered += *span;
        }
        Some((self.length - 1, T::ONE))
    }

    /// The point on the curve at `parameter`.
    ///
    /// Parameters outside the curve clamp to its ends. Returns [`PolynomialError::Empty`] when
    /// there are no pieces, which is the only way this can fail.
    pub fn evaluate(&self, parameter: T) -> Result<Vector<DIMENSION, T>, PolynomialError> {
        let (piece, along) = self.locate(parameter).ok_or(PolynomialError::Empty)?;
        let polynomials = self.pieces.get(piece).ok_or(PolynomialError::Empty)?;

        let mut values = [T::ZERO; DIMENSION];
        for (slot, polynomial) in values.iter_mut().zip(polynomials.iter()) {
            *slot = polynomial.evaluate(along);
        }
        Ok(Vector::new(values))
    }

    /// The point at `parameter` together with the first `ORDER_COUNT - 1` derivatives there, all
    /// measured against the shared parameter.
    ///
    /// This is what a tracker calls to get position, velocity and acceleration in one go. Each
    /// piece works on its own 0-to-1 clock, so each order is divided by the piece's span once more
    /// than the order below it. Parameters outside the curve clamp to its ends; returns
    /// [`PolynomialError::Empty`] when there are no pieces.
    pub fn evaluate_with_derivatives<const ORDER_COUNT: usize>(
        &self,
        parameter: T,
    ) -> Result<[Vector<DIMENSION, T>; ORDER_COUNT], PolynomialError> {
        let (piece, along) = self.locate(parameter).ok_or(PolynomialError::Empty)?;
        let polynomials = self.pieces.get(piece).ok_or(PolynomialError::Empty)?;
        let span = self
            .spans
            .get(piece)
            .copied()
            .ok_or(PolynomialError::Empty)?;

        // Every order for one axis comes out of a single sweep, so do that per axis first.
        let mut per_axis = [[T::ZERO; ORDER_COUNT]; DIMENSION];
        for (slot, polynomial) in per_axis.iter_mut().zip(polynomials.iter()) {
            *slot = polynomial.evaluate_with_derivatives(along);
        }

        // Then gather them by order, converting each from the piece's clock to the shared one. The
        // running product keeps that to one division per order rather than one per axis.
        let mut result = [Vector::<DIMENSION, T>::zeros(); ORDER_COUNT];
        let mut span_raised = T::ONE;
        for (order, vector) in result.iter_mut().enumerate() {
            if order > 0 {
                span_raised *= span;
            }
            let converted = T::ONE / span_raised;

            let mut values = [T::ZERO; DIMENSION];
            for (slot, orders) in values.iter_mut().zip(per_axis.iter()) {
                *slot = orders.get(order).copied().unwrap_or(T::ZERO) * converted;
            }
            *vector = Vector::new(values);
        }
        Ok(result)
    }

    /// The area under each axis of the curve between two parameters.
    ///
    /// Unlike evaluation, this does **not** treat the curve as holding its end values forever: the
    /// range is trimmed to the curve first, so asking past either end adds nothing rather than
    /// adding the end value times the overshoot. Integrating from before the start to past the end
    /// therefore gives the whole curve's area, whatever numbers are passed.
    ///
    /// Bounds the wrong way round give the answer negated, matching
    /// [`Polynomial::definite_integral`]. Returns [`PolynomialError::Empty`] when there are no
    /// pieces, which is the only way this can fail.
    ///
    /// ```
    /// use multicalc::{PiecewisePolynomial, Polynomial};
    ///
    /// // Climbing to 1 over two units of the parameter, then on to 3 over one more.
    /// let first = [Polynomial::<2>::new([0.0, 1.0])];
    /// let second = [Polynomial::<2>::new([1.0, 2.0])];
    /// let curve =
    ///     PiecewisePolynomial::<2, 2, 1>::try_from_pieces(&[first, second], &[2.0, 1.0]).unwrap();
    ///
    /// // The first piece covers half of a 2-by-1 rectangle, the second covers 2.
    /// let [whole] = curve.definite_integral(0.0, 3.0).unwrap().into_array();
    /// assert!((whole - 3.0).abs() < 1e-12);
    ///
    /// // Part of one piece.
    /// let [part] = curve.definite_integral(0.0, 1.0).unwrap().into_array();
    /// assert!((part - 0.25).abs() < 1e-12);
    ///
    /// // Reaching past the end adds nothing.
    /// let [beyond] = curve.definite_integral(0.0, 99.0).unwrap().into_array();
    /// assert!((beyond - 3.0).abs() < 1e-12);
    /// ```
    pub fn definite_integral(
        &self,
        lower: T,
        upper: T,
    ) -> Result<Vector<DIMENSION, T>, PolynomialError> {
        if self.length == 0 {
            return Err(PolynomialError::Empty);
        }
        // Work low to high and put the sign back at the end.
        let flipped = upper < lower;
        let (wanted_start, wanted_end) = if flipped {
            (upper, lower)
        } else {
            (lower, upper)
        };

        let total = self.total_span();
        let wanted_start = wanted_start.max(T::ZERO).min(total);
        let wanted_end = wanted_end.max(T::ZERO).min(total);

        let mut areas = [T::ZERO; DIMENSION];
        let mut covered = T::ZERO;
        for (polynomials, span) in self.pieces.iter().zip(self.spans.iter()).take(self.length) {
            let piece_start = covered;
            covered += *span;

            // The part of this piece the wanted range actually reaches.
            let start = wanted_start.max(piece_start);
            let end = wanted_end.min(covered);
            if end <= start {
                continue;
            }

            // Move those onto the piece's own clock. Working there measures the area against that
            // clock too, so multiplying by the span puts it back on the shared parameter.
            let along_start = (start - piece_start) / *span;
            let along_end = (end - piece_start) / *span;
            for (slot, polynomial) in areas.iter_mut().zip(polynomials.iter()) {
                *slot += polynomial.definite_integral(along_start, along_end) * *span;
            }
        }

        if flipped {
            for slot in areas.iter_mut() {
                *slot = -*slot;
            }
        }
        Ok(Vector::new(areas))
    }

    /// The curve of the slope, measured against the shared parameter.
    ///
    /// The pieces and their spans stay as they are; only the polynomials change. Evaluating this is
    /// the same as asking [`evaluate_with_derivatives`](Self::evaluate_with_derivatives) for one
    /// order up, so use whichever reads better — this when the slope is wanted as a curve in its own
    /// right, that when the value and the slope are both wanted at one parameter.
    ///
    /// Where the original has a corner, the slope jumps at that join, and this reports the piece the
    /// parameter falls in. Outside the curve it holds at the end piece's slope rather than dropping
    /// to zero, matching how evaluation clamps.
    ///
    /// ```
    /// use multicalc::{PiecewisePolynomial, Polynomial};
    ///
    /// // Climbing to 1 over two units of the parameter, so the slope along there is 0.5.
    /// let first = [Polynomial::<2>::new([0.0, 1.0])];
    /// let second = [Polynomial::<2>::new([1.0, 2.0])];
    /// let curve =
    ///     PiecewisePolynomial::<2, 2, 1>::try_from_pieces(&[first, second], &[2.0, 1.0]).unwrap();
    ///
    /// let slope = curve.derivative();
    /// let [early] = slope.evaluate(1.0).unwrap().into_array();
    /// assert!((early - 0.5).abs() < 1e-12);
    ///
    /// // The second piece covers twice the climb in half the parameter, so it is four times as steep.
    /// let [late] = slope.evaluate(2.5).unwrap().into_array();
    /// assert!((late - 2.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn derivative(&self) -> Self {
        self.nth_derivative(1)
    }

    /// The curve left after taking the slope `order` times, measured against the shared parameter.
    ///
    /// An `order` of zero gives the curve back. An `order` at or above the coefficients per piece
    /// gives a curve that is zero everywhere, since by then every term has been differentiated away.
    ///
    /// ```
    /// use multicalc::{PiecewisePolynomial, Polynomial};
    ///
    /// // One piece: 3τ² over a span of two, so the bend against the parameter is 6/4.
    /// let piece = [Polynomial::<3>::new([0.0, 0.0, 3.0])];
    /// let curve = PiecewisePolynomial::<1, 3, 1>::try_from_pieces(&[piece], &[2.0]).unwrap();
    ///
    /// let [bend] = curve.nth_derivative(2).evaluate(1.0).unwrap().into_array();
    /// assert!((bend - 1.5).abs() < 1e-12);
    /// assert!(curve.nth_derivative(3).evaluate(1.0).unwrap().into_array()[0].abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn nth_derivative(&self, order: usize) -> Self {
        let mut result = *self;
        for (piece, span) in result
            .pieces
            .iter_mut()
            .zip(self.spans.iter())
            .take(self.length)
        {
            // Differentiating works against the piece's own clock, so undo one span per order to
            // measure against the shared parameter instead. Past the coefficients per piece there is
            // nothing left to differentiate, so stop raising the span there rather than let it run
            // away on a result that is already zero.
            let mut span_raised = T::ONE;
            for _ in 0..order.min(COEFFICIENTS_PER_PIECE) {
                span_raised *= *span;
            }
            let converted = T::ONE / span_raised;

            for polynomial in piece.iter_mut() {
                *polynomial = polynomial.nth_derivative(order).scale(converted);
            }
        }
        result
    }
}
