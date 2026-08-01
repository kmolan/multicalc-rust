//! The real roots of a polynomial.
//!
//! Up to the fourth power there are exact closed form solutions, so [`Polynomial::real_roots`] answers are
//! exact and differentiates cleanly. Past that no formula exists, so
//! [`Polynomial::count_real_roots`] says how many roots a range holds and
//! [`Polynomial::real_roots_in`] separates them and closes in by halving.
#![deny(clippy::indexing_slicing)]

use crate::error::PolynomialError;
use crate::polynomial::Polynomial;
use crate::scalar::Numeric;

/// The real roots of a polynomial, in increasing order.
///
/// Repeats are included, so a doubled root appears twice. `MAX_ROOTS` is one less than the
/// polynomial's coefficient count, which is the most real roots it can have.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealRoots<const MAX_ROOTS: usize, T: Numeric = f64> {
    values: [T; MAX_ROOTS],
    length: usize,
}

impl<const MAX_ROOTS: usize, T: Numeric> Default for RealRoots<MAX_ROOTS, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_ROOTS: usize, T: Numeric> RealRoots<MAX_ROOTS, T> {
    /// An empty set of roots.
    fn new() -> Self {
        Self {
            values: [T::ZERO; MAX_ROOTS],
            length: 0,
        }
    }

    /// Records a root. Anything past the capacity is dropped, which cannot happen while the
    /// capacity matches what the degree allows.
    fn push(&mut self, value: T) {
        if let Some(slot) = self.values.get_mut(self.length) {
            *slot = value;
            self.length += 1;
        }
    }

    /// Puts the roots in increasing order. There are only ever a handful, so working each one back
    /// into place is the whole story.
    fn sort_ascending(&mut self) {
        for position in 1..self.length {
            let mut index = position;
            while index > 0 {
                let previous = self.values.get(index - 1).copied().unwrap_or(T::ZERO);
                let current = self.values.get(index).copied().unwrap_or(T::ZERO);
                if previous <= current {
                    break;
                }
                self.values.swap(index - 1, index);
                index -= 1;
            }
        }
    }

    /// The roots, in increasing order.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.values.get(..self.length).unwrap_or(&[])
    }

    /// How many real roots there are.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether there are no real roots at all.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

/// Rejects a polynomial the formulas cannot answer for.
///
/// The zero test on the highest coefficient is exact rather than approximate: a coefficient that is
/// merely tiny still describes a polynomial of that degree, with one root racing off toward
/// infinity. A caller who means to drop the term should drop it and ask the next size down.
fn check<const COEFFICIENT_COUNT: usize, T: Numeric>(
    polynomial: &Polynomial<COEFFICIENT_COUNT, T>,
    highest: T,
) -> Result<(), PolynomialError> {
    if !polynomial.is_finite() {
        return Err(PolynomialError::NonFinite);
    }
    if highest == T::ZERO {
        return Err(PolynomialError::LeadingCoefficientZero);
    }
    Ok(())
}

impl<T: Numeric> Polynomial<2, T> {
    /// The real root, which a straight line always has exactly one of.
    ///
    /// Returns [`PolynomialError::LeadingCoefficientZero`] when the `x` coefficient is zero, and
    /// [`PolynomialError::NonFinite`] when any coefficient is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 2x - 6, whose root is 3
    /// let p = Polynomial::<2>::new([-6.0, 2.0]);
    /// let roots = p.real_roots().unwrap();
    /// assert_eq!(roots.len(), 1);
    /// for (found, expected) in roots.as_slice().iter().zip([3.0]) {
    ///     assert!((found - expected).abs() < 1e-12);
    /// }
    /// ```
    pub fn real_roots(&self) -> Result<RealRoots<1, T>, PolynomialError> {
        let &[constant, linear] = self.coefficients();
        check(self, linear)?;

        let mut roots = RealRoots::new();
        roots.push(-constant / linear);
        Ok(roots)
    }
}

impl<T: Numeric> Polynomial<3, T> {
    /// The real roots: two, one repeated pair, or none.
    ///
    /// Returns [`PolynomialError::LeadingCoefficientZero`] when the `x²` coefficient is zero, and
    /// [`PolynomialError::NonFinite`] when any coefficient is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // (x - 1)(x - 2) = 2 - 3x + x²
    /// let p = Polynomial::<3>::new([2.0, -3.0, 1.0]);
    /// let roots = p.real_roots().unwrap();
    /// assert_eq!(roots.len(), 2);
    /// for (found, expected) in roots.as_slice().iter().zip([1.0, 2.0]) {
    ///     assert!((found - expected).abs() < 1e-12);
    /// }
    /// ```
    pub fn real_roots(&self) -> Result<RealRoots<2, T>, PolynomialError> {
        let &[constant, linear, quadratic] = self.coefficients();
        check(self, quadratic)?;

        let mut roots = RealRoots::new();
        let discriminant = linear * linear - T::from_f64(4.0) * quadratic * constant;
        if discriminant < T::ZERO {
            return Ok(roots);
        }

        // Giving the square root the same sign as the `x` coefficient adds two like-signed numbers
        // instead of subtracting two close ones, which is where the usual formula loses digits when
        // one term dwarfs the other. The second root then comes from the first by division.
        let combined = -T::HALF * (linear + discriminant.sqrt().copysign(linear));
        if combined == T::ZERO {
            roots.push(T::ZERO);
            roots.push(T::ZERO);
        } else {
            roots.push(combined / quadratic);
            roots.push(constant / combined);
        }
        roots.sort_ascending();
        Ok(roots)
    }
}

impl<T: Numeric> Polynomial<4, T> {
    /// The real roots: three, or one, or two with one of them repeated.
    ///
    /// Returns [`PolynomialError::LeadingCoefficientZero`] when the `x³` coefficient is zero, and
    /// [`PolynomialError::NonFinite`] when any coefficient is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // (x - 1)(x - 2)(x - 3) = -6 + 11x - 6x² + x³
    /// let p = Polynomial::<4>::new([-6.0, 11.0, -6.0, 1.0]);
    /// let roots = p.real_roots().unwrap();
    /// assert_eq!(roots.len(), 3);
    /// for (found, expected) in roots.as_slice().iter().zip([1.0, 2.0, 3.0]) {
    ///     assert!((found - expected).abs() < 1e-10);
    /// }
    /// ```
    pub fn real_roots(&self) -> Result<RealRoots<3, T>, PolynomialError> {
        let &[constant, linear, quadratic, cubic] = self.coefficients();
        check(self, cubic)?;

        let three = T::THREE;
        let four = T::from_f64(4.0);
        let twenty_seven = T::from_f64(27.0);

        // Sliding the curve sideways by this much removes the squared term, leaving something of
        // the form y³ + (a number)·y + (a number), which the formulas below expect.
        let slide = -quadratic / (three * cubic);
        let reduced_linear =
            (three * cubic * linear - quadratic * quadratic) / (three * cubic * cubic);
        let reduced_constant = (T::TWO * quadratic * quadratic * quadratic
            - T::from_f64(9.0) * cubic * quadratic * linear
            + twenty_seven * cubic * cubic * constant)
            / (twenty_seven * cubic * cubic * cubic);

        let mut roots = RealRoots::new();
        if reduced_linear == T::ZERO {
            roots.push((-reduced_constant).cbrt() + slide);
        } else if four * reduced_linear.powi(3) + twenty_seven * reduced_constant * reduced_constant
            <= T::ZERO
        {
            // All three roots are real, and they sit a third of a turn apart on a circle, so one
            // angle gives all of them.
            let radius = T::TWO * (-reduced_linear / three).sqrt();
            let cosine = (three * reduced_constant) / (T::TWO * reduced_linear)
                * (-three / reduced_linear).sqrt();
            // At a repeated root this lands on 1 or -1, where rounding can carry it just past what
            // acos accepts, so pull it back in first.
            let angle = cosine.max(-T::ONE).min(T::ONE).acos();
            for step in 0..3 {
                let turn = T::TWO * T::PI * T::from_usize(step) / three;
                roots.push(radius * (angle / three - turn).cos() + slide);
            }
        } else {
            // One real root, from the sum of two cube roots.
            let half_constant = -reduced_constant * T::HALF;
            let spread = (reduced_constant * reduced_constant / four
                + reduced_linear.powi(3) / twenty_seven)
                .sqrt();
            roots.push((half_constant + spread).cbrt() + (half_constant - spread).cbrt() + slide);
        }
        roots.sort_ascending();
        Ok(roots)
    }
}

impl<T: Numeric> Polynomial<5, T> {
    /// The real roots: four, two, or none, with repeats included.
    ///
    /// Returns [`PolynomialError::LeadingCoefficientZero`] when the `x⁴` coefficient is zero, and
    /// [`PolynomialError::NonFinite`] when any coefficient is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // x(x - 1)(x - 2)(x - 4) = -8x + 14x² - 7x³ + x⁴
    /// let p = Polynomial::<5>::new([0.0, -8.0, 14.0, -7.0, 1.0]);
    /// let roots = p.real_roots().unwrap();
    /// assert_eq!(roots.len(), 4);
    /// for (found, expected) in roots.as_slice().iter().zip([0.0, 1.0, 2.0, 4.0]) {
    ///     assert!((found - expected).abs() < 1e-9);
    /// }
    /// ```
    pub fn real_roots(&self) -> Result<RealRoots<4, T>, PolynomialError> {
        let &[constant, linear, quadratic, cubic, quartic] = self.coefficients();
        check(self, quartic)?;

        let four = T::from_f64(4.0);
        let eight = T::from_f64(8.0);
        let two_hundred_fifty_six = T::from_f64(256.0);

        // As with the cubic, sliding sideways removes the cubed term, leaving only y⁴, y², y and a
        // number.
        let slide = -cubic / (four * quartic);
        let reduced_quadratic =
            (eight * quartic * quadratic - T::THREE * cubic * cubic) / (eight * quartic * quartic);
        let reduced_linear = (cubic * cubic * cubic - four * quartic * cubic * quadratic
            + eight * quartic * quartic * linear)
            / (eight * quartic * quartic * quartic);
        let reduced_constant = (-T::THREE * cubic.powi(4)
            + two_hundred_fifty_six * quartic.powi(3) * constant
            - T::from_f64(64.0) * quartic * quartic * cubic * linear
            + T::from_f64(16.0) * quartic * cubic * cubic * quadratic)
            / (two_hundred_fifty_six * quartic.powi(4));

        let mut roots = RealRoots::new();
        if reduced_linear == T::ZERO {
            // Only even powers are left, so solve for y² and take both square roots of each answer.
            let squares = Polynomial::<3, T>::new([reduced_constant, reduced_quadratic, T::ONE])
                .real_roots()?;
            for square in squares.as_slice() {
                if *square >= T::ZERO {
                    let root = square.sqrt();
                    roots.push(root + slide);
                    roots.push(-root + slide);
                }
            }
        } else {
            // The quartic splits into two quadratics once this helper cubic is solved. Its lowest
            // coefficient is negative, so it always has a positive root, and the largest root is
            // the one that works.
            let helper = Polynomial::<4, T>::new([
                -reduced_linear * reduced_linear,
                T::TWO * reduced_quadratic * reduced_quadratic - eight * reduced_constant,
                eight * reduced_quadratic,
                eight,
            ])
            .real_roots()?;
            let largest = helper.as_slice().last().copied().unwrap_or(T::ZERO);

            let separation = (T::TWO * largest).sqrt();
            let shared = reduced_quadratic * T::HALF + largest;
            let apart = reduced_linear / (T::TWO * separation);

            for factor in [
                Polynomial::<3, T>::new([shared + apart, -separation, T::ONE]),
                Polynomial::<3, T>::new([shared - apart, separation, T::ONE]),
            ] {
                let factor_roots = factor.real_roots()?;
                for root in factor_roots.as_slice() {
                    roots.push(*root + slide);
                }
            }
        }
        roots.sort_ascending();
        Ok(roots)
    }
}

/// What is left of `dividend` after taking out as many copies of `divisor` as fit.
///
/// Both sides and the answer are the same size, which is what lets a whole chain of these sit in
/// one array. The answer's coefficients at and above the divisor's degree are zero.
fn remainder<const COEFFICIENT_COUNT: usize, T: Numeric>(
    dividend: &Polynomial<COEFFICIENT_COUNT, T>,
    divisor: &Polynomial<COEFFICIENT_COUNT, T>,
) -> Option<Polynomial<COEFFICIENT_COUNT, T>> {
    let divisor_degree = divisor.degree()?;
    let divisor_highest = divisor.coefficient(divisor_degree)?;

    let mut working = *dividend.coefficients();
    let mut power = COEFFICIENT_COUNT;
    while power > divisor_degree {
        power -= 1;
        let highest = working.get(power).copied().unwrap_or(T::ZERO);
        if highest == T::ZERO {
            continue;
        }
        let share = highest / divisor_highest;
        let offset = power - divisor_degree;
        for (divisor_power, divisor_coefficient) in divisor.coefficients().iter().enumerate() {
            if let Some(slot) = working.get_mut(offset + divisor_power) {
                *slot -= share * *divisor_coefficient;
            }
        }
        // The subtraction is meant to clear this power exactly; rounding can leave a speck.
        if let Some(slot) = working.get_mut(power) {
            *slot = T::ZERO;
        }
    }
    Some(Polynomial::new(working))
}

/// How many times the chain's values change sign at `at`, skipping any that land exactly on zero.
fn sign_changes<const COEFFICIENT_COUNT: usize, T: Numeric>(
    chain: &[Polynomial<COEFFICIENT_COUNT, T>; COEFFICIENT_COUNT],
    length: usize,
    at: T,
) -> usize {
    let mut changes = 0;
    let mut previous_positive = None;
    for polynomial in chain.iter().take(length) {
        let value = polynomial.evaluate(at);
        if value == T::ZERO {
            continue;
        }
        let positive = value > T::ZERO;
        if previous_positive.is_some_and(|previous| previous != positive) {
            changes += 1;
        }
        previous_positive = Some(positive);
    }
    changes
}

impl<const COEFFICIENT_COUNT: usize, T: Numeric> Polynomial<COEFFICIENT_COUNT, T> {
    /// A number every real root is smaller than in size, so `[-bound, bound]` holds all of them.
    ///
    /// This hands any search that needs a starting range one for free, with no work beyond a pass
    /// over the coefficients. Returns [`PolynomialError::LeadingCoefficientZero`] when every
    /// coefficient is zero, and [`PolynomialError::NonFinite`] when any of them is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // (x - 1)(x - 2) = 2 - 3x + x²; both roots sit inside [-4, 4].
    /// let p = Polynomial::<3>::new([2.0, -3.0, 1.0]);
    /// assert!((p.cauchy_root_bound().unwrap() - 4.0).abs() < 1e-12);
    /// ```
    pub fn cauchy_root_bound(&self) -> Result<T, PolynomialError> {
        if !self.is_finite() {
            return Err(PolynomialError::NonFinite);
        }
        let degree = self
            .degree()
            .ok_or(PolynomialError::LeadingCoefficientZero)?;
        let highest = self
            .coefficient(degree)
            .ok_or(PolynomialError::LeadingCoefficientZero)?;

        let mut largest = T::ZERO;
        for coefficient in self.coefficients().iter().take(degree) {
            largest = largest.max((*coefficient / highest).abs());
        }
        Ok(T::ONE + largest)
    }

    /// Builds the chain of polynomials whose sign changes count roots.
    ///
    /// It starts with the polynomial and its derivative, and each one after that is what is left of
    /// the pair before it, negated. Every entry has a lower degree than the one before, so the
    /// chain always finishes within `COEFFICIENT_COUNT` entries.
    fn root_counting_chain(
        &self,
    ) -> Result<([Polynomial<COEFFICIENT_COUNT, T>; COEFFICIENT_COUNT], usize), PolynomialError>
    {
        if !self.is_finite() {
            return Err(PolynomialError::NonFinite);
        }
        let degree = self
            .degree()
            .ok_or(PolynomialError::LeadingCoefficientZero)?;

        let mut chain = [Polynomial::<COEFFICIENT_COUNT, T>::zeros(); COEFFICIENT_COUNT];
        let mut length = 0;
        // The polynomial itself starts the chain.
        if let Some(slot) = chain.get_mut(0) {
            *slot = *self;
            length = 1;
        }
        // Its derivative comes next, unless it is a flat line with nothing to differentiate.
        if degree >= 1
            && let Some(slot) = chain.get_mut(1)
        {
            *slot = self.derivative();
            length = 2;
        }

        while length < COEFFICIENT_COUNT {
            // Each new entry is what is left over from dividing the two before it, with the sign
            // turned around.
            let previous = chain.get(length - 2).copied().unwrap_or_default();
            let current = chain.get(length - 1).copied().unwrap_or_default();
            let Some(next) = remainder(&previous, &current) else {
                break;
            };
            // Nothing left over means the chain is finished.
            if next.is_zero() {
                break;
            }
            if let Some(slot) = chain.get_mut(length) {
                *slot = -next;
            }
            length += 1;
        }
        Ok((chain, length))
    }

    /// How many real roots lie in the range, counting a repeated root once.
    ///
    /// This is a real difference from the closed-form finders, which report a doubled root twice.
    /// It also takes no iteration at all: the answer comes from comparing sign changes at the two
    /// ends. The range takes in `upper` but not `lower`, so a root sitting exactly on `lower` is
    /// not counted.
    ///
    /// Returns [`PolynomialError::LeadingCoefficientZero`] when every coefficient is zero, and
    /// [`PolynomialError::NonFinite`] when any of them is infinite or NaN.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // (x - 2)²(x + 1) = 4 - 3x² + x³, with a doubled root at 2 and a single one at -1.
    /// let p = Polynomial::<4>::new([4.0, 0.0, -3.0, 1.0]);
    /// assert_eq!(p.count_real_roots(-5.0, 5.0).unwrap(), 2);
    /// ```
    pub fn count_real_roots(&self, lower: T, upper: T) -> Result<usize, PolynomialError> {
        let (chain, length) = self.root_counting_chain()?;
        Ok(sign_changes(&chain, length, lower).saturating_sub(sign_changes(&chain, length, upper)))
    }

    /// The real roots in the range, each narrowed until it is within `tolerance`.
    ///
    /// Roots are separated first, by splitting the range until each piece holds exactly one, and
    /// then each is closed in on by halving. A repeated root appears once, matching
    /// [`count_real_roots`](Self::count_real_roots). The range takes in `upper` but not `lower`.
    ///
    /// `maximum_bisections` caps the total halving work; 200 is a sensible default, and running out
    /// gives [`PolynomialError::DidNotConverge`] rather than looping. The chain this needs is
    /// `COEFFICIENT_COUNT` squared numbers of working memory — 512 bytes at degree 7, about 3.4 KB
    /// at degree 20 — all of it on the stack.
    ///
    /// Each root here is a number arrived at by halving, so it carries no derivative information:
    /// differentiating a root against a coefficient is a question for the closed-form finders, or
    /// for the implicit function theorem, not for this call.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // (x + 1)(x - 1)(x - 3) = 3 - x - 3x² + x³
    /// let p = Polynomial::<4>::new([3.0, -1.0, -3.0, 1.0]);
    /// let roots = p.real_roots_in(-5.0, 5.0, 1e-10, 200).unwrap();
    /// assert_eq!(roots.len(), 3);
    /// for (found, expected) in roots.as_slice().iter().zip([-1.0, 1.0, 3.0]) {
    ///     assert!((found - expected).abs() < 1e-9);
    /// }
    /// ```
    pub fn real_roots_in(
        &self,
        lower: T,
        upper: T,
        tolerance: T,
        maximum_bisections: usize,
    ) -> Result<RealRoots<COEFFICIENT_COUNT, T>, PolynomialError> {
        let (chain, length) = self.root_counting_chain()?;
        // How many roots sit between two points.
        let count_between = |from: T, to: T| {
            sign_changes(&chain, length, from).saturating_sub(sign_changes(&chain, length, to))
        };

        // Ranges still to look at. Only ones holding a root ever go in, and they never overlap, so
        // there are never more of them than the polynomial has roots.
        let mut pending = [(T::ZERO, T::ZERO); COEFFICIENT_COUNT];
        let mut pending_length = 0;
        // Start with the whole range, unless it is empty of roots.
        if count_between(lower, upper) > 0
            && let Some(slot) = pending.get_mut(0)
        {
            *slot = (lower, upper);
            pending_length = 1;
        }

        let mut roots = RealRoots::new();
        let mut steps = 0;
        while pending_length > 0 {
            // Take the range added most recently.
            pending_length -= 1;
            let (range_lower, range_upper) = pending
                .get(pending_length)
                .copied()
                .unwrap_or((T::ZERO, T::ZERO));

            // One root in here on its own, so close in on it.
            if count_between(range_lower, range_upper) == 1 {
                let (mut low, mut high) = (range_lower, range_upper);
                while high - low > tolerance {
                    if steps >= maximum_bisections {
                        return Err(PolynomialError::DidNotConverge { steps });
                    }
                    steps += 1;
                    // Cut the range in two and keep whichever half still holds the root.
                    let middle = (low + high) * T::HALF;
                    if count_between(low, middle) == 1 {
                        high = middle;
                    } else {
                        low = middle;
                    }
                }
                // Near enough now, so take the middle as the answer.
                roots.push((low + high) * T::HALF);
                continue;
            }

            // Several roots in here, so cut the range in two and come back to each half.
            if steps >= maximum_bisections {
                return Err(PolynomialError::DidNotConverge { steps });
            }
            steps += 1;
            let middle = (range_lower + range_upper) * T::HALF;
            for (piece_lower, piece_upper) in [(range_lower, middle), (middle, range_upper)] {
                // An empty half is dropped here rather than pushed and thrown away later, which is
                // what keeps the waiting list short enough to fit.
                if count_between(piece_lower, piece_upper) > 0
                    && let Some(slot) = pending.get_mut(pending_length)
                {
                    *slot = (piece_lower, piece_upper);
                    pending_length += 1;
                }
            }
        }
        roots.sort_ascending();
        Ok(roots)
    }
}
