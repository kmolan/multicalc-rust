//! Polynomials in several variables, held as a list of terms.
#![deny(clippy::indexing_slicing)]

use crate::error::PolynomialError;
use crate::linear_algebra::Vector;
use crate::polynomial::Polynomial;
use crate::scalar::Numeric;

/// Raises `value` to a whole-number power, for exponents that fit the underlying routine.
fn raise<T: Numeric>(value: T, exponent: u32) -> T {
    value.powi(i32::try_from(exponent).unwrap_or(i32::MAX))
}

/// One term of a polynomial in several variables: a number, and a power for each variable.
///
/// The powers are held in order, one per variable, so `[2, 3]` with two variables means the first
/// variable squared times the second cubed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultivariateTerm<const VARIABLES: usize, T: Numeric = f64> {
    coefficient: T,
    exponents: [u32; VARIABLES],
}

impl<const VARIABLES: usize, T: Numeric> MultivariateTerm<VARIABLES, T> {
    /// A term with the given number and powers.
    ///
    /// ```
    /// use multicalc::MultivariateTerm;
    ///
    /// // 2.5·x²·y³
    /// let term = MultivariateTerm::new(2.5, [2, 3]);
    /// assert_eq!(term.exponents(), &[2, 3]);
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(coefficient: T, exponents: [u32; VARIABLES]) -> Self {
        Self {
            coefficient,
            exponents,
        }
    }

    /// The number the powers are multiplied by.
    #[inline]
    #[must_use]
    pub fn coefficient(&self) -> T {
        self.coefficient
    }

    /// The power each variable is raised to.
    #[inline]
    #[must_use]
    pub fn exponents(&self) -> &[u32; VARIABLES] {
        &self.exponents
    }
}

/// A polynomial in several variables, held as a list of terms.
///
/// Each term carries its own powers, so there is no ordering to remember and no index to work out:
/// storage grows with how many terms there are rather than with the degree. A polynomial in three
/// variables with eight terms is under 200 bytes, against 1728 for every coefficient up to the fifth
/// power.
///
/// Only whole-number powers are held, and that is what keeps every operation closed: the sum,
/// product, partial derivative, partial antiderivative and substitution of one of these is another
/// one. Square roots or sines would need something this type cannot represent.
///
/// ```
/// use multicalc::{MultivariatePolynomial, MultivariateTerm};
///
/// // 3x²y + 2xy - 1
/// let p = MultivariatePolynomial::<2, 3>::try_from_terms(&[
///     MultivariateTerm::new(3.0, [2, 1]),
///     MultivariateTerm::new(2.0, [1, 1]),
///     MultivariateTerm::new(-1.0, [0, 0]),
/// ])
/// .unwrap();
///
/// // At x = 2, y = 3: 3·4·3 + 2·2·3 - 1 = 47
/// assert!((p.evaluate(&[2.0, 3.0]) - 47.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultivariatePolynomial<const VARIABLES: usize, const MAX_TERMS: usize, T: Numeric = f64>
{
    terms: [MultivariateTerm<VARIABLES, T>; MAX_TERMS],
    length: usize,
}

impl<const VARIABLES: usize, const MAX_TERMS: usize, T: Numeric> Default
    for MultivariatePolynomial<VARIABLES, MAX_TERMS, T>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const VARIABLES: usize, const MAX_TERMS: usize, T: Numeric>
    MultivariatePolynomial<VARIABLES, MAX_TERMS, T>
{
    /// A polynomial with no terms, which is zero everywhere.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            terms: [MultivariateTerm::new(T::ZERO, [0; VARIABLES]); MAX_TERMS],
            length: 0,
        }
    }

    /// Builds a polynomial from a list of terms.
    ///
    /// Returns [`PolynomialError::CapacityExceeded`] when there are more terms than fit, and
    /// [`PolynomialError::NonFinite`] when a term's number is infinite or NaN.
    pub fn try_from_terms(
        terms: &[MultivariateTerm<VARIABLES, T>],
    ) -> Result<Self, PolynomialError> {
        let mut result = Self::new();
        for term in terms {
            result.push(*term)?;
        }
        Ok(result)
    }

    /// Builds a polynomial holding exactly these terms, with room for no more.
    ///
    /// This is what the [`multivariate_polynomial!`](crate::multivariate_polynomial) macro uses,
    /// since taking the terms as an array is what lets the capacity follow from how many were
    /// written. Returns [`PolynomialError::NonFinite`] when a term's number is infinite or NaN.
    pub fn try_from_array(
        terms: [MultivariateTerm<VARIABLES, T>; MAX_TERMS],
    ) -> Result<Self, PolynomialError> {
        if terms.iter().any(|term| !term.coefficient.is_finite()) {
            return Err(PolynomialError::NonFinite);
        }
        Ok(Self {
            terms,
            length: MAX_TERMS,
        })
    }

    /// Appends a term.
    ///
    /// Returns [`PolynomialError::CapacityExceeded`] when the polynomial is already full, and
    /// [`PolynomialError::NonFinite`] when the term's number is infinite or NaN.
    pub fn push(&mut self, term: MultivariateTerm<VARIABLES, T>) -> Result<(), PolynomialError> {
        if !term.coefficient.is_finite() {
            return Err(PolynomialError::NonFinite);
        }
        match self.terms.get_mut(self.length) {
            Some(slot) => {
                *slot = term;
                self.length += 1;
                Ok(())
            }
            None => Err(PolynomialError::CapacityExceeded),
        }
    }

    /// The terms, in the order they were added.
    #[inline]
    #[must_use]
    pub fn terms(&self) -> &[MultivariateTerm<VARIABLES, T>] {
        self.terms.get(..self.length).unwrap_or(&[])
    }

    /// How many terms there are.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Whether there are no terms at all, which means the polynomial is zero everywhere.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Whether every term's number is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.terms().iter().all(|term| term.coefficient.is_finite())
    }

    /// The value at a point.
    ///
    /// The work is the number of terms times the number of variables, with no allocation, so it is
    /// bounded and safe in a tight loop.
    ///
    /// ```
    /// use multicalc::{MultivariatePolynomial, MultivariateTerm};
    ///
    /// // x²y at x = 3, y = 2 is 18
    /// let p = MultivariatePolynomial::<2, 1>::try_from_terms(&[MultivariateTerm::new(1.0, [2, 1])])
    ///     .unwrap();
    /// assert!((p.evaluate(&[3.0, 2.0]) - 18.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn evaluate(&self, variables: &[T; VARIABLES]) -> T {
        let mut total = T::ZERO;
        for term in self.terms() {
            let mut product = term.coefficient;
            for (value, exponent) in variables.iter().zip(term.exponents.iter()) {
                product *= raise(*value, *exponent);
            }
            total += product;
        }
        total
    }

    /// The polynomial left after differentiating with respect to one variable.
    ///
    /// Terms that do not mention the variable disappear, so the result never has more terms than
    /// the original. Returns [`PolynomialError::VariableOutOfRange`] when the index names no
    /// variable.
    ///
    /// ```
    /// use multicalc::{MultivariatePolynomial, MultivariateTerm};
    ///
    /// // 3x²y differentiated in x is 6xy
    /// let p = MultivariatePolynomial::<2, 1>::try_from_terms(&[MultivariateTerm::new(3.0, [2, 1])])
    ///     .unwrap();
    /// let slope = p.partial_derivative(0).unwrap();
    /// assert_eq!(slope.terms()[0].exponents(), &[1, 1]);
    /// assert!((slope.terms()[0].coefficient() - 6.0).abs() < 1e-12);
    /// ```
    pub fn partial_derivative(&self, variable: usize) -> Result<Self, PolynomialError> {
        if variable >= VARIABLES {
            return Err(PolynomialError::VariableOutOfRange);
        }
        let mut result = Self::new();
        for term in self.terms() {
            let exponent = term.exponents.get(variable).copied().unwrap_or(0);
            // A variable the term does not mention differentiates the whole term away.
            if exponent == 0 {
                continue;
            }
            let mut exponents = term.exponents;
            if let Some(slot) = exponents.get_mut(variable) {
                *slot = exponent - 1;
            }
            let coefficient = term.coefficient * T::from_usize(exponent as usize);
            result.push(MultivariateTerm::new(coefficient, exponents))?;
        }
        Ok(result)
    }

    /// The slope in every variable at once, at a point.
    ///
    /// This exists next to [`partial_derivative`](Self::partial_derivative) because it takes one
    /// pass over the terms rather than building a separate polynomial per variable and evaluating
    /// each.
    pub fn gradient_at(&self, variables: &[T; VARIABLES]) -> Vector<VARIABLES, T> {
        let mut totals = [T::ZERO; VARIABLES];
        for term in self.terms() {
            for (variable, exponent) in term.exponents.iter().enumerate() {
                if *exponent == 0 {
                    continue;
                }
                // Differentiating brings the power down in front and drops it by one, leaving every
                // other variable as it was.
                let mut product = term.coefficient * T::from_usize(*exponent as usize);
                for (other, (value, power)) in
                    variables.iter().zip(term.exponents.iter()).enumerate()
                {
                    let power = if other == variable {
                        *power - 1
                    } else {
                        *power
                    };
                    product *= raise(*value, power);
                }
                if let Some(slot) = totals.get_mut(variable) {
                    *slot += product;
                }
            }
        }
        Vector::new(totals)
    }

    /// The polynomial that differentiates back to this one in the given variable, with no constant
    /// added.
    ///
    /// Returns [`PolynomialError::VariableOutOfRange`] when the index names no variable, and
    /// [`PolynomialError::DegreeOverflow`] when a power would run past what a term can hold.
    pub fn partial_antiderivative(&self, variable: usize) -> Result<Self, PolynomialError> {
        if variable >= VARIABLES {
            return Err(PolynomialError::VariableOutOfRange);
        }
        let mut result = Self::new();
        for term in self.terms() {
            let exponent = term.exponents.get(variable).copied().unwrap_or(0);
            let raised = exponent
                .checked_add(1)
                .ok_or(PolynomialError::DegreeOverflow)?;
            let mut exponents = term.exponents;
            if let Some(slot) = exponents.get_mut(variable) {
                *slot = raised;
            }
            let coefficient = term.coefficient / T::from_usize(raised as usize);
            result.push(MultivariateTerm::new(coefficient, exponents))?;
        }
        Ok(result)
    }

    /// The polynomial left after fixing one variable at a value.
    ///
    /// The result still names the same variables — the fixed one simply appears with a power of
    /// zero everywhere, so evaluating it ignores whatever is passed for that slot. Terms that match
    /// once the variable is gone are merged, so this usually leaves fewer terms than it started
    /// with.
    ///
    /// Crossing to the dense [`Polynomial`] is a separate step and needs a polynomial that already
    /// names one variable: see [`to_univariate`](MultivariatePolynomial::to_univariate).
    ///
    /// Returns [`PolynomialError::VariableOutOfRange`] when the index names no variable.
    pub fn substitute(&self, variable: usize, value: T) -> Result<Self, PolynomialError> {
        if variable >= VARIABLES {
            return Err(PolynomialError::VariableOutOfRange);
        }
        let mut result = Self::new();
        for term in self.terms() {
            let exponent = term.exponents.get(variable).copied().unwrap_or(0);
            let mut exponents = term.exponents;
            if let Some(slot) = exponents.get_mut(variable) {
                *slot = 0;
            }
            let coefficient = term.coefficient * raise(value, exponent);
            result.push(MultivariateTerm::new(coefficient, exponents))?;
        }
        // Fixing a variable can leave several terms with matching powers.
        result.collect_like_terms();
        Ok(result)
    }

    /// The sum with another polynomial, written into one of the caller's size.
    ///
    /// Returns [`PolynomialError::CapacityExceeded`] when the two term lists together do not fit,
    /// which is checked before matching terms are merged, so size `OUT` for the combined count.
    pub fn add_into<const OTHER: usize, const OUT: usize>(
        &self,
        other: &MultivariatePolynomial<VARIABLES, OTHER, T>,
    ) -> Result<MultivariatePolynomial<VARIABLES, OUT, T>, PolynomialError> {
        let mut result = MultivariatePolynomial::<VARIABLES, OUT, T>::new();
        for term in self.terms().iter().chain(other.terms().iter()) {
            result.push(*term)?;
        }
        result.collect_like_terms();
        Ok(result)
    }

    /// The product with another polynomial, written into one of the caller's size.
    ///
    /// Every pair of terms gives one term of the product, so `OUT` needs to hold the two term
    /// counts multiplied together — matching terms are merged only afterwards. Returns
    /// [`PolynomialError::CapacityExceeded`] when they do not fit, and
    /// [`PolynomialError::DegreeOverflow`] when a power would run past what a term can hold.
    pub fn multiply_into<const OTHER: usize, const OUT: usize>(
        &self,
        other: &MultivariatePolynomial<VARIABLES, OTHER, T>,
    ) -> Result<MultivariatePolynomial<VARIABLES, OUT, T>, PolynomialError> {
        let mut result = MultivariatePolynomial::<VARIABLES, OUT, T>::new();
        for left in self.terms() {
            for right in other.terms() {
                // Multiplying terms adds their powers together, variable by variable.
                let mut exponents = left.exponents;
                for (slot, added) in exponents.iter_mut().zip(right.exponents.iter()) {
                    *slot = slot
                        .checked_add(*added)
                        .ok_or(PolynomialError::DegreeOverflow)?;
                }
                let coefficient = left.coefficient * right.coefficient;
                result.push(MultivariateTerm::new(coefficient, exponents))?;
            }
        }
        result.collect_like_terms();
        Ok(result)
    }

    /// Merges terms that raise every variable to the same powers, and drops any that add up to
    /// nothing.
    ///
    /// Every term is compared against every term kept before it, so the work grows with the square
    /// of the term count. That is fine at the sizes this type is meant for, and worth knowing before
    /// reaching for a few hundred terms.
    pub fn collect_like_terms(&mut self) {
        let mut kept = 0;
        for index in 0..self.length {
            let Some(term) = self.terms.get(index).copied() else {
                continue;
            };
            // Look for a term already kept that raises everything to the same powers.
            let match_found = self.terms.get_mut(..kept).and_then(|kept_terms| {
                kept_terms
                    .iter_mut()
                    .find(|existing| existing.exponents == term.exponents)
            });
            if let Some(existing) = match_found {
                existing.coefficient += term.coefficient;
            } else if let Some(slot) = self.terms.get_mut(kept) {
                *slot = term;
                kept += 1;
            }
        }

        // Anything that cancelled exactly is no longer a term at all.
        let mut written = 0;
        for index in 0..kept {
            let Some(term) = self.terms.get(index).copied() else {
                continue;
            };
            if term.coefficient == T::ZERO {
                continue;
            }
            if let Some(slot) = self.terms.get_mut(written) {
                *slot = term;
                written += 1;
            }
        }
        self.length = written;
    }

    /// The largest total power any term reaches, adding each term's powers together. `None` when
    /// there are no terms.
    #[must_use]
    pub fn total_degree(&self) -> Option<u32> {
        let mut largest: Option<u32> = None;
        for term in self.terms() {
            let total = term
                .exponents
                .iter()
                .fold(0_u32, |running, power| running.saturating_add(*power));
            largest = Some(match largest {
                Some(previous) => previous.max(total),
                None => total,
            });
        }
        largest
    }

    /// The largest power one variable is raised to. `None` when there are no terms, or when the
    /// index names no variable — this is a question, not an operation, so it reports nothing rather
    /// than failing.
    #[must_use]
    pub fn degree_in(&self, variable: usize) -> Option<u32> {
        if variable >= VARIABLES || self.length == 0 {
            return None;
        }
        let mut largest = 0;
        for term in self.terms() {
            largest = largest.max(term.exponents.get(variable).copied().unwrap_or(0));
        }
        Some(largest)
    }
}

impl<const MAX_TERMS: usize, T: Numeric> MultivariatePolynomial<1, MAX_TERMS, T> {
    /// The same polynomial as a dense list of coefficients.
    ///
    /// This is the way across to [`Polynomial`], and with it to roots, division, and everything else
    /// the dense type can do. Returns [`PolynomialError::DegreeOverflow`] when a power is too high
    /// for the size asked for.
    ///
    /// ```
    /// use multicalc::{MultivariatePolynomial, MultivariateTerm, Polynomial};
    ///
    /// // 3x² - 1
    /// let p = MultivariatePolynomial::<1, 2>::try_from_terms(&[
    ///     MultivariateTerm::new(3.0, [2]),
    ///     MultivariateTerm::new(-1.0, [0]),
    /// ])
    /// .unwrap();
    /// let dense: Polynomial<3> = p.to_univariate().unwrap();
    /// assert_eq!(dense.coefficients(), &[-1.0, 0.0, 3.0]);
    /// ```
    pub fn to_univariate<const COEFFICIENT_COUNT: usize>(
        &self,
    ) -> Result<Polynomial<COEFFICIENT_COUNT, T>, PolynomialError> {
        let mut coefficients = [T::ZERO; COEFFICIENT_COUNT];
        for term in self.terms() {
            let power = term.exponents.first().copied().unwrap_or(0) as usize;
            match coefficients.get_mut(power) {
                Some(slot) => *slot += term.coefficient,
                None => return Err(PolynomialError::DegreeOverflow),
            }
        }
        Ok(Polynomial::new(coefficients))
    }

    /// The same polynomial as a list of terms, one per coefficient that is not zero.
    ///
    /// Returns [`PolynomialError::CapacityExceeded`] when there are more non-zero coefficients than
    /// terms fit.
    pub fn from_univariate<const COEFFICIENT_COUNT: usize>(
        polynomial: &Polynomial<COEFFICIENT_COUNT, T>,
    ) -> Result<Self, PolynomialError> {
        let mut result = Self::new();
        for (power, coefficient) in polynomial.coefficients().iter().enumerate() {
            if *coefficient == T::ZERO {
                continue;
            }
            let exponent = u32::try_from(power).map_err(|_| PolynomialError::DegreeOverflow)?;
            result.push(MultivariateTerm::new(*coefficient, [exponent]))?;
        }
        Ok(result)
    }
}
