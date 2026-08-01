//! A polynomial held as its coefficients, lowest power first.
#![deny(clippy::indexing_slicing)]

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::PolynomialError;
use crate::scalar::Numeric;

/// A polynomial in one variable, held as a fixed array of coefficients.
///
/// Coefficients run from the lowest power upward, so `coefficients[k]` multiplies `x^k` and
/// `[1.0, -2.0, 3.0]` means `1 - 2x + 3x²`. That one order holds everywhere in the module.
///
/// The array is always `COEFFICIENT_COUNT` long, which is one more than the highest power the
/// polynomial can hold. Coefficients above the polynomial's degree are zero, and one that uses
/// fewer of them is still the same value — [`degree`](Self::degree) reports the highest power that
/// is actually there. Storage is a plain array, so the type is `Copy`, sits on the stack, and
/// allocates nothing.
///
/// ```
/// use multicalc::Polynomial;
///
/// // 1 - 2x + 3x²
/// let p: Polynomial<3> = Polynomial::new([1.0, -2.0, 3.0]);
/// assert!((p.evaluate(2.0) - 9.0).abs() < 1e-12);
///
/// // The value and the first two derivatives at the same point, from one pass.
/// let [value, slope, bend] = p.evaluate_with_derivatives(2.0);
/// assert!((value - 9.0).abs() < 1e-12);
/// assert!((slope - 10.0).abs() < 1e-12);
/// assert!((bend - 6.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Polynomial<const COEFFICIENT_COUNT: usize, T: Numeric = f64> {
    coefficients: [T; COEFFICIENT_COUNT],
}

impl<const COEFFICIENT_COUNT: usize, T: Numeric> Default for Polynomial<COEFFICIENT_COUNT, T> {
    fn default() -> Self {
        Self::zeros()
    }
}

impl<const COEFFICIENT_COUNT: usize, T: Numeric> Polynomial<COEFFICIENT_COUNT, T> {
    /// Wraps an array of coefficients, lowest power first.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 1 - 2x + 3x²
    /// let p = Polynomial::new([1.0, -2.0, 3.0]);
    /// assert_eq!(p.coefficient(0), Some(1.0));
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(coefficients: [T; COEFFICIENT_COUNT]) -> Self {
        Self { coefficients }
    }

    /// The polynomial that is zero everywhere.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// let p = Polynomial::<4>::zeros();
    /// assert_eq!(p.coefficients(), &[0.0; 4]);
    /// ```
    #[inline]
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            coefficients: [T::ZERO; COEFFICIENT_COUNT],
        }
    }

    /// The coefficients, lowest power first.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, -2.0, 3.0]);
    /// assert_eq!(p.coefficients(), &[1.0, -2.0, 3.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn coefficients(&self) -> &[T; COEFFICIENT_COUNT] {
        &self.coefficients
    }

    /// The coefficient multiplying `x^power`, or `None` when the polynomial holds no coefficient
    /// that high.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, -2.0, 3.0]);
    /// assert_eq!(p.coefficient(2), Some(3.0)); // the x² term
    /// assert_eq!(p.coefficient(7), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn coefficient(&self, power: usize) -> Option<T> {
        self.coefficients.get(power).copied()
    }

    /// The highest power whose coefficient is not zero, or `None` when every coefficient is zero.
    ///
    /// Trailing zeros do not count, so a `Polynomial<8, _>` holding a cubic reports 3.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 1 + 2x, with room for one more coefficient
    /// let p = Polynomial::new([1.0, 2.0, 0.0]);
    /// assert_eq!(p.degree(), Some(1));
    /// assert_eq!(Polynomial::<3>::zeros().degree(), None);
    /// ```
    #[inline]
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        self.coefficients
            .iter()
            .rposition(|coefficient| *coefficient != T::ZERO)
    }

    /// The coefficient at the [`degree`](Self::degree), or `None` when every coefficient is zero.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, 2.0, 0.0]);
    /// assert_eq!(p.leading_coefficient(), Some(2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn leading_coefficient(&self) -> Option<T> {
        self.degree().and_then(|power| self.coefficient(power))
    }

    /// Whether every coefficient is zero.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// assert!(Polynomial::<3>::zeros().is_zero());
    /// assert!(!Polynomial::new([0.0, 1.0]).is_zero());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coefficients
            .iter()
            .all(|coefficient| *coefficient == T::ZERO)
    }

    /// Whether every coefficient is finite.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// assert!(Polynomial::new([1.0, 2.0]).is_finite());
    /// assert!(!Polynomial::new([1.0, f64::INFINITY]).is_finite());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.coefficients
            .iter()
            .all(|coefficient| coefficient.is_finite())
    }

    /// The same polynomial with room for a different number of coefficients.
    ///
    /// Growing fills the new coefficients with zero. Shrinking returns `None` when it would drop a
    /// coefficient that is not exactly zero, so the value can never change silently.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 1 + 2x, with room for four coefficients
    /// let p = Polynomial::new([1.0, 2.0, 0.0, 0.0]);
    ///
    /// let grown: Polynomial<6> = p.try_resize().unwrap();
    /// assert_eq!(grown.coefficients(), &[1.0, 2.0, 0.0, 0.0, 0.0, 0.0]);
    ///
    /// // Two coefficients still hold every term, but one would drop the 2x.
    /// assert_eq!(p.try_resize::<2>().unwrap().coefficients(), &[1.0, 2.0]);
    /// assert!(p.try_resize::<1>().is_none());
    /// ```
    #[must_use]
    pub fn try_resize<const OTHER: usize>(&self) -> Option<Polynomial<OTHER, T>> {
        if let Some(dropped) = self.coefficients.get(OTHER..)
            && dropped.iter().any(|coefficient| *coefficient != T::ZERO)
        {
            return None;
        }
        let mut resized = Polynomial::<OTHER, T>::zeros();
        for (slot, coefficient) in resized
            .coefficients
            .iter_mut()
            .zip(self.coefficients.iter())
        {
            *slot = *coefficient;
        }
        Some(resized)
    }

    /// The value at `x`.
    ///
    /// Works down from the top coefficient, multiplying by `x` and adding the next one, so a
    /// polynomial of any size costs one multiply and one add per coefficient and never raises `x`
    /// to a power directly.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 1 - 2x + 3x², at x = 2
    /// let p: Polynomial<3> = Polynomial::new([1.0, -2.0, 3.0]);
    /// assert!((p.evaluate(2.0) - 9.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn evaluate(&self, x: T) -> T {
        let mut accumulator = T::ZERO;
        for coefficient in self.coefficients.iter().rev() {
            accumulator = accumulator.mul_add(x, *coefficient);
        }
        accumulator
    }

    /// The value at `x` together with the first `ORDER_COUNT - 1` derivatives there, as
    /// `[value, first derivative, second derivative, …]`.
    ///
    /// One pass over the coefficients produces every order at once, rather than one pass per
    /// order — this is what a trajectory tracker calls to get position, velocity, and acceleration
    /// together. Orders above the polynomial's degree come back zero, which is the right answer
    /// rather than a gap.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 1 - 2x + 3x², at x = 2
    /// let p: Polynomial<3> = Polynomial::new([1.0, -2.0, 3.0]);
    /// let [value, slope, bend, third] = p.evaluate_with_derivatives(2.0);
    /// assert!((value - 9.0).abs() < 1e-12);
    /// assert!((slope - 10.0).abs() < 1e-12);
    /// assert!((bend - 6.0).abs() < 1e-12);
    /// assert_eq!(third, 0.0); // past the degree
    /// ```
    #[must_use]
    pub fn evaluate_with_derivatives<const ORDER_COUNT: usize>(&self, x: T) -> [T; ORDER_COUNT] {
        let mut result = [T::ZERO; ORDER_COUNT];
        for coefficient in self.coefficients.iter().rev() {
            for order in (1..ORDER_COUNT).rev() {
                let lower = result.get(order - 1).copied().unwrap_or(T::ZERO);
                if let Some(slot) = result.get_mut(order) {
                    *slot = slot.mul_add(x, lower);
                }
            }
            if let Some(slot) = result.get_mut(0) {
                *slot = slot.mul_add(x, *coefficient);
            }
        }
        // The sweep leaves each order short by the count of ways its terms could be picked, which
        // is that order multiplied by every whole number below it.
        let mut scale = T::ONE;
        for (order, slot) in result.iter_mut().enumerate() {
            if order > 0 {
                scale *= T::from_usize(order);
            }
            *slot *= scale;
        }
        result
    }

    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 1 - 2x + 3x², whose derivative is -2 + 6x
    /// let p = Polynomial::new([1.0, -2.0, 3.0]);
    /// assert_eq!(p.derivative().coefficients(), &[-2.0, 6.0, 0.0]);
    /// ```
    #[must_use]
    pub fn derivative(&self) -> Self {
        let mut result = Self::zeros();
        for (power, slot) in result.coefficients.iter_mut().enumerate() {
            if let Some(above) = self.coefficient(power + 1) {
                *slot = above * T::from_usize(power + 1);
            }
        }
        result
    }

    /// The derivative applied `order` times.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 1 - 2x + 3x², differentiated twice, is 6
    /// let p = Polynomial::new([1.0, -2.0, 3.0]);
    /// assert_eq!(p.nth_derivative(2).coefficients(), &[6.0, 0.0, 0.0]);
    /// assert!(p.nth_derivative(3).is_zero());
    /// ```
    #[must_use]
    pub fn nth_derivative(&self, order: usize) -> Self {
        if order >= COEFFICIENT_COUNT {
            return Self::zeros();
        }
        let mut result = *self;
        for _ in 0..order {
            result = result.derivative();
        }
        result
    }

    /// The area under the curve between `lower` and `upper`.
    ///
    /// This adds up each term's contribution directly, so it needs no spare coefficient however
    /// high the polynomial's degree.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 3x² from 0 to 2 covers 8
    /// let p: Polynomial<3> = Polynomial::new([0.0, 0.0, 3.0]);
    /// assert!((p.definite_integral(0.0, 2.0) - 8.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn definite_integral(&self, lower: T, upper: T) -> T {
        let mut total = T::ZERO;
        let mut lower_power = lower;
        let mut upper_power = upper;
        for (power, coefficient) in self.coefficients.iter().enumerate() {
            total += *coefficient * (upper_power - lower_power) / T::from_usize(power + 1);
            lower_power *= lower;
            upper_power *= upper;
        }
        total
    }

    /// Every coefficient multiplied by `factor`.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// let p = Polynomial::new([1.0, -2.0]).scale(3.0);
    /// assert_eq!(p.coefficients(), &[3.0, -6.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn scale(mut self, factor: T) -> Self {
        for slot in self.coefficients.iter_mut() {
            *slot *= factor;
        }
        self
    }

    /// The product with another polynomial, written into a polynomial of the caller's size.
    ///
    /// A product runs to the sum of the two degrees, which neither input's size gives, so `OUT` is
    /// named by the caller. Returns [`PolynomialError::DegreeOverflow`] when a term of the product
    /// lands past the last coefficient and is not zero.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // (1 + x)(2 - x) = 2 + x - x²
    /// let left = Polynomial::new([1.0, 1.0]);
    /// let right = Polynomial::new([2.0, -1.0]);
    /// let product = left.multiply_into::<2, 3>(&right).unwrap();
    /// assert_eq!(product.coefficients(), &[2.0, 1.0, -1.0]);
    ///
    /// // Two coefficients cannot hold the x² term.
    /// assert!(left.multiply_into::<2, 2>(&right).is_err());
    /// ```
    pub fn multiply_into<const OTHER: usize, const OUT: usize>(
        &self,
        other: &Polynomial<OTHER, T>,
    ) -> Result<Polynomial<OUT, T>, PolynomialError> {
        let mut product = Polynomial::<OUT, T>::zeros();
        for (left_power, left) in self.coefficients.iter().enumerate() {
            for (right_power, right) in other.coefficients.iter().enumerate() {
                let term = *left * *right;
                match product.coefficients.get_mut(left_power + right_power) {
                    Some(slot) => *slot += term,
                    None if term != T::ZERO => return Err(PolynomialError::DegreeOverflow),
                    None => {}
                }
            }
        }
        Ok(product)
    }

    /// This polynomial with `inner` substituted for its variable, written into a polynomial of the
    /// caller's size.
    ///
    /// The result runs to the two degrees multiplied together, so `OUT` is named by the caller.
    /// Returns [`PolynomialError::DegreeOverflow`] when a term lands past the last coefficient and
    /// is not zero.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // outer(x) = 1 + x², inner(x) = x + 1, so outer(inner(x)) = 2 + 2x + x²
    /// let outer = Polynomial::new([1.0, 0.0, 1.0]);
    /// let inner = Polynomial::new([1.0, 1.0]);
    /// let composed = outer.compose_into::<2, 3>(&inner).unwrap();
    /// assert_eq!(composed.coefficients(), &[2.0, 2.0, 1.0]);
    /// ```
    pub fn compose_into<const INNER: usize, const OUT: usize>(
        &self,
        inner: &Polynomial<INNER, T>,
    ) -> Result<Polynomial<OUT, T>, PolynomialError> {
        let mut accumulator = Polynomial::<OUT, T>::zeros();
        for coefficient in self.coefficients.iter().rev() {
            accumulator = accumulator.multiply_into::<INNER, OUT>(inner)?;
            match accumulator.coefficients.get_mut(0) {
                Some(slot) => *slot += *coefficient,
                None if *coefficient != T::ZERO => return Err(PolynomialError::DegreeOverflow),
                None => {}
            }
        }
        Ok(accumulator)
    }

    /// Long division by another polynomial, giving the quotient and the remainder.
    ///
    /// The three sizes are, in order, the divisor's, the quotient's, and the remainder's. Neither
    /// output's size follows from the inputs, so the caller names both. Returns
    /// [`PolynomialError::LeadingCoefficientZero`] when the divisor is zero everywhere, and
    /// [`PolynomialError::DegreeOverflow`] when either output has too few coefficients.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // x³ + 2x² - x - 2, which factors as (x + 2)(x + 1)(x - 1)
    /// let p = Polynomial::new([-2.0, -1.0, 2.0, 1.0]);
    /// // x + 2, in two coefficients
    /// let divisor = Polynomial::new([2.0, 1.0]);
    ///
    /// // Two coefficients for the divisor, four for the quotient, two for the remainder.
    /// let (quotient, remainder) = p.divide::<2, 4, 2>(&divisor).unwrap();
    /// assert_eq!(quotient.coefficients(), &[-1.0, 0.0, 1.0, 0.0]); // x² - 1
    /// assert!(remainder.is_zero());
    /// ```
    pub fn divide<const DIVISOR: usize, const QUOTIENT: usize, const REMAINDER: usize>(
        &self,
        divisor: &Polynomial<DIVISOR, T>,
    ) -> Result<(Polynomial<QUOTIENT, T>, Polynomial<REMAINDER, T>), PolynomialError> {
        let divisor_degree = divisor
            .degree()
            .ok_or(PolynomialError::LeadingCoefficientZero)?;
        let divisor_leading = divisor
            .coefficient(divisor_degree)
            .ok_or(PolynomialError::LeadingCoefficientZero)?;

        let mut working = self.coefficients;
        let mut quotient = Polynomial::<QUOTIENT, T>::zeros();

        // Take the highest power off the working copy at a time, until what is left is lower than
        // the divisor.
        let mut power = COEFFICIENT_COUNT;
        while power > divisor_degree {
            power -= 1;
            let leading = working.get(power).copied().unwrap_or(T::ZERO);
            if leading == T::ZERO {
                continue;
            }
            let share = leading / divisor_leading;
            let quotient_power = power - divisor_degree;
            match quotient.coefficients.get_mut(quotient_power) {
                Some(slot) => *slot += share,
                None => return Err(PolynomialError::DegreeOverflow),
            }
            for (divisor_power, divisor_coefficient) in divisor.coefficients.iter().enumerate() {
                if let Some(slot) = working.get_mut(quotient_power + divisor_power) {
                    *slot -= share * *divisor_coefficient;
                }
            }
            // The subtraction is meant to clear this power exactly; rounding can leave a speck.
            if let Some(slot) = working.get_mut(power) {
                *slot = T::ZERO;
            }
        }

        let mut remainder = Polynomial::<REMAINDER, T>::zeros();
        for (power, coefficient) in working.iter().enumerate().take(divisor_degree) {
            match remainder.coefficients.get_mut(power) {
                Some(slot) => *slot = *coefficient,
                None if *coefficient != T::ZERO => return Err(PolynomialError::DegreeOverflow),
                None => {}
            }
        }
        Ok((quotient, remainder))
    }

    /// The polynomial shifted along its variable: the value at `x` here is the original's value at
    /// `x + offset`.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // x² shifted by one is (x + 1)² = 1 + 2x + x²
    /// let p: Polynomial<3> = Polynomial::new([0.0, 0.0, 1.0]);
    /// assert_eq!(p.shift_argument(1.0).coefficients(), &[1.0, 2.0, 1.0]);
    /// ```
    #[must_use]
    pub fn shift_argument(&self, offset: T) -> Self {
        let mut shifted = Self::zeros();
        let mut working = *self;
        for slot in shifted.coefficients.iter_mut() {
            // Dividing by `x - offset` leaves the value at `offset` behind, which is the next
            // coefficient of the answer. The quotient is what the following round divides.
            let mut quotient = Self::zeros();
            let mut carry = T::ZERO;
            for power in (0..COEFFICIENT_COUNT).rev() {
                let coefficient = working.coefficient(power).unwrap_or(T::ZERO);
                carry = carry.mul_add(offset, coefficient);
                if let Some(lower) = power.checked_sub(1)
                    && let Some(target) = quotient.coefficients.get_mut(lower)
                {
                    *target = carry;
                }
            }
            *slot = carry;
            working = quotient;
        }
        shifted
    }

    /// The polynomial stretched along its variable: the value at `x` here is the original's value
    /// at `factor · x`.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 1 + x + x² at twice the rate is 1 + 2x + 4x²
    /// let p = Polynomial::new([1.0, 1.0, 1.0]);
    /// assert_eq!(p.scale_argument(2.0).coefficients(), &[1.0, 2.0, 4.0]);
    /// ```
    #[must_use]
    pub fn scale_argument(&self, factor: T) -> Self {
        let mut scaled = *self;
        let mut power_of_factor = T::ONE;
        for slot in scaled.coefficients.iter_mut() {
            *slot *= power_of_factor;
            power_of_factor *= factor;
        }
        scaled
    }

    /// The coefficients in the opposite order.
    ///
    /// This turns every root into one divided by itself, which is only meaningful when the lowest
    /// coefficient is not zero — a zero there means the original has a root at zero, which has no
    /// reciprocal, and the reversed polynomial quietly drops a degree instead.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// // 2 - 3x + x², whose roots are 1 and 2
    /// let p = Polynomial::new([2.0, -3.0, 1.0]);
    /// // reversed: 1 - 3x + 2x², whose roots are 1 and 1/2
    /// assert_eq!(p.reverse().coefficients(), &[1.0, -3.0, 2.0]);
    /// ```
    #[must_use]
    pub fn reverse(&self) -> Self {
        let mut reversed = *self;
        reversed.coefficients.reverse();
        reversed
    }
}

impl<const COEFFICIENT_COUNT: usize, T: Numeric> Add for Polynomial<COEFFICIENT_COUNT, T> {
    type Output = Self;

    /// Adds matching powers. Both sides hold the same number of coefficients, so nothing can
    /// outgrow the result.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// let sum = Polynomial::new([1.0, 2.0]) + Polynomial::new([10.0, 20.0]);
    /// assert_eq!(sum.coefficients(), &[11.0, 22.0]);
    /// ```
    #[inline]
    fn add(mut self, other: Self) -> Self {
        for (slot, addend) in self.coefficients.iter_mut().zip(other.coefficients.iter()) {
            *slot += *addend;
        }
        self
    }
}

impl<const COEFFICIENT_COUNT: usize, T: Numeric> Sub for Polynomial<COEFFICIENT_COUNT, T> {
    type Output = Self;

    /// Subtracts matching powers.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// let difference = Polynomial::new([10.0, 20.0]) - Polynomial::new([1.0, 2.0]);
    /// assert_eq!(difference.coefficients(), &[9.0, 18.0]);
    /// ```
    #[inline]
    fn sub(mut self, other: Self) -> Self {
        for (slot, subtrahend) in self.coefficients.iter_mut().zip(other.coefficients.iter()) {
            *slot -= *subtrahend;
        }
        self
    }
}

impl<const COEFFICIENT_COUNT: usize, T: Numeric> Neg for Polynomial<COEFFICIENT_COUNT, T> {
    type Output = Self;

    /// Flips the sign of every coefficient.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// assert_eq!((-Polynomial::new([1.0, -2.0])).coefficients(), &[-1.0, 2.0]);
    /// ```
    #[inline]
    fn neg(mut self) -> Self {
        for slot in self.coefficients.iter_mut() {
            *slot = -*slot;
        }
        self
    }
}

impl<const COEFFICIENT_COUNT: usize, T: Numeric> Mul<T> for Polynomial<COEFFICIENT_COUNT, T> {
    type Output = Self;

    /// Multiplies every coefficient by a number, the same as [`scale`](Polynomial::scale).
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// assert_eq!((Polynomial::new([1.0, -2.0]) * 3.0).coefficients(), &[3.0, -6.0]);
    /// ```
    #[inline]
    fn mul(self, factor: T) -> Self {
        self.scale(factor)
    }
}

impl<const COEFFICIENT_COUNT: usize, T: Numeric> Div<T> for Polynomial<COEFFICIENT_COUNT, T> {
    type Output = Self;

    /// Divides every coefficient by a number.
    ///
    /// ```
    /// use multicalc::Polynomial;
    ///
    /// assert_eq!((Polynomial::new([3.0, -6.0]) / 3.0).coefficients(), &[1.0, -2.0]);
    /// ```
    #[inline]
    fn div(mut self, divisor: T) -> Self {
        for slot in self.coefficients.iter_mut() {
            *slot /= divisor;
        }
        self
    }
}
