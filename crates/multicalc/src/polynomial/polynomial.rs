//! A polynomial held as its coefficients, lowest power first.
#![deny(clippy::indexing_slicing)]

use crate::scalar::Numeric;

/// A polynomial in one variable, held as a fixed array of coefficients.
///
/// Coefficients run from the lowest power upward, so `coefficients[k]` multiplies `x^k` and
/// `[1.0, -2.0, 3.0]` means `1 - 2x + 3x²`. That one order holds everywhere in the module.
///
/// The array is always `COEFFICIENT_COUNT` long, which is one more than the highest power the
/// polynomial can hold. Unused top slots are zero, and a polynomial that fills fewer of them is
/// still the same value — [`degree`](Self::degree) reports the highest power that is actually
/// there. Storage is a plain array, so the type is `Copy`, sits on the stack, and allocates
/// nothing.
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
    #[inline]
    #[must_use]
    pub const fn new(coefficients: [T; COEFFICIENT_COUNT]) -> Self {
        Self { coefficients }
    }

    /// The polynomial that is zero everywhere.
    #[inline]
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            coefficients: [T::ZERO; COEFFICIENT_COUNT],
        }
    }

    /// The coefficients, lowest power first.
    #[inline]
    #[must_use]
    pub fn coefficients(&self) -> &[T; COEFFICIENT_COUNT] {
        &self.coefficients
    }

    /// The coefficient multiplying `x^power`, or `None` when the polynomial has no slot that high.
    #[inline]
    #[must_use]
    pub fn coefficient(&self, power: usize) -> Option<T> {
        self.coefficients.get(power).copied()
    }

    /// The highest power whose coefficient is not zero, or `None` when every coefficient is zero.
    ///
    /// Trailing zeros do not count, so a `Polynomial<8, _>` holding a cubic reports 3.
    #[inline]
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        self.coefficients
            .iter()
            .rposition(|coefficient| *coefficient != T::ZERO)
    }

    /// The coefficient at the [`degree`](Self::degree), or `None` when every coefficient is zero.
    #[inline]
    #[must_use]
    pub fn leading_coefficient(&self) -> Option<T> {
        self.degree().and_then(|power| self.coefficient(power))
    }

    /// Whether every coefficient is zero.
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coefficients
            .iter()
            .all(|coefficient| *coefficient == T::ZERO)
    }

    /// Whether every coefficient is finite.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.coefficients
            .iter()
            .all(|coefficient| coefficient.is_finite())
    }

    /// The same polynomial with room for a different number of coefficients.
    ///
    /// Growing fills the new top slots with zero. Shrinking returns `None` when it would drop a
    /// coefficient that is not exactly zero, so the value can never change silently.
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
}
