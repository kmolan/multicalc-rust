//! The scalar abstraction the calculus modules are generic over.
//!
//! [`Numeric`] wraps the `libm` transcendentals, the float constants, and the numeric
//! conversions the modules need, so each algorithm is written once and runs at either
//! precision. It is implemented for `f64` and `f32`.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A floating-point scalar usable by the differentiation, integration, and approximation
/// modules. Transcendentals come from [`libm`], so the trait works without `std`.
pub trait Numeric:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + core::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// The value `0`.
    const ZERO: Self;
    /// The value `1`.
    const ONE: Self;
    /// The value `2`.
    const TWO: Self;
    /// The value `0.5`.
    const HALF: Self;
    /// Archimedes' constant, π.
    const PI: Self;
    /// The difference between `1` and the next larger representable value.
    const EPSILON: Self;
    /// Not a Number.
    const NAN: Self;
    /// Positive infinity.
    const INFINITY: Self;
    /// Negative infinity.
    const NEG_INFINITY: Self;
    /// The largest finite value.
    ///
    /// ```
    /// use multicalc::Numeric;
    /// assert_eq!(<f64 as Numeric>::MAX, f64::MAX);
    /// ```
    const MAX: Self;
    /// The smallest positive normal value.
    ///
    /// ```
    /// use multicalc::Numeric;
    /// assert_eq!(<f64 as Numeric>::MIN_POSITIVE, f64::MIN_POSITIVE);
    /// ```
    const MIN_POSITIVE: Self;

    /// Converts from `f64`, narrowing if necessary. Used for table values and literals.
    fn from_f64(value: f64) -> Self;
    /// Converts from `u64`, e.g. an iteration count.
    fn from_u64(value: u64) -> Self;
    /// Converts from `usize`, e.g. a point or variable count.
    fn from_usize(value: usize) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Sine, with `self` in radians.
    fn sin(self) -> Self;
    /// Cosine, with `self` in radians.
    fn cos(self) -> Self;
    /// Tangent, with `self` in radians.
    fn tan(self) -> Self;
    /// `e` raised to the power `self`.
    fn exp(self) -> Self;
    /// Natural logarithm of `self`.
    fn ln(self) -> Self;

    /// `self` raised to an integer power, by exponentiation-by-squaring.
    ///
    /// The zero exponent gives [`Numeric::ONE`]; a negative exponent inverts the result.
    /// Built from `*` and `/` alone, so any `Numeric` (including a dual number) gets a
    /// correct derivative without overriding this method.
    #[inline]
    fn powi(self, n: i32) -> Self {
        let mut exponent = n.unsigned_abs();
        let mut base = self;
        let mut acc = Self::ONE;

        while exponent > 0 {
            if exponent & 1 == 1 {
                acc *= base;
            }
            exponent >>= 1;
            if exponent > 0 {
                base *= base;
            }
        }

        if n < 0 { Self::ONE / acc } else { acc }
    }

    /// Returns `true` if `self` is NaN.
    fn is_nan(self) -> bool;
    /// Returns `true` if `self` is neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

impl Numeric for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;
    const EPSILON: Self = f64::EPSILON;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const MAX: Self = f64::MAX;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;

    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }
    #[inline]
    fn from_u64(value: u64) -> Self {
        value as f64
    }
    #[inline]
    fn from_usize(value: usize) -> Self {
        value as f64
    }

    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        libm::tan(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }

    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

impl Numeric for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const EPSILON: Self = f32::EPSILON;
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const MAX: Self = f32::MAX;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    #[inline]
    fn from_u64(value: u64) -> Self {
        value as f32
    }
    #[inline]
    fn from_usize(value: usize) -> Self {
        value as f32
    }

    #[inline]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline]
    fn tan(self) -> Self {
        libm::tanf(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::expf(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::logf(self)
    }

    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}
