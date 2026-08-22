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
    /// The value `3`.
    const THREE: Self;
    /// The value `0.5`.
    const HALF: Self;
    /// The value `10`.
    const TEN: Self;
    /// The value `100`.
    const HUNDRED: Self;
    // The value `180`.
    const ONE_HUNDRED_EIGHTY: Self;
    /// Archimedes' constant, π.
    #[allow(clippy::min_ident_chars)]
    const PI: Self;
    /// Two times π (the circle constant τ) — one full turn in radians.
    const TWO_PI: Self;
    /// The difference between `1` and the next larger representable value.
    const EPSILON: Self;
    /// `4 × EPSILON`.
    const EPSILON_X4: Self;
    /// `30 × EPSILON`.
    const EPSILON_X30: Self;
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

    /// The type used to represent constant values.
    /// For normal numbers (such as `f32` and `f64`) this is simply `Self`.
    /// For auto differentiation types this is the underlying scalar value.
    type Constant: Numeric;

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
    /// Cube root.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let x: f64 = 8.0;
    /// assert_eq!(x.cbrt(), 2.0);
    /// ```
    fn cbrt(self) -> Self;
    /// Sine, with `self` in radians.
    fn sin(self) -> Self;
    /// Cosine, with `self` in radians.
    fn cos(self) -> Self;
    /// Tangent, with `self` in radians.
    fn tan(self) -> Self;
    /// `e` raised to the power `self`.
    fn exp(self) -> Self;
    /// Natural logarithm of `self`.
    fn log(self) -> Self;

    /// Calculates `(e^self) - 1`.
    /// The default implementation computes the result using `exp` and normal subtraction,
    /// however overrides for `f32` and `f64` use a more accurate primitive.
    /// Other implementors of the trait will also likely want to override this method.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let x: f64 = 0.0;
    /// assert_eq!(x.expm1(), 0.0);
    /// ```
    fn expm1(self) -> Self {
        self.exp() - Self::ONE
    }

    /// Calculates `ln(1 + x)`.
    /// The default implementation computes the result using `log` and normal addition,
    /// however overrides for `f32` and `f64` use a more accurate primitive.
    /// Other implementors of the trait will also likely want to override this method.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let x: f64 = 0.0;
    /// assert_eq!(x.ln_1p(), 0.0);
    /// ```
    fn ln_1p(self) -> Self {
        (self + Self::ONE).log()
    }

    /// Base 2 logarithm.
    /// The default implementation uses two calls to `log`.
    /// More efficient overrides are used for `f32` and `f64`.
    /// Other implementors of the trait will also likely want to override this method.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let x: f64 = 2.0;
    /// assert_eq!(x.log2(), 1.0);
    /// ```
    fn log2(self) -> Self {
        self.log() / Self::TWO.log()
    }

    /// Base 10 logarithm.
    /// The default implementation uses two calls to `log`.
    /// More efficient overrides are used for `f32` and `f64`.
    /// Other implementors of the trait will also likely want to override this method.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let x: f64 = 10.0;
    /// assert_eq!(x.log10(), 1.0);
    /// ```
    fn log10(self) -> Self {
        self.log() / Self::TEN.log()
    }

    /// Four-quadrant arctangent of `self / other`, in radians, taking `self` as the y
    /// coordinate and `other` as the x coordinate. Result in `(-π, π]`.
    fn atan2(self, other: Self) -> Self;

    /// A value with the magnitude of `self` and the sign of `sign`.
    fn copysign(self, sign: Self) -> Self;

    /// The largest integer less than or equal to `self`.
    ///
    /// A step function, so its derivative is zero everywhere it is differentiable; the dual
    /// implementations therefore carry a zero derivative.
    fn floor(self) -> Self;

    /// The largest integer greater than or equal to `self`.
    ///
    /// A step function, so its derivative is zero everywhere it is differentiable; the dual
    /// implementations therefore carry a zero derivative.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let x: f64 = 2.1;
    /// assert_eq!(x.ceil(), 3.0);
    /// ```
    fn ceil(self) -> Self;

    /// The nearest integer to `self`, rounding half away from zero.
    ///
    /// A step function, so its derivative is zero everywhere it is differentiable; the dual
    /// implementations therefore carry a zero derivative.
    fn round(self) -> Self;

    /// The rounds a number toward zero, effectively removing the decimal part.
    ///
    /// A step function, so its derivative is zero everywhere it is differentiable; the dual
    /// implementations therefore carry a zero derivative.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let x: f64 = 2.8;
    /// assert_eq!(x.trunc(), 2.0);
    /// ```
    fn trunc(self) -> Self;

    /// Restrict a value to the interval `[min, max]` unless it is NaN.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let min: f64 = -1.0;
    /// let max: f64 = 2.0;
    ///
    /// assert_eq!(Numeric::clamp(17.0, min, max), max);
    /// assert_eq!(Numeric::clamp(1.0, min, max), 1.0);
    /// assert_eq!(Numeric::clamp(-1.3, min, max), min);
    /// assert!(Numeric::clamp(f64::NAN, min, max).is_nan());
    /// ```
    fn clamp(self, min: Self::Constant, max: Self::Constant) -> Self;

    /// Sign of `self`: `1` for positive, `-1` for negative, `0` at zero.
    ///
    /// The derivative is zero wherever it is defined. This default maps NaN and both zeros to
    /// `0`; the `f32`/`f64` impls override to match the primitive `signum` (NaN maps to NaN,
    /// signed zeros are preserved).
    #[inline]
    fn signum(self) -> Self {
        if self < Self::ZERO {
            -Self::ONE
        } else if self > Self::ZERO {
            Self::ONE
        } else {
            Self::ZERO
        }
    }

    /// The larger of `self` and `other`, compared with `>`.
    ///
    /// On the autodiff scalars this selects the larger operand together with its full
    /// derivative payload. The `f32`/`f64` impls override to match the primitive `max`,
    /// which returns the non-NaN operand.
    #[inline]
    fn max(self, other: Self) -> Self {
        if self > other { self } else { other }
    }

    /// The smaller of `self` and `other`, compared with `<`.
    ///
    /// On the autodiff scalars this selects the smaller operand together with its full
    /// derivative payload. The `f32`/`f64` impls override to match the primitive `min`,
    /// which returns the non-NaN operand.
    #[inline]
    fn min(self, other: Self) -> Self {
        if self < other { self } else { other }
    }

    /// Folds an angle into a ±π band by subtracting whole turns.
    ///
    /// Note: This function can fail to produce a result in the range [-π, π] due to
    /// loss of precision in floating point numbers.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// // Small-ish number of whole turns removed
    /// let x: f64 = 0.132 + 10.0 * f64::TWO_PI;
    /// assert!((x.wrap_to_pi() - 0.132).abs() < 1e-8);
    ///
    /// // A very large number fails to be in range due to precision loss.
    /// let x: f64 = 3.6663732791101215e224;
    /// assert!(x.wrap_to_pi() < -4e200);
    /// ```
    #[inline]
    fn wrap_to_pi(self) -> Self {
        self - Self::TWO_PI * (self / Self::TWO_PI).round()
    }

    /// Convert an angle measured in degrees to on measured in radians.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let deg: f64 = 90.0;
    /// let rad = deg.to_radians();
    /// assert!((rad - f64::PI / 2.0).abs() < 1e-12);
    /// ```
    #[inline]
    fn to_radians(self) -> Self {
        self / Self::ONE_HUNDRED_EIGHTY * Self::PI
    }

    /// Convert an angle measured in radians to on measured in degrees.
    ///
    /// ```
    /// use multicalc::Numeric;
    ///
    /// let rad: f64 = f64::PI / 4.0;
    /// let deg = rad.to_degrees();
    /// assert!((deg - 45.0).abs() < 1e-12);
    /// ```
    #[inline]
    fn to_degrees(self) -> Self {
        self / Self::PI * Self::ONE_HUNDRED_EIGHTY
    }

    /// Arctangent, in radians.
    #[inline]
    fn atan(self) -> Self {
        self.atan2(Self::ONE)
    }

    /// Arcsine, in radians, for `self` in `[-1, 1]`.
    #[inline]
    fn asin(self) -> Self {
        self.atan2((Self::ONE - self * self).sqrt())
    }

    /// Arccosine, in radians, for `self` in `[-1, 1]`.
    #[inline]
    fn acos(self) -> Self {
        (Self::ONE - self * self).sqrt().atan2(self)
    }

    /// Hyperbolic sine.
    #[inline]
    fn sinh(self) -> Self {
        (self.exp() - (-self).exp()) * Self::HALF
    }

    /// Hyperbolic cosine.
    #[inline]
    fn cosh(self) -> Self {
        (self.exp() + (-self).exp()) * Self::HALF
    }

    /// Hyperbolic tangent.
    #[inline]
    fn tanh(self) -> Self {
        self.sinh().safe_div(self.cosh())
    }

    /// Euclidean distance `√(self² + other²)`, scaled to avoid overflow and underflow.
    #[inline]
    fn hypot(self, other: Self) -> Self {
        let a = self.abs();
        let b = other.abs();
        let (max, min) = if a > b { (a, b) } else { (b, a) };
        if max == Self::ZERO {
            Self::ZERO
        } else {
            let ratio = min / max;
            max * (Self::ONE + ratio * ratio).sqrt()
        }
    }

    /// `self` raised to a floating-point power, via `exp(n · ln self)`. Defined for
    /// `self > 0`; the `f32`/`f64` impls override for negative bases and edge cases.
    #[inline]
    fn powf(self, n: Self) -> Self {
        (n * self.log()).exp()
    }

    /// `self * a + b`. The `f32`/`f64` impls fuse the operation for extra precision.
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    /// The reciprocal `1 / self`.
    #[inline]
    fn recip(self) -> Self {
        Self::ONE.safe_div(self)
    }

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

    /// Division with a zero check to prevent possible panics.
    /// The behavior is meant to match the floating point standard.
    /// I.e. this has the following rules:
    /// (finite positive) / zero = INFINITY
    /// (finite negative) / zero = NEG_INFINITY
    /// INFINITY / zero = INFINITY
    /// NEG_INFINITY / zero = NEG_INFINITY
    /// zero / zero = NAN
    /// NAN / zero = NAN
    #[inline]
    fn safe_div(self, other: Self) -> Self {
        if other == Self::ZERO {
            if self.is_nan() {
                return Self::NAN;
            } else if self < Self::ZERO {
                return Self::NEG_INFINITY;
            } else if Self::ZERO < self {
                return Self::INFINITY;
            } else {
                return Self::NAN;
            }
        }

        self / other
    }

    /// Returns `true` if `self` is NaN.
    fn is_nan(self) -> bool;
    /// Returns `true` if `self` is neither infinite nor NaN.
    fn is_finite(self) -> bool;
    /// Returns `true` if `self` is positive or negative infinity and false otherwise.
    fn is_infinite(self) -> bool;
}

impl Numeric for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const THREE: Self = 3.0;
    const HALF: Self = 0.5;
    const TEN: Self = 10.0;
    const HUNDRED: Self = 100.0;
    const ONE_HUNDRED_EIGHTY: Self = 180.0;
    const PI: Self = core::f64::consts::PI;
    const TWO_PI: Self = core::f64::consts::PI * 2.0;
    const EPSILON: Self = f64::EPSILON;
    const EPSILON_X4: Self = f64::EPSILON * 4.0;
    const EPSILON_X30: Self = f64::EPSILON * 30.0;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const MAX: Self = f64::MAX;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;

    type Constant = f64;

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
    fn cbrt(self) -> Self {
        libm::cbrt(self)
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
    fn expm1(self) -> Self {
        libm::expm1(self)
    }
    #[inline]
    fn log(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        libm::log1p(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2(self)
    }
    #[inline]
    fn log10(self) -> Self {
        libm::log10(self)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysign(self, sign)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::round(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        libm::trunc(self)
    }
    #[inline]
    fn clamp(self, min: Self::Constant, max: Self::Constant) -> Self {
        self.clamp(min, max)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        libm::fmax(self, other)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        libm::fmin(self, other)
    }
    #[inline]
    fn signum(self) -> Self {
        if f64::is_nan(self) {
            f64::NAN
        } else {
            libm::copysign(1.0, self)
        }
    }
    #[inline]
    fn atan(self) -> Self {
        libm::atan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        libm::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acos(self)
    }
    #[inline]
    fn sinh(self) -> Self {
        libm::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        libm::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        libm::tanh(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypot(self, other)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        libm::pow(self, n)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fma(self, a, b)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }

    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn is_infinite(self) -> bool {
        f64::is_infinite(self)
    }
}

impl Numeric for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const THREE: Self = 3.0;
    const HALF: Self = 0.5;
    const TEN: Self = 10.0;
    const HUNDRED: Self = 100.0;
    const ONE_HUNDRED_EIGHTY: Self = 180.0;
    const PI: Self = core::f32::consts::PI;
    const TWO_PI: Self = core::f32::consts::PI * 2.0;
    const EPSILON: Self = f32::EPSILON;
    const EPSILON_X4: Self = f32::EPSILON * 4.0;
    const EPSILON_X30: Self = f32::EPSILON * 30.0;
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const MAX: Self = f32::MAX;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;

    type Constant = f32;

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
    fn cbrt(self) -> Self {
        libm::cbrtf(self)
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
    fn expm1(self) -> Self {
        libm::expm1f(self)
    }
    #[inline]
    fn log(self) -> Self {
        libm::logf(self)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        libm::log1pf(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2f(self)
    }
    #[inline]
    fn log10(self) -> Self {
        libm::log10f(self)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2f(self, other)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        libm::copysignf(self, sign)
    }
    #[inline]
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        libm::ceilf(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::roundf(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        libm::truncf(self)
    }
    #[inline]
    fn clamp(self, min: Self::Constant, max: Self::Constant) -> Self {
        self.clamp(min, max)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        libm::fmaxf(self, other)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        libm::fminf(self, other)
    }
    #[inline]
    fn signum(self) -> Self {
        if f32::is_nan(self) {
            f32::NAN
        } else {
            libm::copysignf(1.0, self)
        }
    }
    #[inline]
    fn atan(self) -> Self {
        libm::atanf(self)
    }
    #[inline]
    fn asin(self) -> Self {
        libm::asinf(self)
    }
    #[inline]
    fn acos(self) -> Self {
        libm::acosf(self)
    }
    #[inline]
    fn sinh(self) -> Self {
        libm::sinhf(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        libm::coshf(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        libm::tanhf(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypotf(self, other)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        libm::powf(self, n)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fmaf(self, a, b)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }

    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn is_infinite(self) -> bool {
        f32::is_infinite(self)
    }
}
