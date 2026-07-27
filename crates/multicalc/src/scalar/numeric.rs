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
    /// The value `100`.
    const HUNDRED: Self;
    /// Archimedes' constant, π.
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

    /// The nearest integer to `self`, rounding half away from zero.
    ///
    /// A step function, so its derivative is zero everywhere it is differentiable; the dual
    /// implementations therefore carry a zero derivative.
    fn round(self) -> Self;

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
    ///
    /// Evaluated as `(1 − e^(−2x)) / (1 + e^(−2x))` for `x ≥ 0`, mirrored for
    /// `x < 0`, rather than as `sinh/cosh`. The quotient form overflows to
    /// `inf/inf` — that is, NaN — once `|x|` passes about 88 in `f32` or 709
    /// in `f64`, where the true value has simply settled at ±1. Taking `exp`
    /// of a non-positive argument cannot overflow: it underflows to zero,
    /// giving exactly ±1 with a derivative that decays to zero, both correct
    /// in the limit.
    ///
    /// The branch is on the sign rather than a multiply by `signum`, because
    /// `signum(0)` is zero and would wipe out the derivative at the origin,
    /// where `tanh'(0) = 1`.
    #[inline]
    fn tanh(self) -> Self {
        if self < Self::ZERO {
            let e = (self * Self::TWO).exp();
            (e - Self::ONE) / (e + Self::ONE)
        } else {
            let e = (-(self * Self::TWO)).exp();
            (Self::ONE - e) / (Self::ONE + e)
        }
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

    /// `self` raised to a floating-point power.
    ///
    /// `exp(n · ln self)` alone is only valid for `self > 0`: `ln` of zero is `-inf` and
    /// `ln` of a negative is NaN, so the plain form returns NaN — or the right value with
    /// a NaN derivative — for cases that are perfectly well defined. `x.powf(2.0)` is the
    /// obvious one; it should agree with `x * x` everywhere, and on the autodiff scalars
    /// it did not.
    ///
    /// Non-positive bases are handled explicitly:
    ///
    /// - `self == 0`: the value is `0` for `n > 0` and the derivative is `n · 0^(n−1)`,
    ///   which is `0` for `n > 1`, `1` for `n == 1`, and divergent below that. Computed
    ///   directly rather than through `ln 0 = -inf`.
    /// - `self < 0` with an integral `n`: real-valued, equal to `|self|^n` with the sign
    ///   of `(-1)^n`. The derivative is `n · self^(n−1)`, carried by rebuilding the
    ///   expression from `|self|` and applying the sign, so the chain rule stays intact.
    /// - `self < 0` with a fractional `n`: no real value; stays NaN, matching `libm::pow`.
    #[inline]
    fn powf(self, n: Self) -> Self {
        if self > Self::ZERO {
            return (n * self.ln()).exp();
        }

        if self == Self::ZERO {
            // 0^0 is 1 by convention; 0^n is 0 for n > 0 and diverges for n < 0.
            // The derivative n·0^(n-1) is 0 for n > 1, 1 for n == 1, and divergent
            // otherwise — all of which fall out of powi for integral n.
            if n == Self::ZERO {
                return Self::ONE;
            }
            if n == Self::ONE {
                return self;
            }
            if n > Self::ONE && n == n.floor() {
                return self * self * (n - Self::ONE);
            }
            return (n * self.ln()).exp();
        }

        // self < 0. Only an integral exponent has a real value.
        if n != n.floor() || !n.is_finite() {
            return (n * self.ln()).exp(); // NaN, as libm::pow gives
        }

        // |self|^n carries the correct magnitude and the correct |derivative|; the sign
        // of both follows (-1)^n. Recovering n's parity without leaving the scalar type:
        // n - 2·floor(n/2) is 0 for even n and 1 for odd n.
        let magnitude = (n * (-self).ln()).exp();
        let half = n * Self::HALF;
        let parity = n - Self::TWO * half.floor();
        if parity == Self::ZERO {
            magnitude
        } else {
            -magnitude
        }
    }

    /// `self * a + b`. The `f32`/`f64` impls fuse the operation for extra precision.
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }

    /// The reciprocal `1 / self`.
    #[inline]
    fn recip(self) -> Self {
        Self::ONE / self
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
    const HUNDRED: Self = 100.0;
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
    fn round(self) -> Self {
        libm::round(self)
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
}

impl Numeric for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const HUNDRED: Self = 100.0;
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
    fn round(self) -> Self {
        libm::roundf(self)
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
}
