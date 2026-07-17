//! Hyper-dual numbers for exact first and second derivatives.
//!
//! A [`HyperDual`] carries a value alongside first- and second-order derivative parts.
//! Evaluating a function on a [`HyperDual::variable`] returns `f(x)`, `f'(x)`, and `f''(x)`
//! in one pass, exact to rounding and with no allocation. Seeding two inputs along the two
//! `ε` directions yields an exact mixed partial — the building block of a Hessian. Because
//! `HyperDual` implements [`Numeric`], any function written generically over `Numeric` can be
//! differentiated by calling it with `HyperDual`.

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::scalar::Numeric;

/// A hyper-dual number `real + eps1·ε₁ + eps2·ε₂ + eps1eps2·ε₁ε₂`, where `ε₁² = ε₂² = 0`.
///
/// After evaluating a function, `real` holds the value, `eps1`/`eps2` hold the first
/// derivatives along each seeded direction, and `eps1eps2` holds the second (mixed) derivative.
#[derive(Debug, Clone, Copy)]
pub struct HyperDual<T: Numeric> {
    /// The value, `f`.
    pub real: T,
    /// First derivative along direction 1.
    pub eps1: T,
    /// First derivative along direction 2.
    pub eps2: T,
    /// Second derivative across the two directions.
    pub eps1eps2: T,
}

impl<T: Numeric> HyperDual<T> {
    /// A hyper-dual number with explicit components.
    #[inline]
    pub fn new(real: T, eps1: T, eps2: T, eps1eps2: T) -> Self {
        HyperDual {
            real,
            eps1,
            eps2,
            eps1eps2,
        }
    }

    /// A constant, whose derivatives are all zero.
    #[inline]
    pub fn constant(real: T) -> Self {
        HyperDual {
            real,
            eps1: T::ZERO,
            eps2: T::ZERO,
            eps1eps2: T::ZERO,
        }
    }

    /// The independent variable, seeded along both directions to read `f`, `f'`, and `f''` of a
    /// single-variable function. For a mixed partial, seed two inputs on separate directions with
    /// `new(xᵢ, 1, 0, 0)` and `new(xⱼ, 0, 1, 0)`.
    #[inline]
    pub fn variable(real: T) -> Self {
        HyperDual {
            real,
            eps1: T::ONE,
            eps2: T::ONE,
            eps1eps2: T::ZERO,
        }
    }

    /// Applies a univariate function `φ` given its value and first two derivatives at `real`.
    ///
    /// With `val = φ(real)`, `d1 = φ'(real)`, `d2 = φ''(real)`, this carries the chain rule
    /// through to second order.
    #[inline]
    fn chain(self, val: T, d1: T, d2: T) -> Self {
        HyperDual {
            real: val,
            eps1: d1 * self.eps1,
            eps2: d1 * self.eps2,
            eps1eps2: d1 * self.eps1eps2 + d2 * self.eps1 * self.eps2,
        }
    }

    /// The reciprocal `1/self`, via `φ(x) = 1/x` (`φ' = -1/x²`, `φ'' = 2/x³`).
    #[inline]
    fn recip(self) -> Self {
        let t = T::ONE / self.real;
        self.chain(t, -(t * t), T::TWO * t * t * t)
    }
}

impl<T: Numeric> Add for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        HyperDual {
            real: self.real + rhs.real,
            eps1: self.eps1 + rhs.eps1,
            eps2: self.eps2 + rhs.eps2,
            eps1eps2: self.eps1eps2 + rhs.eps1eps2,
        }
    }
}

impl<T: Numeric> Sub for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        HyperDual {
            real: self.real - rhs.real,
            eps1: self.eps1 - rhs.eps1,
            eps2: self.eps2 - rhs.eps2,
            eps1eps2: self.eps1eps2 - rhs.eps1eps2,
        }
    }
}

impl<T: Numeric> Mul for HyperDual<T> {
    type Output = Self;
    /// Second-order product rule: each derivative part collects the cross terms that survive
    /// `ε₁² = ε₂² = 0`.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        HyperDual {
            real: self.real * rhs.real,
            eps1: self.real * rhs.eps1 + self.eps1 * rhs.real,
            eps2: self.real * rhs.eps2 + self.eps2 * rhs.real,
            eps1eps2: self.real * rhs.eps1eps2
                + self.eps1 * rhs.eps2
                + self.eps2 * rhs.eps1
                + self.eps1eps2 * rhs.real,
        }
    }
}

impl<T: Numeric> Div for HyperDual<T> {
    type Output = Self;
    /// `self · (1/rhs)`. A zero divisor yields `inf`/`NaN`, as with plain floats.
    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)] // division is multiplication by the reciprocal
    fn div(self, rhs: Self) -> Self {
        self * rhs.recip()
    }
}

impl<T: Numeric> Neg for HyperDual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        HyperDual {
            real: -self.real,
            eps1: -self.eps1,
            eps2: -self.eps2,
            eps1eps2: -self.eps1eps2,
        }
    }
}

impl<T: Numeric> AddAssign for HyperDual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Numeric> SubAssign for HyperDual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Numeric> MulAssign for HyperDual<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Numeric> DivAssign for HyperDual<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Comparison uses only the real part, so ordering and equality match the underlying scalar; the
// derivatives do not take part. Two values with equal real but different derivatives therefore
// compare equal, so `HyperDual` is not suited as a map/set key.
impl<T: Numeric> PartialEq for HyperDual<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real
    }
}

impl<T: Numeric> PartialOrd for HyperDual<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl<T: Numeric> Numeric for HyperDual<T> {
    const ZERO: Self = HyperDual {
        real: T::ZERO,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const ONE: Self = HyperDual {
        real: T::ONE,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const TWO: Self = HyperDual {
        real: T::TWO,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const HALF: Self = HyperDual {
        real: T::HALF,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const HUNDRED: Self = HyperDual {
        real: T::HUNDRED,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const PI: Self = HyperDual {
        real: T::PI,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const EPSILON: Self = HyperDual {
        real: T::EPSILON,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const EPSILON_X4: Self = HyperDual {
        real: T::EPSILON_X4,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const EPSILON_X30: Self = HyperDual {
        real: T::EPSILON_X30,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const NAN: Self = HyperDual {
        real: T::NAN,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const INFINITY: Self = HyperDual {
        real: T::INFINITY,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const NEG_INFINITY: Self = HyperDual {
        real: T::NEG_INFINITY,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const MAX: Self = HyperDual {
        real: T::MAX,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };
    const MIN_POSITIVE: Self = HyperDual {
        real: T::MIN_POSITIVE,
        eps1: T::ZERO,
        eps2: T::ZERO,
        eps1eps2: T::ZERO,
    };

    #[inline]
    fn from_f64(value: f64) -> Self {
        HyperDual::constant(T::from_f64(value))
    }
    #[inline]
    fn from_u64(value: u64) -> Self {
        HyperDual::constant(T::from_u64(value))
    }
    #[inline]
    fn from_usize(value: usize) -> Self {
        HyperDual::constant(T::from_usize(value))
    }

    /// Derivative of `|x|` is its sign (the subgradient at zero is `+1`); its second derivative
    /// is zero.
    #[inline]
    fn abs(self) -> Self {
        let sign = if self.real < T::ZERO { -T::ONE } else { T::ONE };
        self.chain(self.real.abs(), sign, T::ZERO)
    }
    /// At `real == 0` the derivatives are unbounded and become `inf`/`NaN`.
    #[inline]
    fn sqrt(self) -> Self {
        let root = self.real.sqrt();
        let d1 = T::ONE / (T::TWO * root);
        let d2 = -(d1 / (T::TWO * self.real));
        self.chain(root, d1, d2)
    }
    #[inline]
    fn sin(self) -> Self {
        self.chain(self.real.sin(), self.real.cos(), -(self.real.sin()))
    }
    #[inline]
    fn cos(self) -> Self {
        self.chain(self.real.cos(), -(self.real.sin()), -(self.real.cos()))
    }
    #[inline]
    fn tan(self) -> Self {
        let t = self.real.tan();
        let sec2 = T::ONE + t * t;
        self.chain(t, sec2, T::TWO * t * sec2)
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.real.exp();
        self.chain(e, e, e)
    }
    /// Defined for `real > 0`; at `0` the value is `-inf` and the derivatives unbounded.
    #[inline]
    fn ln(self) -> Self {
        self.chain(
            self.real.ln(),
            T::ONE / self.real,
            -(T::ONE / (self.real * self.real)),
        )
    }

    /// Four-quadrant arctangent carried to second order. With `Y = self.real`,
    /// `X = other.real`, `r² = X² + Y²`, the partials are `f_Y = X/r²`, `f_X = −Y/r²`,
    /// `f_YY = −2XY/r⁴`, `f_XX = 2XY/r⁴`, `f_XY = (Y² − X²)/r⁴`. The `r²` denominator is
    /// nonzero off the origin, so the `x = 0` axis is handled; only `X = Y = 0` degenerates,
    /// as for float `atan2`.
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let (y, x) = (self, other);
        let (yr, xr) = (y.real, x.real);
        let r2 = xr * xr + yr * yr;
        let r4 = r2 * r2;
        let f_y = xr / r2;
        let f_x = -yr / r2;
        let f_yy = -(T::TWO * xr * yr) / r4;
        let f_xx = (T::TWO * xr * yr) / r4;
        let f_xy = (yr * yr - xr * xr) / r4;
        HyperDual {
            real: yr.atan2(xr),
            eps1: f_y * y.eps1 + f_x * x.eps1,
            eps2: f_y * y.eps2 + f_x * x.eps2,
            eps1eps2: f_y * y.eps1eps2
                + f_x * x.eps1eps2
                + f_yy * y.eps1 * y.eps2
                + f_xx * x.eps1 * x.eps2
                + f_xy * (y.eps1 * x.eps2 + y.eps2 * x.eps1),
        }
    }
    /// Magnitude of `self` with the sign of `sign`: a linear scaling by `±1`, so it rides the
    /// univariate `chain` helper with slope `s` and zero curvature.
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        let s = if (self.real < T::ZERO) == (sign.real < T::ZERO) {
            T::ONE
        } else {
            -T::ONE
        };
        self.chain(self.real.copysign(sign.real), s, T::ZERO)
    }
    /// Largest integer `<= self`; the derivatives of a step function are zero.
    #[inline]
    fn floor(self) -> Self {
        HyperDual {
            real: self.real.floor(),
            eps1: T::ZERO,
            eps2: T::ZERO,
            eps1eps2: T::ZERO,
        }
    }

    /// Reflects the real part only; the derivatives are not inspected.
    #[inline]
    fn is_nan(self) -> bool {
        self.real.is_nan()
    }
    /// Reflects the real part only; a finite value can still carry a non-finite derivative
    /// (e.g. from `sqrt(0)` or `ln(0)`).
    #[inline]
    fn is_finite(self) -> bool {
        self.real.is_finite()
    }
}
