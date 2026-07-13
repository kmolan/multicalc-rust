//! Forward-mode dual numbers for exact first derivatives.
//!
//! A [`Dual`] carries a value and a derivative together. Evaluating a function on a
//! [`Dual::variable`] returns `f(x)` and `f'(x)` in one pass, exact to rounding and with
//! no allocation. Because `Dual` implements [`Numeric`], any function written generically
//! over `Numeric` can be differentiated by calling it with `Dual` instead of a plain float.

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::scalar::Numeric;

/// A dual number `value + deriv·ε`, where `ε² = 0`.
///
/// The arithmetic and [`Numeric`] methods propagate the derivative by the chain rule, so
/// `deriv` tracks the first derivative of whatever expression built the value.
#[derive(Debug, Clone, Copy)]
pub struct Dual<T: Numeric> {
    /// The value, `f(x)`.
    pub value: T,
    /// The first derivative, `f'(x)`.
    pub deriv: T,
}

impl<T: Numeric> Dual<T> {
    /// A dual number with an explicit value and derivative.
    #[inline]
    pub fn new(value: T, deriv: T) -> Self {
        Dual { value, deriv }
    }

    /// A constant, whose derivative with respect to the variable is zero.
    #[inline]
    pub fn constant(value: T) -> Self {
        Dual {
            value,
            deriv: T::ZERO,
        }
    }

    /// The independent variable, seeded with derivative one.
    #[inline]
    pub fn variable(value: T) -> Self {
        Dual {
            value,
            deriv: T::ONE,
        }
    }
}

impl<T: Numeric> Add for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Dual {
            value: self.value + rhs.value,
            deriv: self.deriv + rhs.deriv,
        }
    }
}

impl<T: Numeric> Sub for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Dual {
            value: self.value - rhs.value,
            deriv: self.deriv - rhs.deriv,
        }
    }
}

impl<T: Numeric> Mul for Dual<T> {
    type Output = Self;
    /// Product rule: `(uv)' = u'v + uv'`.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Dual {
            value: self.value * rhs.value,
            deriv: self.value * rhs.deriv + self.deriv * rhs.value,
        }
    }
}

impl<T: Numeric> Div for Dual<T> {
    type Output = Self;
    /// Quotient rule: `(u/v)' = (u'v − uv') / v²`. A zero divisor yields `inf`/`NaN`, as with
    /// plain floats.
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Dual {
            value: self.value / rhs.value,
            deriv: (self.deriv * rhs.value - self.value * rhs.deriv) / (rhs.value * rhs.value),
        }
    }
}

impl<T: Numeric> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Dual {
            value: -self.value,
            deriv: -self.deriv,
        }
    }
}

impl<T: Numeric> AddAssign for Dual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Numeric> SubAssign for Dual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Numeric> MulAssign for Dual<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Numeric> DivAssign for Dual<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Comparison uses only the value, so ordering and equality of a `Dual` match the
// underlying scalar; the derivative does not take part. Two duals with equal value but
// different derivative therefore compare equal, so `Dual` is not suited as a map/set key.
impl<T: Numeric> PartialEq for Dual<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: Numeric> PartialOrd for Dual<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: Numeric> Numeric for Dual<T> {
    const ZERO: Self = Dual {
        value: T::ZERO,
        deriv: T::ZERO,
    };
    const ONE: Self = Dual {
        value: T::ONE,
        deriv: T::ZERO,
    };
    const TWO: Self = Dual {
        value: T::TWO,
        deriv: T::ZERO,
    };
    const HALF: Self = Dual {
        value: T::HALF,
        deriv: T::ZERO,
    };
    const PI: Self = Dual {
        value: T::PI,
        deriv: T::ZERO,
    };
    const EPSILON: Self = Dual {
        value: T::EPSILON,
        deriv: T::ZERO,
    };
    const NAN: Self = Dual {
        value: T::NAN,
        deriv: T::ZERO,
    };
    const INFINITY: Self = Dual {
        value: T::INFINITY,
        deriv: T::ZERO,
    };
    const NEG_INFINITY: Self = Dual {
        value: T::NEG_INFINITY,
        deriv: T::ZERO,
    };
    const MAX: Self = Dual {
        value: T::MAX,
        deriv: T::ZERO,
    };
    const MIN_POSITIVE: Self = Dual {
        value: T::MIN_POSITIVE,
        deriv: T::ZERO,
    };

    #[inline]
    fn from_f64(value: f64) -> Self {
        Dual::constant(T::from_f64(value))
    }
    #[inline]
    fn from_u64(value: u64) -> Self {
        Dual::constant(T::from_u64(value))
    }
    #[inline]
    fn from_usize(value: usize) -> Self {
        Dual::constant(T::from_usize(value))
    }

    /// Derivative of `|x|` is its sign; the subgradient at zero is taken as `+1`.
    #[inline]
    fn abs(self) -> Self {
        let deriv = if self.value < T::ZERO {
            -self.deriv
        } else {
            self.deriv
        };
        Dual {
            value: self.value.abs(),
            deriv,
        }
    }
    /// At `value == 0` the derivative is unbounded (`1/(2·0)`) and becomes `inf`/`NaN`.
    #[inline]
    fn sqrt(self) -> Self {
        let root = self.value.sqrt();
        Dual {
            value: root,
            deriv: self.deriv / (T::TWO * root),
        }
    }
    #[inline]
    fn sin(self) -> Self {
        Dual {
            value: self.value.sin(),
            deriv: self.value.cos() * self.deriv,
        }
    }
    #[inline]
    fn cos(self) -> Self {
        Dual {
            value: self.value.cos(),
            deriv: -(self.value.sin()) * self.deriv,
        }
    }
    #[inline]
    fn tan(self) -> Self {
        let t = self.value.tan();
        Dual {
            value: t,
            deriv: (T::ONE + t * t) * self.deriv,
        }
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.value.exp();
        Dual {
            value: e,
            deriv: e * self.deriv,
        }
    }
    /// Defined for `value > 0`; at `0` the value is `-inf` and the derivative unbounded.
    #[inline]
    fn ln(self) -> Self {
        Dual {
            value: self.value.ln(),
            deriv: self.deriv / self.value,
        }
    }

    /// Four-quadrant arctangent. With `y = self` and `x = other`, the derivative is
    /// `(x·y′ − y·x′) / (x² + y²)`.
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let denom = self.value * self.value + other.value * other.value;
        Dual {
            value: self.value.atan2(other.value),
            deriv: (other.value * self.deriv - self.value * other.deriv) / denom,
        }
    }
    /// Magnitude of `self` with the sign of `sign`. The derivative follows `self`, flipping
    /// sign when `self` and `sign` disagree; the sign argument carries no derivative.
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        let same = (self.value < T::ZERO) == (sign.value < T::ZERO);
        Dual {
            value: self.value.copysign(sign.value),
            deriv: if same { self.deriv } else { -self.deriv },
        }
    }
    /// Largest integer `<= self`. A step function, so the derivative is zero.
    #[inline]
    fn floor(self) -> Self {
        Dual {
            value: self.value.floor(),
            deriv: T::ZERO,
        }
    }

    /// Reflects the value only; the derivative is not inspected.
    #[inline]
    fn is_nan(self) -> bool {
        self.value.is_nan()
    }
    /// Reflects the value only; a finite value can still carry a non-finite derivative
    /// (e.g. from `sqrt(0)` or `ln(0)`).
    #[inline]
    fn is_finite(self) -> bool {
        self.value.is_finite()
    }
}
