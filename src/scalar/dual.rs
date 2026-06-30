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

#[cfg(test)]
mod test {
    use super::Dual;
    use crate::scalar::Numeric;

    // Dual results are exact to rounding, so the tolerances are tight.
    const TOL: f64 = 1e-12;
    const TOL_F32: f32 = 1e-5;

    #[test]
    fn test_polynomial_powi() {
        // f(x) = x^3, f'(x) = 3x^2; at x = 2 -> 8 and 12
        let y = Dual::variable(2.0_f64).powi(3);
        assert!(f64::abs(y.value - 8.0) < TOL);
        assert!(f64::abs(y.deriv - 12.0) < TOL);
    }

    #[test]
    fn test_polynomial_sum() {
        // f(x) = 3x^2 + 2x, f'(x) = 6x + 2; at x = 2 -> 16 and 14
        let x = Dual::variable(2.0_f64);
        let y = Dual::constant(3.0) * x * x + Dual::constant(2.0) * x;
        assert!(f64::abs(y.value - 16.0) < TOL);
        assert!(f64::abs(y.deriv - 14.0) < TOL);
    }

    #[test]
    fn test_negative_powi() {
        // f(x) = x^-2, f'(x) = -2 x^-3; at x = 2 -> 0.25 and -0.25
        let y = Dual::variable(2.0_f64).powi(-2);
        assert!(f64::abs(y.value - 0.25) < TOL);
        assert!(f64::abs(y.deriv - (-0.25)) < TOL);
    }

    #[test]
    fn test_sqrt() {
        // f(x) = sqrt(x), f'(x) = 1/(2 sqrt(x)); at x = 4 -> 2 and 0.25
        let y = Dual::variable(4.0_f64).sqrt();
        assert!(f64::abs(y.value - 2.0) < TOL);
        assert!(f64::abs(y.deriv - 0.25) < TOL);
    }

    #[test]
    fn test_sin_cos_tan() {
        let x0 = 0.7_f64;
        let s = Dual::variable(x0).sin();
        assert!(f64::abs(s.value - f64::sin(x0)) < TOL);
        assert!(f64::abs(s.deriv - f64::cos(x0)) < TOL);

        let c = Dual::variable(x0).cos();
        assert!(f64::abs(c.value - f64::cos(x0)) < TOL);
        assert!(f64::abs(c.deriv - (-f64::sin(x0))) < TOL);

        let t = Dual::variable(x0).tan();
        assert!(f64::abs(t.value - f64::tan(x0)) < TOL);
        assert!(f64::abs(t.deriv - (1.0 + f64::tan(x0) * f64::tan(x0))) < TOL);
    }

    #[test]
    fn test_exp_ln() {
        let x0 = 1.3_f64;
        let e = Dual::variable(x0).exp();
        assert!(f64::abs(e.value - f64::exp(x0)) < TOL);
        assert!(f64::abs(e.deriv - f64::exp(x0)) < TOL);

        // f(x) = ln(x), f'(x) = 1/x; at x = 2 -> ln 2 and 0.5
        let l = Dual::variable(2.0_f64).ln();
        assert!(f64::abs(l.value - f64::ln(2.0)) < TOL);
        assert!(f64::abs(l.deriv - 0.5) < TOL);
    }

    #[test]
    fn test_chain_exp_of_sin() {
        // f(x) = exp(sin(x)), f'(x) = cos(x) exp(sin(x))
        let x0 = 0.6_f64;
        let y = Dual::variable(x0).sin().exp();
        assert!(f64::abs(y.value - f64::exp(f64::sin(x0))) < TOL);
        assert!(f64::abs(y.deriv - f64::cos(x0) * f64::exp(f64::sin(x0))) < TOL);
    }

    #[test]
    fn test_rational() {
        // f(x) = x / (1 + x^2), f'(x) = (1 - x^2) / (1 + x^2)^2
        let x0 = 1.5_f64;
        let x = Dual::variable(x0);
        let y = x / (Dual::constant(1.0) + x * x);
        let denom = (1.0 + x0 * x0) * (1.0 + x0 * x0);
        assert!(f64::abs(y.value - x0 / (1.0 + x0 * x0)) < TOL);
        assert!(f64::abs(y.deriv - (1.0 - x0 * x0) / denom) < TOL);
    }

    #[test]
    fn test_product_sin_cos() {
        // f(x) = sin(x) cos(x), f'(x) = cos^2(x) - sin^2(x)
        let x0 = 0.9_f64;
        let x = Dual::variable(x0);
        let y = x.sin() * x.cos();
        assert!(f64::abs(y.value - f64::sin(x0) * f64::cos(x0)) < TOL);
        let expected = f64::cos(x0) * f64::cos(x0) - f64::sin(x0) * f64::sin(x0);
        assert!(f64::abs(y.deriv - expected) < TOL);
    }

    #[test]
    fn test_abs_both_sides() {
        // derivative of |x| is +1 for x > 0 and -1 for x < 0
        let pos = Dual::variable(2.0_f64).abs();
        assert!(f64::abs(pos.value - 2.0) < TOL);
        assert!(f64::abs(pos.deriv - 1.0) < TOL);

        let neg = Dual::variable(-2.0_f64).abs();
        assert!(f64::abs(neg.value - 2.0) < TOL);
        assert!(f64::abs(neg.deriv - (-1.0)) < TOL);
    }

    #[test]
    fn test_generic_over_numeric() {
        // The same function runs with a plain float or with a Dual.
        fn poly<T: Numeric>(t: T) -> T {
            t.powi(3) + T::from_f64(2.0) * t
        }

        let x0 = 1.7_f64;
        let plain = poly(x0);
        let dual = poly(Dual::variable(x0));
        assert!(f64::abs(dual.value - plain) < TOL);
        // f'(x) = 3x^2 + 2
        assert!(f64::abs(dual.deriv - (3.0 * x0 * x0 + 2.0)) < TOL);
    }

    #[test]
    fn test_partial_derivatives() {
        // f(x, y) = x^2 * y + sin(x)
        fn f<T: Numeric>(v: &[T; 2]) -> T {
            v[0] * v[0] * v[1] + v[0].sin()
        }

        let (x0, y0) = (1.0_f64, 2.0_f64);

        // seed x: df/dx = 2xy + cos(x)
        let dfdx = f(&[Dual::variable(x0), Dual::constant(y0)]).deriv;
        assert!(f64::abs(dfdx - (2.0 * x0 * y0 + f64::cos(x0))) < TOL);

        // seed y: df/dy = x^2
        let dfdy = f(&[Dual::constant(x0), Dual::variable(y0)]).deriv;
        assert!(f64::abs(dfdy - x0 * x0) < TOL);
    }

    #[test]
    fn test_generic_over_f32() {
        // Dual is generic over the scalar; here it carries f32.
        let y = Dual::variable(2.0_f32).powi(3);
        assert!(f32::abs(y.value - 8.0) < TOL_F32);
        assert!(f32::abs(y.deriv - 12.0) < TOL_F32);
    }

    #[test]
    fn test_powi_zero() {
        // x^0 = 1, derivative 0
        let y = Dual::variable(3.0_f64).powi(0);
        assert!(f64::abs(y.value - 1.0) < TOL);
        assert!(f64::abs(y.deriv) < TOL);
    }

    #[test]
    fn test_constant_has_zero_derivative() {
        // a constant carries no derivative through any operation
        let c = Dual::constant(1.3_f64);
        let y = c.exp() * c.sin() + c.powi(2);
        assert!(f64::abs(y.deriv) < TOL);
    }

    #[test]
    fn test_sqrt_zero_derivative_is_infinite() {
        // the derivative of sqrt at 0 is unbounded, while the value stays finite
        let y = Dual::variable(0.0_f64).sqrt();
        assert!(f64::abs(y.value) < TOL);
        assert!(y.deriv.is_infinite());
        // is_finite reflects the value only, so it still reports finite here
        assert!(y.is_finite());
    }

    #[test]
    fn test_ln_zero_blows_up() {
        // ln(0) = -inf with an unbounded derivative
        let y = Dual::variable(0.0_f64).ln();
        assert!(y.value.is_infinite() && y.value < 0.0);
        assert!(y.deriv.is_infinite() && y.deriv > 0.0);
    }
}
