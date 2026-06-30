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

#[cfg(test)]
mod test {
    use super::HyperDual;
    use crate::scalar::Numeric;

    // Hyper-dual results are exact to rounding, so the tolerances are tight.
    const TOL: f64 = 1e-12;
    const TOL_F32: f32 = 1e-3;

    #[test]
    fn test_single_var_cubic() {
        // f(x) = x^3 -> f'(x) = 3x^2, f''(x) = 6x; at x = 3: 27, 27, 18
        let y = HyperDual::variable(3.0_f64).powi(3);
        assert!(f64::abs(y.real - 27.0) < TOL);
        assert!(f64::abs(y.eps1 - 27.0) < TOL);
        assert!(f64::abs(y.eps2 - 27.0) < TOL);
        assert!(f64::abs(y.eps1eps2 - 18.0) < TOL);
    }

    #[test]
    fn test_powi_second_order() {
        // f(x) = x^4 -> f'(x) = 4x^3, f''(x) = 12x^2; at x = 2: 16, 32, 48
        let y = HyperDual::variable(2.0_f64).powi(4);
        assert!(f64::abs(y.real - 16.0) < TOL);
        assert!(f64::abs(y.eps1 - 32.0) < TOL);
        assert!(f64::abs(y.eps1eps2 - 48.0) < TOL);
    }

    #[test]
    fn test_sin_second_order() {
        // f(x) = sin(x) -> f' = cos(x), f'' = -sin(x)
        let x0 = 0.7_f64;
        let y = HyperDual::variable(x0).sin();
        assert!(f64::abs(y.real - f64::sin(x0)) < TOL);
        assert!(f64::abs(y.eps1 - f64::cos(x0)) < TOL);
        assert!(f64::abs(y.eps1eps2 - (-f64::sin(x0))) < TOL);
    }

    #[test]
    fn test_exp_second_order() {
        // f(x) = exp(x) is its own derivative to all orders
        let x0 = 1.3_f64;
        let y = HyperDual::variable(x0).exp();
        assert!(f64::abs(y.real - f64::exp(x0)) < TOL);
        assert!(f64::abs(y.eps1 - f64::exp(x0)) < TOL);
        assert!(f64::abs(y.eps1eps2 - f64::exp(x0)) < TOL);
    }

    #[test]
    fn test_reciprocal_second_order() {
        // f(x) = 1/x -> f' = -1/x^2, f'' = 2/x^3; at x = 2: 0.5, -0.25, 0.25
        let y = HyperDual::constant(1.0_f64) / HyperDual::variable(2.0_f64);
        assert!(f64::abs(y.real - 0.5) < TOL);
        assert!(f64::abs(y.eps1 - (-0.25)) < TOL);
        assert!(f64::abs(y.eps1eps2 - 0.25) < TOL);
    }

    #[test]
    fn test_full_hessian_matches_analytic() {
        // f(x, y) = x^2 * y + sin(x)
        // grad  = [2xy + cos x, x^2]
        // H     = [[2y - sin x, 2x], [2x, 0]]
        fn f<T: Numeric>(v: &[T; 2]) -> T {
            v[0] * v[0] * v[1] + v[0].sin()
        }

        let (x0, y0) = (1.0_f64, 2.0_f64);

        // diagonal Hxx and the x-gradient: seed x on both directions, y constant
        let hxx = f(&[HyperDual::variable(x0), HyperDual::constant(y0)]);
        assert!(f64::abs(hxx.eps1 - (2.0 * x0 * y0 + f64::cos(x0))) < TOL); // df/dx
        assert!(f64::abs(hxx.eps1eps2 - (2.0 * y0 - f64::sin(x0))) < TOL); // d2f/dx2

        // diagonal Hyy and the y-gradient: seed y on both directions, x constant
        let hyy = f(&[HyperDual::constant(x0), HyperDual::variable(y0)]);
        assert!(f64::abs(hyy.eps1 - x0 * x0) < TOL); // df/dy
        assert!(f64::abs(hyy.eps1eps2) < TOL); // d2f/dy2 = 0

        // mixed Hxy: seed x on direction 1, y on direction 2
        let hxy = f(&[
            HyperDual::new(x0, 1.0, 0.0, 0.0),
            HyperDual::new(y0, 0.0, 1.0, 0.0),
        ]);
        assert!(f64::abs(hxy.eps1eps2 - 2.0 * x0) < TOL);

        // symmetry: swapping the directions gives the same mixed partial
        let hyx = f(&[
            HyperDual::new(x0, 0.0, 1.0, 0.0),
            HyperDual::new(y0, 1.0, 0.0, 0.0),
        ]);
        assert!(f64::abs(hxy.eps1eps2 - hyx.eps1eps2) < TOL);
    }

    #[test]
    fn test_generic_over_numeric() {
        // The same function runs with a plain float or with a HyperDual.
        fn g<T: Numeric>(t: T) -> T {
            t.powi(3) + T::from_f64(2.0) * t
        }

        let x0 = 1.7_f64;
        let plain = g(x0);
        let hd = g(HyperDual::variable(x0));
        assert!(f64::abs(hd.real - plain) < TOL);
        assert!(f64::abs(hd.eps1 - (3.0 * x0 * x0 + 2.0)) < TOL); // f'
        assert!(f64::abs(hd.eps1eps2 - 6.0 * x0) < TOL); // f''
    }

    #[test]
    fn test_generic_over_f32() {
        // HyperDual is generic over the scalar; here it carries f32.
        let y = HyperDual::variable(2.0_f32).powi(4);
        assert!(f32::abs(y.real - 16.0) < TOL_F32);
        assert!(f32::abs(y.eps1 - 32.0) < TOL_F32);
        assert!(f32::abs(y.eps1eps2 - 48.0) < TOL_F32);
    }

    #[test]
    fn test_powi_zero() {
        // x^0 = 1 with all derivatives 0
        let y = HyperDual::variable(3.0_f64).powi(0);
        assert!(f64::abs(y.real - 1.0) < TOL);
        assert!(f64::abs(y.eps1) < TOL);
        assert!(f64::abs(y.eps1eps2) < TOL);
    }

    #[test]
    fn test_constant_has_zero_derivatives() {
        // a constant carries no derivative through any operation
        let c = HyperDual::constant(1.3_f64);
        let y = c.exp() * c.sin() + c.powi(2);
        assert!(f64::abs(y.eps1) < TOL);
        assert!(f64::abs(y.eps2) < TOL);
        assert!(f64::abs(y.eps1eps2) < TOL);
    }

    #[test]
    fn test_sqrt_zero_blows_up() {
        // the derivative of sqrt at 0 is unbounded, while the value stays finite
        let y = HyperDual::variable(0.0_f64).sqrt();
        assert!(f64::abs(y.real) < TOL);
        assert!(y.eps1.is_infinite());
        // is_finite reflects the real part only
        assert!(y.is_finite());
    }

    #[test]
    fn test_ln_zero_blows_up() {
        // ln(0) = -inf with an unbounded first derivative
        let y = HyperDual::variable(0.0_f64).ln();
        assert!(y.real.is_infinite() && y.real < 0.0);
        assert!(y.eps1.is_infinite() && y.eps1 > 0.0);
    }
}
