//! Jets (truncated Taylor series) for arbitrary nth-order forward derivatives.
//!
//! A [`Jet`] carries the first `N` Taylor coefficients of a function around a point. Evaluating a
//! function on a [`Jet::variable`] returns every derivative up to order `N-1` in one pass, exact to
//! rounding and with no allocation. `Dual` is the order-1 case (`Jet<T, 2>`). Because `Jet`
//! implements [`Numeric`], any function written generically over `Numeric` can be differentiated by
//! calling it with a `Jet`.

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::scalar::Numeric;

/// A truncated Taylor series with `N` coefficients.
///
/// `coeffs[k]` is the normalized coefficient `f⁽ᵏ⁾(x) / k!`, so the type captures derivatives of
/// order `0` through `N-1`. The k-th derivative itself is [`Jet::derivative`] (`k! · coeffs[k]`).
#[derive(Debug, Clone, Copy)]
pub struct Jet<T: Numeric, const N: usize> {
    /// The Taylor coefficients, `coeffs[k] = f⁽ᵏ⁾(x) / k!`.
    pub coeffs: [T; N],
}

impl<T: Numeric, const N: usize> Jet<T, N> {
    /// A jet with explicit coefficients.
    #[inline]
    pub fn new(coeffs: [T; N]) -> Self {
        Jet { coeffs }
    }

    /// A constant: value in `coeffs[0]`, all derivatives zero.
    #[inline]
    pub const fn constant(value: T) -> Self {
        let mut coeffs = [T::ZERO; N];
        coeffs[0] = value;
        Jet { coeffs }
    }

    /// The independent variable, seeded to read every derivative of a single-variable function
    /// (`coeffs[0] = x`, `coeffs[1] = 1`). Requires `N >= 2`.
    #[inline]
    pub fn variable(value: T) -> Self {
        const { assert!(N >= 2, "Jet::variable needs at least 2 coefficients") };
        let mut coeffs = [T::ZERO; N];
        coeffs[0] = value;
        coeffs[1] = T::ONE;
        Jet { coeffs }
    }

    /// The value `f(x)` (= `coeffs[0]`).
    #[inline]
    pub fn value(&self) -> T {
        self.coeffs[0]
    }

    /// The `k`-th Taylor coefficient `f⁽ᵏ⁾(x) / k!`.
    #[inline]
    pub fn coefficient(&self, k: usize) -> T {
        self.coeffs[k]
    }

    /// The `k`-th derivative `f⁽ᵏ⁾(x)` (= `k! · coeffs[k]`).
    #[inline]
    pub fn derivative(&self, k: usize) -> T {
        let mut factorial = T::ONE;
        for i in 2..=k {
            factorial *= T::from_usize(i);
        }
        factorial * self.coeffs[k]
    }
}

impl<T: Numeric, const N: usize> Add for Jet<T, N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Jet {
            coeffs: core::array::from_fn(|k| self.coeffs[k] + rhs.coeffs[k]),
        }
    }
}

impl<T: Numeric, const N: usize> Sub for Jet<T, N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Jet {
            coeffs: core::array::from_fn(|k| self.coeffs[k] - rhs.coeffs[k]),
        }
    }
}

impl<T: Numeric, const N: usize> Mul for Jet<T, N> {
    type Output = Self;
    /// Cauchy product: `cₖ = Σ_{i=0..k} aᵢ·b₍ₖ₋ᵢ₎`.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Jet {
            coeffs: core::array::from_fn(|k| {
                let mut acc = T::ZERO;
                for i in 0..=k {
                    acc += self.coeffs[i] * rhs.coeffs[k - i];
                }
                acc
            }),
        }
    }
}

impl<T: Numeric, const N: usize> Div for Jet<T, N> {
    type Output = Self;
    /// Series division recurrence: `cₖ = (aₖ − Σ_{i=0..k-1} cᵢ·b₍ₖ₋ᵢ₎) / b₀`. A zero `b₀` yields
    /// `inf`/`NaN`, as with plain floats.
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let b0 = rhs.coeffs[0];
        let mut c = [T::ZERO; N];
        for k in 0..N {
            let mut acc = self.coeffs[k];
            for (i, &ci) in c.iter().enumerate().take(k) {
                acc -= ci * rhs.coeffs[k - i];
            }
            c[k] = acc / b0;
        }
        Jet { coeffs: c }
    }
}

impl<T: Numeric, const N: usize> Neg for Jet<T, N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Jet {
            coeffs: core::array::from_fn(|k| -self.coeffs[k]),
        }
    }
}

impl<T: Numeric, const N: usize> AddAssign for Jet<T, N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Numeric, const N: usize> SubAssign for Jet<T, N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Numeric, const N: usize> MulAssign for Jet<T, N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Numeric, const N: usize> DivAssign for Jet<T, N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Comparison uses only the value (coeffs[0]), so ordering and equality match the underlying scalar;
// the higher coefficients do not take part. Two jets with equal value but different coefficients
// therefore compare equal, so `Jet` is not suited as a map/set key.
impl<T: Numeric, const N: usize> PartialEq for Jet<T, N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.coeffs[0] == other.coeffs[0]
    }
}

impl<T: Numeric, const N: usize> PartialOrd for Jet<T, N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.coeffs[0].partial_cmp(&other.coeffs[0])
    }
}

impl<T: Numeric, const N: usize> Jet<T, N> {
    /// `sin` and `cos` of the jet, computed together (each recurrence needs the other).
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let v = &self.coeffs;
        let mut s = [T::ZERO; N];
        let mut c = [T::ZERO; N];
        s[0] = v[0].sin();
        c[0] = v[0].cos();
        for k in 1..N {
            let mut sk = T::ZERO;
            let mut ck = T::ZERO;
            for i in 1..=k {
                let weighted = T::from_usize(i) * v[i];
                sk += weighted * c[k - i];
                ck += weighted * s[k - i];
            }
            let kth = T::from_usize(k);
            s[k] = sk / kth;
            c[k] = -(ck / kth);
        }
        (Jet { coeffs: s }, Jet { coeffs: c })
    }
}

impl<T: Numeric, const N: usize> Numeric for Jet<T, N> {
    const ZERO: Self = Self::constant(T::ZERO);
    const ONE: Self = Self::constant(T::ONE);
    const TWO: Self = Self::constant(T::TWO);
    const HALF: Self = Self::constant(T::HALF);
    const PI: Self = Self::constant(T::PI);
    const EPSILON: Self = Self::constant(T::EPSILON);
    const NAN: Self = Self::constant(T::NAN);
    const INFINITY: Self = Self::constant(T::INFINITY);
    const NEG_INFINITY: Self = Self::constant(T::NEG_INFINITY);

    #[inline]
    fn from_f64(value: f64) -> Self {
        Self::constant(T::from_f64(value))
    }
    #[inline]
    fn from_u64(value: u64) -> Self {
        Self::constant(T::from_u64(value))
    }
    #[inline]
    fn from_usize(value: usize) -> Self {
        Self::constant(T::from_usize(value))
    }

    /// Away from zero `|f|` equals `sign(f) · f`, so every coefficient is scaled by the sign of the
    /// value; the derivatives are unbounded at `value == 0`.
    #[inline]
    fn abs(self) -> Self {
        let sign = if self.coeffs[0] < T::ZERO {
            -T::ONE
        } else {
            T::ONE
        };
        Jet {
            coeffs: core::array::from_fn(|k| sign * self.coeffs[k]),
        }
    }

    /// `uₖ = (vₖ − Σ_{i=1..k-1} uᵢ·u₍ₖ₋ᵢ₎) / (2u₀)`. Unbounded at `value == 0`.
    #[inline]
    fn sqrt(self) -> Self {
        let v = &self.coeffs;
        let mut u = [T::ZERO; N];
        u[0] = v[0].sqrt();
        for k in 1..N {
            let mut acc = T::ZERO;
            for i in 1..k {
                acc += u[i] * u[k - i];
            }
            u[k] = (v[k] - acc) / (T::TWO * u[0]);
        }
        Jet { coeffs: u }
    }

    #[inline]
    fn sin(self) -> Self {
        self.sin_cos().0
    }

    #[inline]
    fn cos(self) -> Self {
        self.sin_cos().1
    }

    #[inline]
    fn tan(self) -> Self {
        let (sin, cos) = self.sin_cos();
        sin / cos
    }

    /// `uₖ = (1/k) Σ_{i=1..k} i·vᵢ·u₍ₖ₋ᵢ₎`.
    #[inline]
    fn exp(self) -> Self {
        let v = &self.coeffs;
        let mut u = [T::ZERO; N];
        u[0] = v[0].exp();
        for k in 1..N {
            let mut acc = T::ZERO;
            for i in 1..=k {
                acc += T::from_usize(i) * v[i] * u[k - i];
            }
            u[k] = acc / T::from_usize(k);
        }
        Jet { coeffs: u }
    }

    /// `uₖ = (1/v₀)( vₖ − (1/k) Σ_{j=1..k-1} j·uⱼ·v₍ₖ₋ⱼ₎ )`. Defined for `value > 0`.
    #[inline]
    fn ln(self) -> Self {
        let v = &self.coeffs;
        let mut u = [T::ZERO; N];
        u[0] = v[0].ln();
        for k in 1..N {
            let mut acc = T::ZERO;
            for j in 1..k {
                acc += T::from_usize(j) * u[j] * v[k - j];
            }
            u[k] = (v[k] - acc / T::from_usize(k)) / v[0];
        }
        Jet { coeffs: u }
    }

    /// Reflects the value only; the higher coefficients are not inspected.
    #[inline]
    fn is_nan(self) -> bool {
        self.coeffs[0].is_nan()
    }

    /// Reflects the value only; a finite value can still carry non-finite coefficients (e.g. from
    /// `sqrt(0)` or `ln(0)`).
    #[inline]
    fn is_finite(self) -> bool {
        self.coeffs[0].is_finite()
    }
}

#[cfg(test)]
mod test {
    use super::Jet;
    use crate::scalar::{Dual, Numeric};

    const TOL: f64 = 1e-9;
    const TOL_F32: f32 = 1e-3;

    #[test]
    fn test_exp_all_orders() {
        // every derivative of exp(x) is exp(x)
        let x0 = 0.4_f64;
        let y = Jet::<f64, 6>::variable(x0).exp();
        for k in 0..6 {
            assert!(f64::abs(y.derivative(k) - f64::exp(x0)) < TOL);
        }
    }

    #[test]
    fn test_high_order_polynomial() {
        // f(x) = x^4: f'=4x^3, f''=12x^2, f'''=24x, f''''=24, f'''''=0
        let x0 = 2.0_f64;
        let y = Jet::<f64, 6>::variable(x0).powi(4);
        assert!(f64::abs(y.value() - 16.0) < TOL);
        assert!(f64::abs(y.derivative(1) - 32.0) < TOL);
        assert!(f64::abs(y.derivative(2) - 48.0) < TOL);
        assert!(f64::abs(y.derivative(3) - 48.0) < TOL);
        assert!(f64::abs(y.derivative(4) - 24.0) < TOL);
        assert!(f64::abs(y.derivative(5)) < TOL);
    }

    #[test]
    fn test_sin_derivative_cycle() {
        // derivatives of sin cycle: sin, cos, -sin, -cos, sin
        let x0 = 0.6_f64;
        let y = Jet::<f64, 5>::variable(x0).sin();
        assert!(f64::abs(y.derivative(0) - f64::sin(x0)) < TOL);
        assert!(f64::abs(y.derivative(1) - f64::cos(x0)) < TOL);
        assert!(f64::abs(y.derivative(2) - (-f64::sin(x0))) < TOL);
        assert!(f64::abs(y.derivative(3) - (-f64::cos(x0))) < TOL);
        assert!(f64::abs(y.derivative(4) - f64::sin(x0)) < TOL);
    }

    #[test]
    fn test_reciprocal_all_orders() {
        // f(x) = 1/(1+x): f^(k) = (-1)^k k! / (1+x)^(k+1)
        let x0 = 0.3_f64;
        let denom = Jet::<f64, 5>::constant(1.0) + Jet::variable(x0);
        let y = Jet::<f64, 5>::constant(1.0) / denom;
        let mut sign = 1.0;
        let mut factorial = 1.0;
        for k in 0..5 {
            if k >= 1 {
                factorial *= k as f64;
            }
            let expected = sign * factorial / (1.0 + x0).powi(k as i32 + 1);
            assert!(f64::abs(y.derivative(k) - expected) < TOL);
            sign = -sign;
        }
    }

    #[test]
    fn test_sqrt_orders() {
        // f(x) = sqrt(x): f'=1/(2√x), f''=-1/(4 x^{3/2}), f'''=3/(8 x^{5/2})
        let x0 = 1.7_f64;
        let y = Jet::<f64, 4>::variable(x0).sqrt();
        assert!(f64::abs(y.derivative(0) - f64::sqrt(x0)) < TOL);
        assert!(f64::abs(y.derivative(1) - 1.0 / (2.0 * f64::sqrt(x0))) < TOL);
        assert!(f64::abs(y.derivative(2) - (-1.0 / (4.0 * x0 * f64::sqrt(x0)))) < TOL);
        assert!(f64::abs(y.derivative(3) - 3.0 / (8.0 * x0 * x0 * f64::sqrt(x0))) < TOL);
    }

    #[test]
    fn test_ln_orders() {
        // f(x) = ln(x): f'=1/x, f''=-1/x^2, f'''=2/x^3
        let x0 = 2.0_f64;
        let y = Jet::<f64, 4>::variable(x0).ln();
        assert!(f64::abs(y.derivative(0) - f64::ln(x0)) < TOL);
        assert!(f64::abs(y.derivative(1) - 1.0 / x0) < TOL);
        assert!(f64::abs(y.derivative(2) - (-1.0 / (x0 * x0))) < TOL);
        assert!(f64::abs(y.derivative(3) - 2.0 / (x0 * x0 * x0)) < TOL);
    }

    #[test]
    fn test_tan_orders() {
        // f(x) = tan(x): f' = 1+tan^2, f'' = 2 tan (1+tan^2)
        let x0 = 0.5_f64;
        let t = f64::tan(x0);
        let y = Jet::<f64, 3>::variable(x0).tan();
        assert!(f64::abs(y.derivative(0) - t) < TOL);
        assert!(f64::abs(y.derivative(1) - (1.0 + t * t)) < TOL);
        assert!(f64::abs(y.derivative(2) - 2.0 * t * (1.0 + t * t)) < TOL);
    }

    #[test]
    fn test_matches_dual_at_order_one() {
        // Jet<T,2> carries the same first derivative as Dual.
        fn f<T: Numeric>(t: T) -> T {
            t.sin() * t.exp() + t.powi(3)
        }
        let x0 = 0.8_f64;
        let j = f(Jet::<f64, 2>::variable(x0));
        let d = f(Dual::<f64>::variable(x0));
        assert!(f64::abs(j.value() - d.value) < TOL);
        assert!(f64::abs(j.coeffs[1] - d.deriv) < TOL);
    }

    #[test]
    fn test_generic_over_numeric() {
        // The same function runs with a plain float or with a Jet.
        fn g<T: Numeric>(t: T) -> T {
            t.powi(3) + T::from_f64(2.0) * t
        }
        let x0 = 1.5_f64;
        let plain = g(x0);
        let j = g(Jet::<f64, 4>::variable(x0));
        assert!(f64::abs(j.value() - plain) < TOL);
        assert!(f64::abs(j.derivative(1) - (3.0 * x0 * x0 + 2.0)) < TOL); // f'
        assert!(f64::abs(j.derivative(2) - 6.0 * x0) < TOL); // f''
        assert!(f64::abs(j.derivative(3) - 6.0) < TOL); // f'''
    }

    #[test]
    fn test_generic_over_f32() {
        // Jet is generic over the scalar; here it carries f32.
        let y = Jet::<f32, 4>::variable(2.0).powi(3);
        assert!(f32::abs(y.derivative(0) - 8.0) < TOL_F32);
        assert!(f32::abs(y.derivative(1) - 12.0) < TOL_F32);
        assert!(f32::abs(y.derivative(2) - 12.0) < TOL_F32);
        assert!(f32::abs(y.derivative(3) - 6.0) < TOL_F32);
    }

    #[test]
    fn test_single_coefficient_is_scalar() {
        // N = 1 carries only the value and behaves like the bare scalar.
        let y = Jet::<f64, 1>::constant(2.0) * Jet::<f64, 1>::constant(3.0);
        assert!(f64::abs(y.value() - 6.0) < TOL);
        assert!(f64::abs(Jet::<f64, 1>::constant(1.0).exp().value() - f64::exp(1.0)) < TOL);
    }

    #[test]
    fn test_constant_has_zero_higher_coeffs() {
        let c = Jet::<f64, 4>::constant(2.0);
        for k in 1..4 {
            assert!(f64::abs(c.coeffs[k]) < TOL);
        }
    }

    #[test]
    fn test_sqrt_zero_blows_up() {
        // the derivative of sqrt at 0 is unbounded, while the value stays finite
        let y = Jet::<f64, 3>::variable(0.0).sqrt();
        assert!(f64::abs(y.value()) < TOL);
        assert!(y.coeffs[1].is_infinite());
        // is_finite reflects the value only
        assert!(y.is_finite());
    }
}
