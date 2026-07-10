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
    const MAX: Self = Self::constant(T::MAX);
    const MIN_POSITIVE: Self = Self::constant(T::MIN_POSITIVE);

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

    /// Four-quadrant arctangent. `u = atan2(y, x)` satisfies `(x²+y²)·u′ = x·y′ − y·x′`,
    /// which gives a coefficient recurrence (like `ln`/`exp`): `u₀ = atan2(y₀, x₀)`, and each
    /// higher `uₖ` is solved from that relation. `w₀ = x₀²+y₀²` is zero only at the origin.
    #[inline]
    fn atan2(self, other: Self) -> Self {
        let y = self.coeffs;
        let x = other.coeffs;
        // w = x² + y², via the existing jet operators.
        let w = (other * other + self * self).coeffs;
        let mut u = [T::ZERO; N];
        u[0] = y[0].atan2(x[0]);
        for k in 1..N {
            let mut acc = T::ZERO;
            for i in 0..k {
                let m = T::from_usize(k - i);
                acc += m * (x[i] * y[k - i] - y[i] * x[k - i]);
            }
            for i in 1..k {
                acc -= w[i] * T::from_usize(k - i) * u[k - i];
            }
            u[k] = acc / (T::from_usize(k) * w[0]);
        }
        Jet { coeffs: u }
    }

    /// Magnitude of `self` with the sign of `sign`. Away from a sign flip the whole series is
    /// scaled by `s = ±1`; the value coefficient is set by `copysign` so signed zero is exact.
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        let s = if (self.coeffs[0] < T::ZERO) == (sign.coeffs[0] < T::ZERO) {
            T::ONE
        } else {
            -T::ONE
        };
        let mut coeffs = core::array::from_fn(|k| s * self.coeffs[k]);
        coeffs[0] = self.coeffs[0].copysign(sign.coeffs[0]);
        Jet { coeffs }
    }

    /// Largest integer `<= self`; all higher coefficients (the derivatives) are zero.
    #[inline]
    fn floor(self) -> Self {
        Jet::constant(self.coeffs[0].floor())
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
