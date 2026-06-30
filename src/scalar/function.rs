//! A scalar-function abstraction so one formula can be evaluated at `f64` (finite differences) or
//! at an autodiff scalar (`Dual`/`HyperDual`/`Jet`).
//!
//! A type implementing [`ScalarFn`] / [`ScalarFnN`] is generic over the scalar through its `eval`
//! method, so a single value drives every backend. The [`scalar_fn!`] macro builds these from
//! closure-style syntax, and [`c`] marks numeric constants inside the body.

use core::ops::{Add, Div, Mul, Sub};

use crate::scalar::Numeric;

/// A scalar function of one variable, evaluable at any [`Numeric`] scalar.
pub trait ScalarFn {
    /// Evaluates the function at `x`.
    fn eval<S: Numeric>(&self, x: S) -> S;
}

/// A scalar function of `N` variables, evaluable at any [`Numeric`] scalar.
pub trait ScalarFnN<const N: usize> {
    /// Evaluates the function at `point`.
    fn eval<S: Numeric>(&self, point: &[S; N]) -> S;
}

/// A vector-valued function of `N` variables with `M` outputs, evaluable at any [`Numeric`] scalar.
///
/// Used where an array of separate functions cannot be: a `&dyn ScalarFnN` is impossible because a
/// generic `eval` is not object-safe, so the Jacobian takes one function returning a fixed array.
pub trait VectorFn<const N: usize, const M: usize> {
    /// Evaluates the function at `point`.
    fn eval<S: Numeric>(&self, point: &[S; N]) -> [S; M];
}

/// Wraps one output of a [`VectorFn`] as a [`ScalarFnN`], so a vector-valued function can be
/// differentiated component-by-component through the scalar derivators.
pub(crate) struct Component<'a, F, const N: usize, const M: usize> {
    func: &'a F,
    index: usize,
}

impl<'a, F: VectorFn<N, M>, const N: usize, const M: usize> Component<'a, F, N, M> {
    /// Wraps output `index` of `func`. `N`/`M` are inferred from the function's [`VectorFn`] impl.
    #[inline]
    pub fn new(func: &'a F, index: usize) -> Self {
        Component { func, index }
    }
}

impl<F: VectorFn<N, M>, const N: usize, const M: usize> ScalarFnN<N> for Component<'_, F, N, M> {
    #[inline]
    fn eval<S: Numeric>(&self, point: &[S; N]) -> S {
        self.func.eval(point)[self.index]
    }
}

/// A scalar constant marker produced by [`c`], for use on the **left** of an operator with a
/// [`Numeric`] scalar inside a [`scalar_fn!`] body (e.g. `c(2.0) * x`).
///
/// A bare `2.0 * x` cannot typecheck in a generic body (`2.0` is always `f64`); `Const` carries the
/// constant until it meets the scalar, then takes that scalar's type.
#[derive(Debug, Clone, Copy)]
pub struct Const(f64);

/// Marks a scalar constant in a [`scalar_fn!`] body. Place it on the left of the operator
/// (`c(2.0) * x`, `c(1.0) + x`); the constant takes the function's scalar type.
#[inline]
pub fn c(value: f64) -> Const {
    Const(value)
}

impl<S: Numeric> Mul<S> for Const {
    type Output = S;
    #[inline]
    fn mul(self, rhs: S) -> S {
        S::from_f64(self.0) * rhs
    }
}

impl<S: Numeric> Add<S> for Const {
    type Output = S;
    #[inline]
    fn add(self, rhs: S) -> S {
        S::from_f64(self.0) + rhs
    }
}

impl<S: Numeric> Sub<S> for Const {
    type Output = S;
    #[inline]
    fn sub(self, rhs: S) -> S {
        S::from_f64(self.0) - rhs
    }
}

impl<S: Numeric> Div<S> for Const {
    type Output = S;
    #[inline]
    fn div(self, rhs: S) -> S {
        S::from_f64(self.0) / rhs
    }
}

/// Builds a [`ScalarFn`] (one variable) or [`ScalarFnN`] (`N` variables) from closure-style syntax.
///
/// The body is generic over the scalar, so the same function drives finite differences and
/// autodiff. Write scalar constants with [`c`](crate::scalar::c), on the left of the operator.
///
/// ```
/// use multicalc::scalar_fn;
/// use multicalc::scalar::{c, Dual, ScalarFn};
///
/// // f(x) = 4x^3 - 3x^2
/// let f = scalar_fn!(|x| c(4.0) * x * x * x - c(3.0) * x * x);
/// assert!((f.eval(2.0_f64) - 20.0).abs() < 1e-12); // value
/// assert!((f.eval(Dual::variable(2.0_f64)).deriv - 36.0).abs() < 1e-12); // f'(2) = 36
/// ```
#[macro_export]
macro_rules! scalar_fn {
    (| $p:ident : & [ f64 ; $n:literal ] | $body:expr) => {{
        struct ScalarFnImpl;
        impl $crate::scalar::ScalarFnN<$n> for ScalarFnImpl {
            #[inline]
            fn eval<S: $crate::scalar::Numeric>(&self, $p: &[S; $n]) -> S {
                $body
            }
        }
        ScalarFnImpl
    }};
    (| $p:ident : f64 | $body:expr) => {{
        struct ScalarFnImpl;
        impl $crate::scalar::ScalarFn for ScalarFnImpl {
            #[inline]
            fn eval<S: $crate::scalar::Numeric>(&self, $p: S) -> S {
                $body
            }
        }
        ScalarFnImpl
    }};
    (| $p:ident | $body:expr) => {{
        struct ScalarFnImpl;
        impl $crate::scalar::ScalarFn for ScalarFnImpl {
            #[inline]
            fn eval<S: $crate::scalar::Numeric>(&self, $p: S) -> S {
                $body
            }
        }
        ScalarFnImpl
    }};
}

/// Builds a [`VectorFn`] from a closure returning a fixed-size array, for Jacobians.
///
/// ```
/// use multicalc::scalar_fn_vec;
/// use multicalc::scalar::{Dual, VectorFn};
///
/// // f(x, y) = [x*y, sin(y)]
/// let f = scalar_fn_vec!(|v: &[f64; 2]| [v[0] * v[1], v[1].sin()]);
/// let out = f.eval(&[3.0_f64, 0.5]);
/// assert!((out[0] - 1.5).abs() < 1e-12);
/// ```
#[macro_export]
macro_rules! scalar_fn_vec {
    (| $p:ident : & [ f64 ; $n:literal ] | [ $($e:expr),+ $(,)? ]) => {{
        struct VectorFnImpl;
        impl $crate::scalar::VectorFn<$n, { $crate::__scalar_fn_count!($($e),+) }> for VectorFnImpl {
            #[inline]
            fn eval<S: $crate::scalar::Numeric>(
                &self,
                $p: &[S; $n],
            ) -> [S; { $crate::__scalar_fn_count!($($e),+) }] {
                [$($e),+]
            }
        }
        VectorFnImpl
    }};
}

/// Counts a comma-separated list of expressions at compile time (used by [`scalar_fn_vec!`]).
#[doc(hidden)]
#[macro_export]
macro_rules! __scalar_fn_count {
    () => { 0usize };
    ($head:expr $(, $tail:expr)*) => { 1usize + $crate::__scalar_fn_count!($($tail),*) };
}

#[cfg(test)]
mod test {
    use super::{ScalarFn, ScalarFnN, VectorFn, c};
    use crate::scalar::{Dual, HyperDual, Jet, Numeric};

    // f(x) = 4x^3 - 3x^2, hand-written over the scalar.
    struct Cubic;
    impl ScalarFn for Cubic {
        fn eval<S: Numeric>(&self, x: S) -> S {
            S::from_f64(4.0) * x * x * x - S::from_f64(3.0) * x * x
        }
    }

    #[test]
    fn one_function_drives_every_backend() {
        let f = Cubic;
        // plain f64 (finite-difference path): 4*8 - 3*4 = 20
        assert!(f64::abs(f.eval(2.0_f64) - 20.0) < 1e-12);
        // Dual: f'(x) = 12x^2 - 6x = 36 at x = 2
        assert!(f64::abs(f.eval(Dual::variable(2.0_f64)).deriv - 36.0) < 1e-12);
        // HyperDual: f''(x) = 24x - 6 = 42 at x = 2
        assert!(f64::abs(f.eval(HyperDual::variable(2.0_f64)).eps1eps2 - 42.0) < 1e-12);
        // Jet: f'''(x) = 24
        assert!(f64::abs(f.eval(Jet::<f64, 4>::variable(2.0_f64)).derivative(3) - 24.0) < 1e-9);
    }

    // g(x, y, z) = y*sin(x) + 2*x*e^z, hand-written over the scalar.
    struct Mixed;
    impl ScalarFnN<3> for Mixed {
        fn eval<S: Numeric>(&self, v: &[S; 3]) -> S {
            v[1] * v[0].sin() + S::from_f64(2.0) * v[0] * v[2].exp()
        }
    }

    #[test]
    fn multivariable_partial_via_seeding() {
        let g = Mixed;
        let point = [1.0_f64, 2.0, 0.5];
        let expected = 2.0 * f64::sin(1.0) + 2.0 * f64::exp(0.5);
        assert!(f64::abs(g.eval(&point) - expected) < 1e-12);

        // partial dg/dx via Dual seeding of index 0:
        // dg/dx = y*cos(x) + 2*e^z = 2*cos(1) + 2*e^0.5
        let seeded = [
            Dual::variable(1.0_f64),
            Dual::constant(2.0),
            Dual::constant(0.5),
        ];
        let expected_dx = 2.0 * f64::cos(1.0) + 2.0 * f64::exp(0.5);
        assert!(f64::abs(g.eval(&seeded).deriv - expected_dx) < 1e-12);
    }

    #[test]
    fn macro_single_var() {
        // f(x) = 4x^3 - 3x^2, authored via the macro.
        let f = scalar_fn!(|x| c(4.0) * x * x * x - c(3.0) * x * x);
        assert!(f64::abs(f.eval(2.0_f64) - 20.0) < 1e-12);
        assert!(f64::abs(f.eval(Dual::variable(2.0_f64)).deriv - 36.0) < 1e-12);
        assert!(f64::abs(f.eval(HyperDual::variable(2.0_f64)).eps1eps2 - 42.0) < 1e-12);
        assert!(f64::abs(f.eval(Jet::<f64, 4>::variable(2.0_f64)).derivative(3) - 24.0) < 1e-9);
    }

    #[test]
    fn macro_single_var_typed_param() {
        let f = scalar_fn!(|x: f64| c(2.0) * x.sin());
        assert!(f64::abs(f.eval(0.5_f64) - 2.0 * f64::sin(0.5)) < 1e-12);
    }

    #[test]
    fn macro_multivariable() {
        let f = scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin() + c(2.0) * v[0] * v[2].exp());
        let point = [1.0_f64, 2.0, 0.5];
        let expected = 2.0 * f64::sin(1.0) + 2.0 * f64::exp(0.5);
        assert!(f64::abs(f.eval(&point) - expected) < 1e-12);
    }

    #[test]
    fn macro_vector_valued() {
        // f(x, y) = [x*y, sin(y)]
        let f = scalar_fn_vec!(|v: &[f64; 2]| [v[0] * v[1], v[1].sin()]);
        let out = f.eval(&[3.0_f64, 0.5]);
        assert!(f64::abs(out[0] - 1.5) < 1e-12);
        assert!(f64::abs(out[1] - f64::sin(0.5)) < 1e-12);

        // a Jacobian column via Dual: d/dx [x*y, sin(y)] = [y, 0] at (3, 0.5)
        let seeded = [Dual::variable(3.0_f64), Dual::constant(0.5)];
        let col = f.eval(&seeded);
        assert!(f64::abs(col[0].deriv - 0.5) < 1e-12);
        assert!(f64::abs(col[1].deriv) < 1e-12);
    }
}
