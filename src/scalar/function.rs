//! A scalar-function abstraction so one formula can be evaluated at `f64` (finite differences) or
//! at an autodiff scalar (`Dual`/`HyperDual`/`Jet`).
//!
//! A type implementing [`ScalarFn`] / [`ScalarFnN`] is generic over the scalar through its `eval`
//! method, so a single value drives every backend.

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

#[cfg(test)]
mod test {
    use super::{ScalarFn, ScalarFnN};
    use crate::scalar::{Dual, HyperDual, Jet, Numeric};

    // f(x) = 4x^3 - 3x^2
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

    // g(x, y, z) = y*sin(x) + 2*x*e^z
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
}
