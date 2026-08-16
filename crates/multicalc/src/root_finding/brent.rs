//! Bracketed scalar Brent's method root solver.

use crate::error::SolveError;
use crate::root_finding::{RootReport, RootTermination, same_sign};
use crate::scalar::{Numeric, ScalarFn};

/// A bracketed scalar root solver using Brent's method (Dekker-Brent algorithm).
///
/// Combines root bracketing, bisection, secant method, and inverse quadratic interpolation (IQI).
/// It guarantees convergence by retaining bracket endpoints like [`Bisection`](crate::root_finding::Bisection),
/// but converges superlinearly on smooth functions.
///
/// Cost per iteration: 1 function evaluation.
///
/// # Examples
/// ```
/// use multicalc::Brent;
/// use multicalc::scalar::c;
/// use multicalc::scalar_fn;
///
/// // f(x) = x² − 2, root at √2 ≈ 1.41421356
/// let function = scalar_fn!(|x| c(-2.0) + x * x);
/// let lower_bound = 0.0_f64;
/// let upper_bound = 2.0;
///
/// let report = Brent::default().solve(&function, lower_bound, upper_bound).unwrap();
/// assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-9);
/// ```
pub struct Brent<T = f64> {
    xtol: T,
    ftol: T,
    max_iterations: usize,
}

impl<T: Numeric> Default for Brent<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> Brent<T> {
    /// Const constructor (same as [`Default::default`]).
    ///
    /// ```
    /// use multicalc::Brent;
    ///
    /// const B: Brent = Brent::new();
    /// ```
    pub const fn new() -> Self {
        let tol = T::EPSILON_X4;
        Brent {
            xtol: tol,
            ftol: tol,
            max_iterations: 100,
        }
    }

    /// Sets the bracket-width tolerance (relative: compared against `xtol * (1 + |b|)`).
    #[must_use]
    pub const fn with_xtol(mut self, xtol: T) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the residual tolerance: the solver stops when `|f(b)| ≤ ftol`.
    #[must_use]
    pub const fn with_ftol(mut self, ftol: T) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Finds a root of `f` in the bracket `[a, b]`.
    ///
    /// Returns the root estimate and termination reason, or an error:
    /// [`NonFinite`](SolveError::NonFinite) if `f` returns a non-finite value,
    /// [`InvalidBracket`](SolveError::InvalidBracket) if `f(a)` and `f(b)` share a sign, or
    /// [`DidNotConverge`](SolveError::DidNotConverge) if the budget is exhausted.
    pub fn solve<F: ScalarFn>(&self, f: &F, a: T, b: T) -> Result<RootReport<T>, SolveError> {
        let mut fa = f.eval(a);
        let mut fb = f.eval(b);

        if !fa.is_finite() || !fb.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if fa == T::ZERO {
            return Ok(RootReport {
                root: a,
                residual: fa,
                iterations: 0,
                termination: RootTermination::ResidualTolerance,
            });
        }
        if fb == T::ZERO {
            return Ok(RootReport {
                root: b,
                residual: fb,
                iterations: 0,
                termination: RootTermination::ResidualTolerance,
            });
        }
        if same_sign(fa, fb) {
            return Err(SolveError::InvalidBracket);
        }

        let mut a = a;
        let mut b = b;

        if fa.abs() < fb.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut fa, &mut fb);
        }

        let mut c = a;
        let mut fc = fa;
        let mut d = b - a;
        let mut mflag = true;

        for iter in 1..=self.max_iterations {
            let tol = self.xtol * (T::ONE + b.abs());
            let m = (a - b) * T::HALF;

            if fb.abs() <= self.ftol {
                return Ok(RootReport {
                    root: b,
                    residual: fb,
                    iterations: iter - 1,
                    termination: RootTermination::ResidualTolerance,
                });
            }

            if m.abs() <= tol || (b - a).abs() <= tol {
                return Ok(RootReport {
                    root: b,
                    residual: fb,
                    iterations: iter - 1,
                    termination: RootTermination::BracketWidth,
                });
            }

            let mut s = if fa != fc && fb != fc {
                inverse_quadratic_interpolation(a, fa, b, fb, c, fc)
            } else {
                b - fb * (b - a) / (fb - fa)
            };

            let bound1 = a * T::HALF * T::HALF * T::THREE + b * T::HALF * T::HALF;
            let bound2 = b;
            let (min_b, max_b) = if bound1 <= bound2 {
                (bound1, bound2)
            } else {
                (bound2, bound1)
            };

            let condition1 = s < min_b || s > max_b;
            let condition2 = mflag && (s - b).abs() >= (b - c).abs() * T::HALF;
            let condition3 = !mflag && (s - b).abs() >= (c - d).abs() * T::HALF;
            let condition4 = mflag && (b - c).abs() < tol;
            let condition5 = !mflag && (c - d).abs() < tol;

            if !s.is_finite() || condition1 || condition2 || condition3 || condition4 || condition5
            {
                s = a * T::HALF + b * T::HALF;
                mflag = true;
            } else {
                mflag = false;
            }

            let fs = f.eval(s);
            if !fs.is_finite() {
                return Err(SolveError::NonFinite);
            }

            d = c;
            c = b;
            fc = fb;

            if same_sign(fa, fs) {
                a = s;
                fa = fs;
            } else {
                b = s;
                fb = fs;
            }

            if fa.abs() < fb.abs() {
                core::mem::swap(&mut a, &mut b);
                core::mem::swap(&mut fa, &mut fb);
            }

            // Check convergence immediately after updating the best estimate
            if fb.abs() <= self.ftol {
                return Ok(RootReport {
                    root: b,
                    residual: fb,
                    iterations: iter,
                    termination: RootTermination::ResidualTolerance,
                });
            }

            let tol_after = self.xtol * (T::ONE + b.abs());
            let m_after = (a - b) * T::HALF;
            if m_after.abs() <= tol_after || (b - a).abs() <= tol_after {
                return Ok(RootReport {
                    root: b,
                    residual: fb,
                    iterations: iter,
                    termination: RootTermination::BracketWidth,
                });
            }
        }

        Err(SolveError::DidNotConverge {
            iters: self.max_iterations,
        })
    }
}

/// Evaluates inverse quadratic interpolation (IQI) through three points `(a, fa)`, `(b, fb)`, `(c, fc)`.
#[inline]
#[must_use]
pub(crate) fn inverse_quadratic_interpolation<T: Numeric>(
    a: T,
    fa: T,
    b: T,
    fb: T,
    c: T,
    fc: T,
) -> T {
    let s_a = a * fb * fc / ((fa - fb) * (fa - fc));
    let s_b = b * fa * fc / ((fb - fa) * (fb - fc));
    let s_c = c * fa * fb / ((fc - fa) * (fc - fb));
    s_a + s_b + s_c
}

#[cfg(test)]
mod tests {
    use super::inverse_quadratic_interpolation;

    #[test]
    fn inverse_quadratic_interpolation_matches_analytical_value() {
        let result = inverse_quadratic_interpolation(-1.0_f64, 0.75, 0.0, -0.25, -0.25, -0.1875);

        assert!((result - (-0.85)).abs() < 1e-12);
    }
}
