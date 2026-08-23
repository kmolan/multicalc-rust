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
/// use multicalc::scalar::constant;
/// use multicalc::scalar_fn;
///
/// // f(x) = x² − 2, root at √2 ≈ 1.41421356
/// let function = scalar_fn!(|x| constant(-2.0) + x * x);
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
        let mut f_at_a = f.eval(a);
        let mut f_at_b = f.eval(b);

        if !f_at_a.is_finite() || !f_at_b.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if f_at_a == T::ZERO {
            return Ok(RootReport {
                root: a,
                residual: f_at_a,
                iterations: 0,
                termination: RootTermination::ResidualTolerance,
            });
        }
        if f_at_b == T::ZERO {
            return Ok(RootReport {
                root: b,
                residual: f_at_b,
                iterations: 0,
                termination: RootTermination::ResidualTolerance,
            });
        }
        if same_sign(f_at_a, f_at_b) {
            return Err(SolveError::InvalidBracket);
        }

        let mut a = a;
        let mut b = b;

        if f_at_a.abs() < f_at_b.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut f_at_a, &mut f_at_b);
        }

        let mut point_c = a;
        let mut f_at_c = f_at_a;
        let mut prev_step = b - a;
        let mut mflag = true;

        for iter in 1..=self.max_iterations {
            let tol = self.xtol * (T::ONE + b.abs());
            let midpoint = (a - b) * T::HALF;

            if f_at_b.abs() <= self.ftol {
                return Ok(RootReport {
                    root: b,
                    residual: f_at_b,
                    iterations: iter - 1,
                    termination: RootTermination::ResidualTolerance,
                });
            }

            if midpoint.abs() <= tol || (b - a).abs() <= tol {
                return Ok(RootReport {
                    root: b,
                    residual: f_at_b,
                    iterations: iter - 1,
                    termination: RootTermination::BracketWidth,
                });
            }

            let mut trial = if f_at_a != f_at_c && f_at_b != f_at_c {
                inverse_quadratic_interpolation(a, f_at_a, b, f_at_b, point_c, f_at_c)
            } else {
                b - f_at_b * (b - a) / (f_at_b - f_at_a)
            };

            let bound1 = a * T::HALF * T::HALF * T::THREE + b * T::HALF * T::HALF;
            let bound2 = b;
            let (min_b, max_b) = if bound1 <= bound2 {
                (bound1, bound2)
            } else {
                (bound2, bound1)
            };

            let condition1 = trial < min_b || trial > max_b;
            let condition2 = mflag && (trial - b).abs() >= (b - point_c).abs() * T::HALF;
            let condition3 = !mflag && (trial - b).abs() >= (point_c - prev_step).abs() * T::HALF;
            let condition4 = mflag && (b - point_c).abs() < tol;
            let condition5 = !mflag && (point_c - prev_step).abs() < tol;

            if !trial.is_finite()
                || condition1
                || condition2
                || condition3
                || condition4
                || condition5
            {
                trial = a * T::HALF + b * T::HALF;
                mflag = true;
            } else {
                mflag = false;
            }

            let f_at_trial = f.eval(trial);
            if !f_at_trial.is_finite() {
                return Err(SolveError::NonFinite);
            }

            prev_step = point_c;
            point_c = b;
            f_at_c = f_at_b;

            if same_sign(f_at_a, f_at_trial) {
                a = trial;
                f_at_a = f_at_trial;
            } else {
                b = trial;
                f_at_b = f_at_trial;
            }

            if f_at_a.abs() < f_at_b.abs() {
                core::mem::swap(&mut a, &mut b);
                core::mem::swap(&mut f_at_a, &mut f_at_b);
            }

            // Check convergence immediately after updating the best estimate
            if f_at_b.abs() <= self.ftol {
                return Ok(RootReport {
                    root: b,
                    residual: f_at_b,
                    iterations: iter,
                    termination: RootTermination::ResidualTolerance,
                });
            }

            let tol_after = self.xtol * (T::ONE + b.abs());
            let midpoint_after = (a - b) * T::HALF;
            if midpoint_after.abs() <= tol_after || (b - a).abs() <= tol_after {
                return Ok(RootReport {
                    root: b,
                    residual: f_at_b,
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
    f_at_a: T,
    b: T,
    f_at_b: T,
    point_c: T,
    f_at_c: T,
) -> T {
    let s_a = a * f_at_b * f_at_c / ((f_at_a - f_at_b) * (f_at_a - f_at_c));
    let s_b = b * f_at_a * f_at_c / ((f_at_b - f_at_a) * (f_at_b - f_at_c));
    let s_c = point_c * f_at_a * f_at_b / ((f_at_c - f_at_a) * (f_at_c - f_at_b));
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
