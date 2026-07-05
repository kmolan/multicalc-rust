//! Bracketed scalar bisection solver.

use crate::root_finding::{RootReport, RootTermination, same_sign};
use crate::scalar::{Numeric, ScalarFn};
use crate::utils::error_codes::CalcError;

/// A bracketed scalar root solver using bisection.
///
/// Starting from an interval `[a, b]` that encloses a sign change, each iteration halves
/// the bracket. The budget is guaranteed: from a bracket of width `w`, at most
/// `ceil(log2(w / xtol))` steps are needed to satisfy the step tolerance, which always
/// fits within the default budget of 100 iterations.
///
/// Cost per iteration: 1 function evaluation.
///
/// # Examples
/// ```
/// use multicalc::root_finding::Bisection;
/// use multicalc::scalar::c;
/// use multicalc::scalar_fn;
///
/// // f(x) = x² − 2, root at √2 ≈ 1.41421356
/// let f = scalar_fn!(|x| c(-2.0) + x * x);
/// let report = Bisection::default().solve(&f, 0.0_f64, 2.0).unwrap();
/// assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-9);
/// ```
pub struct Bisection<T = f64> {
    xtol: T,
    ftol: T,
    max_iterations: usize,
}

impl<T: Numeric> Default for Bisection<T> {
    fn default() -> Self {
        let tol = T::EPSILON * T::from_f64(4.0);
        Bisection { xtol: tol, ftol: tol, max_iterations: 100 }
    }
}

impl<T: Numeric> Bisection<T> {
    /// Sets the bracket-width tolerance (relative: compared against `xtol * (1 + |mid|)`).
    #[must_use]
    pub fn with_xtol(mut self, xtol: T) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the residual tolerance: the solver stops when `|f(mid)| ≤ ftol`.
    #[must_use]
    pub fn with_ftol(mut self, ftol: T) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Finds a root of `f` in the bracket `[a, b]`.
    ///
    /// Returns the root estimate and termination reason, or an error:
    /// [`NonFiniteValue`](CalcError::NonFiniteValue) if `f` returns a non-finite value,
    /// [`InvalidBracket`](CalcError::InvalidBracket) if `f(a)` and `f(b)` share a sign, or
    /// [`DidNotConverge`](CalcError::DidNotConverge) if the budget is exhausted.
    pub fn solve<F: ScalarFn>(&self, f: &F, a: T, b: T) -> Result<RootReport<T>, CalcError> {
        let fa = f.eval(a);
        let fb = f.eval(b);

        if !fa.is_finite() || !fb.is_finite() {
            return Err(CalcError::NonFiniteValue);
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
            return Err(CalcError::InvalidBracket);
        }

        // Order so lo < hi numerically; the midpoint formula lo + (hi - lo)/2 then
        // always lands strictly inside the interval.
        let (mut lo, mut flo, mut hi) = if a <= b { (a, fa, b) } else { (b, fb, a) };

        for iter in 1..=self.max_iterations {
            let mid = lo + (hi - lo) * T::HALF;
            let fmid = f.eval(mid);
            if !fmid.is_finite() {
                return Err(CalcError::NonFiniteValue);
            }
            if fmid.abs() <= self.ftol {
                return Ok(RootReport {
                    root: mid,
                    residual: fmid,
                    iterations: iter,
                    termination: RootTermination::ResidualTolerance,
                });
            }
            if (hi - lo) <= self.xtol * (T::ONE + mid.abs()) {
                return Ok(RootReport {
                    root: mid,
                    residual: fmid,
                    iterations: iter,
                    termination: RootTermination::BracketWidth,
                });
            }
            // Replace the endpoint that shares a sign with fmid to keep the sign change
            // bracketed.
            if same_sign(fmid, flo) {
                lo = mid;
                flo = fmid;
            } else {
                hi = mid;
            }
        }

        Err(CalcError::DidNotConverge)
    }
}
