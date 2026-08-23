//! Bracketed scalar bisection solver.

use crate::error::SolveError;
use crate::root_finding::{RootReport, RootTermination, same_sign};
use crate::scalar::{Numeric, ScalarFn};

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
/// use multicalc::scalar::constant;
/// use multicalc::scalar_fn;
///
/// // f(x) = x² − 2, root at √2 ≈ 1.41421356
/// let function = scalar_fn!(|x| constant(-2.0) + x * x);
/// let lower_bound = 0.0_f64;
/// let upper_bound = 2.0;
///
/// let report = Bisection::default().solve(&function, lower_bound, upper_bound).unwrap();
/// assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-9);
/// ```
pub struct Bisection<T = f64> {
    xtol: T,
    ftol: T,
    max_iterations: usize,
}

impl<T: Numeric> Default for Bisection<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Numeric> Bisection<T> {
    /// Const constructor (same as [`Default::default`]).
    ///
    /// ```
    /// use multicalc::root_finding::Bisection;
    ///
    /// const B: Bisection = Bisection::new();
    /// ```
    pub const fn new() -> Self {
        let tol = T::EPSILON_X4;
        Bisection {
            xtol: tol,
            ftol: tol,
            max_iterations: 100,
        }
    }

    /// Sets the bracket-width tolerance (relative: compared against `xtol * (1 + |mid|)`).
    #[must_use]
    pub const fn with_xtol(mut self, xtol: T) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the residual tolerance: the solver stops when `|f(mid)| ≤ ftol`.
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
        let value_a = f.eval(a);
        let value_b = f.eval(b);

        if !value_a.is_finite() || !value_b.is_finite() {
            return Err(SolveError::NonFinite);
        }
        if value_a == T::ZERO {
            return Ok(RootReport {
                root: a,
                residual: value_a,
                iterations: 0,
                termination: RootTermination::ResidualTolerance,
            });
        }
        if value_b == T::ZERO {
            return Ok(RootReport {
                root: b,
                residual: value_b,
                iterations: 0,
                termination: RootTermination::ResidualTolerance,
            });
        }
        if same_sign(value_a, value_b) {
            return Err(SolveError::InvalidBracket);
        }

        // Order so lower < upper numerically; the midpoint formula lower + (upper - lower)/2 then
        // always lands strictly inside the interval.
        let (mut lower, mut value_lower, mut upper) = if a <= b {
            (a, value_a, b)
        } else {
            (b, value_b, a)
        };

        for iter in 1..=self.max_iterations {
            let mid = lower + (upper - lower) * T::HALF;
            let fmid = f.eval(mid);
            if !fmid.is_finite() {
                return Err(SolveError::NonFinite);
            }
            if fmid.abs() <= self.ftol {
                return Ok(RootReport {
                    root: mid,
                    residual: fmid,
                    iterations: iter,
                    termination: RootTermination::ResidualTolerance,
                });
            }
            if (upper - lower) <= self.xtol * (T::ONE + mid.abs()) {
                return Ok(RootReport {
                    root: mid,
                    residual: fmid,
                    iterations: iter,
                    termination: RootTermination::BracketWidth,
                });
            }
            // Replace the endpoint that shares a sign with fmid to keep the sign change
            // bracketed.
            if same_sign(fmid, value_lower) {
                lower = mid;
                value_lower = fmid;
            } else {
                upper = mid;
            }
        }

        Err(SolveError::DidNotConverge {
            iters: self.max_iterations,
        })
    }
}
