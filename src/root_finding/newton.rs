//! Scalar Newton and damped Newton solvers.

use crate::numerical_derivative::autodiff::AutoDiffSingle;
use crate::numerical_derivative::derivator::DerivatorSingleVariable;
use crate::root_finding::{RootReport, RootTermination};
use crate::scalar::{Numeric, ScalarFn};
use crate::utils::error_codes::CalcError;

/// Maximum step halvings per iteration when backtracking is enabled.
const MAX_BACKTRACK: usize = 20;

/// A scalar Newton solver, optionally damped by a backtracking line search.
///
/// The derivative defaults to exact autodiff via [`AutoDiffSingle`] (`Dual` numbers).
/// Any [`DerivatorSingleVariable`] can be passed to [`from_derivator`](Newton::from_derivator)
/// to use a different backend, including finite differences.
///
/// With `backtracking` off (the default), each step is `x − f(x)/f′(x)`. With it on,
/// the step length is halved until `|f|` decreases, rescuing iterates that would otherwise
/// overshoot the root.
///
/// Cost per iteration: 1 function evaluation + 1 derivative evaluation
/// (+ ≤ `MAX_BACKTRACK` extra function evaluations when backtracking is on).
///
/// # Examples
/// ```
/// use multicalc::root_finding::Newton;
/// use multicalc::scalar::c;
/// use multicalc::scalar_fn;
///
/// // f(x) = x² − 2, root at √2 ≈ 1.41421356
/// let f = scalar_fn!(|x| c(-2.0) + x * x);
/// let solver: Newton = Newton::default();
/// let report = solver.solve(&f, 2.0_f64).unwrap();
/// assert!((report.root - 2.0_f64.sqrt()).abs() < 1e-12);
/// ```
pub struct Newton<D: DerivatorSingleVariable = AutoDiffSingle> {
    derivator: D,
    xtol: D::Scalar,
    ftol: D::Scalar,
    max_iterations: usize,
    backtracking: bool,
}

impl<D: DerivatorSingleVariable + Default> Default for Newton<D> {
    fn default() -> Self {
        Self::from_derivator(D::default())
    }
}

impl<D: DerivatorSingleVariable> Newton<D> {
    /// Builds a solver with the given derivator and default settings:
    /// tolerances of `30 × EPSILON`, budget of 100 iterations, backtracking off.
    pub fn from_derivator(derivator: D) -> Self {
        let tol = D::Scalar::EPSILON * D::Scalar::from_f64(30.0);
        Newton { derivator, xtol: tol, ftol: tol, max_iterations: 100, backtracking: false }
    }

    /// Sets the step-size tolerance (relative: compared against `xtol × (1 + |x|)`).
    #[must_use]
    pub fn with_xtol(mut self, xtol: D::Scalar) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the residual tolerance: the solver stops when `|f(x)| ≤ ftol`.
    #[must_use]
    pub fn with_ftol(mut self, ftol: D::Scalar) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Enables or disables backtracking: when on, each Newton step is halved until `|f|`
    /// decreases or the safeguard runs out. Off by default.
    #[must_use]
    pub fn with_backtracking(mut self, backtracking: bool) -> Self {
        self.backtracking = backtracking;
        self
    }

    /// Finds a root of `f` starting from `x0`.
    ///
    /// Returns the root estimate and termination reason, or an error:
    /// [`NonFiniteValue`](CalcError::NonFiniteValue) if `f` or its derivative is non-finite,
    /// [`SingularMatrix`](CalcError::SingularMatrix) if the derivative is zero, or
    /// [`DidNotConverge`](CalcError::DidNotConverge) if the budget is exhausted.
    pub fn solve<F: ScalarFn>(
        &self,
        f: &F,
        x0: D::Scalar,
    ) -> Result<RootReport<D::Scalar>, CalcError> {
        let one = D::Scalar::ONE;
        let half = D::Scalar::HALF;

        let mut x = x0;
        let mut fx = f.eval(x);
        if !fx.is_finite() {
            return Err(CalcError::NonFiniteValue);
        }

        for iter in 1..=self.max_iterations {
            if fx.abs() <= self.ftol {
                return Ok(RootReport {
                    root: x,
                    residual: fx,
                    iterations: iter,
                    termination: RootTermination::ResidualTolerance,
                });
            }

            let dfx = self.derivator.get_single(f, x)?;
            if !dfx.is_finite() {
                return Err(CalcError::NonFiniteValue);
            }
            if dfx == D::Scalar::ZERO {
                return Err(CalcError::SingularMatrix);
            }

            let step = fx / dfx;

            // Try the full Newton step; when backtracking is on, halve alpha until |f|
            // decreases or the per-iteration safeguard runs out.
            let mut alpha = one;
            let mut tries = 0usize;
            let (x_new, fx_new) = loop {
                let candidate = x - alpha * step;
                let trial = f.eval(candidate);
                if !self.backtracking {
                    if !trial.is_finite() {
                        return Err(CalcError::NonFiniteValue);
                    }
                    break (candidate, trial);
                }
                if (trial.is_finite() && trial.abs() < fx.abs()) || tries >= MAX_BACKTRACK {
                    if !trial.is_finite() {
                        return Err(CalcError::NonFiniteValue);
                    }
                    break (candidate, trial);
                }
                alpha *= half;
                tries += 1;
            };

            let step_taken = (x_new - x).abs();
            x = x_new;
            fx = fx_new;

            if step_taken <= self.xtol * (one + x.abs()) {
                return Ok(RootReport {
                    root: x,
                    residual: fx,
                    iterations: iter,
                    termination: RootTermination::StepTolerance,
                });
            }
        }

        Err(CalcError::DidNotConverge)
    }
}
