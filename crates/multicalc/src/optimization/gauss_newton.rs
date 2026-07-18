//! The Gauss-Newton least-squares solver.

use crate::error::SolveError;
use crate::linear_algebra::Vector;
use crate::linear_algebra::qr::{PivotedQr, enorm, max};
use crate::numerical_derivative::autodiff::AutoDiffMulti;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::numerical_derivative::jacobian::Jacobian;
use crate::optimization::{MinimizationReport, TerminationReason, is_finite, report};
use crate::scalar::{Numeric, Primal, VectorFn};

/// Maximum step halvings per iteration when backtracking is enabled.
const MAX_BACKTRACK: usize = 20;

/// A Gauss-Newton least-squares solver — the undamped sibling of `LevenbergMarquardt`, fast when
/// the Jacobian is well conditioned near the solution.
///
/// It has no trust region, so on ill-conditioned or far-from-solution problems it can diverge or
/// fail on a rank-deficient step; use `LevenbergMarquardt` there, or enable backtracking. The
/// Jacobian defaults to exact autodiff ([`AutoDiffMulti`]).
///
/// # Examples
/// ```
/// use multicalc::optimization::GaussNewton;
/// use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
/// use multicalc::scalar::c;
/// use multicalc::scalar_fn_vec;
///
/// // Fit y = a + b*t to points on y = 2t + 1; a linear residual is solved in one step.
/// let f = scalar_fn_vec!(|v: &[f64; 2]| [
///     c(-1.0) + v[1],
///     c(-3.0) + v[0] + v[1],
///     c(-5.0) + c(2.0) * v[0] + v[1],
/// ]);
/// let report = GaussNewton::<AutoDiffMulti>::default().minimize(&f, &[0.0, 0.0]).unwrap();
/// assert!((report.solution[0] - 2.0).abs() < 1e-9);
/// assert!((report.solution[1] - 1.0).abs() < 1e-9);
/// ```
pub struct GaussNewton<D: DerivatorMultiVariable = AutoDiffMulti> {
    derivator: D,
    ftol: D::Scalar,
    xtol: D::Scalar,
    gtol: D::Scalar,
    patience: usize,
    backtracking: bool,
}

impl<D: DerivatorMultiVariable + Default> Default for GaussNewton<D> {
    fn default() -> Self {
        Self::from_derivator(D::default())
    }
}

impl<D: DerivatorMultiVariable> GaussNewton<D> {
    /// Builds a solver with a specific differentiation backend and default settings:
    /// tolerances of `30·EPSILON`, patience `100`, backtracking off.
    pub const fn from_derivator(derivator: D) -> Self {
        let tol = D::Scalar::EPSILON_X30;
        GaussNewton {
            derivator,
            ftol: tol,
            xtol: tol,
            gtol: tol,
            patience: 100,
            backtracking: false,
        }
    }

    /// Sets the relative tolerance on the sum of squared residuals.
    #[must_use]
    pub const fn with_ftol(mut self, ftol: D::Scalar) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the relative tolerance on the parameter step.
    #[must_use]
    pub const fn with_xtol(mut self, xtol: D::Scalar) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the tolerance on the scaled gradient norm.
    #[must_use]
    pub const fn with_gtol(mut self, gtol: D::Scalar) -> Self {
        self.gtol = gtol;
        self
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub const fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Enables a backtracking line search: if a full step does not reduce the residual norm, the
    /// step is halved until it does (or the safeguard gives up). Off by default.
    #[must_use]
    pub const fn with_backtracking(mut self, backtracking: bool) -> Self {
        self.backtracking = backtracking;
        self
    }

    /// Minimizes `‖f(x)‖²` starting from `x0`.
    ///
    /// Errors: [`NonFinite`](SolveError::NonFinite) on a non-finite residual or Jacobian,
    /// [`Linalg`](SolveError::Linalg) wrapping [`Underdetermined`](crate::error::LinalgError::Underdetermined)
    /// if there are fewer residuals than parameters, [`Linalg`](SolveError::Linalg) wrapping
    /// [`Singular`](crate::error::LinalgError::Singular) if a step hits a rank-deficient Jacobian, or
    /// [`DidNotConverge`](SolveError::DidNotConverge) if the budget runs out.
    pub fn minimize<F, const N: usize, const M: usize>(
        &self,
        f: &F,
        x0: &[D::Scalar; N],
    ) -> Result<MinimizationReport<N, D::Scalar>, SolveError>
    where
        D: Clone,
        D::Scalar: Primal,
        F: VectorFn<N, M>,
    {
        let zero = D::Scalar::ZERO;
        let half = D::Scalar::HALF;

        let jacobian = Jacobian::from_derivator(self.derivator.clone());

        let mut x = *x0;
        let mut residuals = f.eval(&x);
        if !is_finite(&residuals) {
            return Err(SolveError::NonFinite);
        }
        let mut fnorm = enorm(&residuals);
        let mut evaluations = 1usize;

        for _ in 0..self.patience {
            let j = jacobian.get(f, &x)?;
            if !j.is_finite() {
                return Err(SolveError::NonFinite);
            }
            let qr = PivotedQr::decompose(j)?;

            // Already at a perfect fit.
            if fnorm == zero {
                return Ok(report(x, fnorm, evaluations, TerminationReason::Gtol));
            }

            // Gradient convergence: max over columns of |(Jᵀr)_c| / (fnorm · ‖columnₙ‖).
            let gradient = j.transpose() * Vector::new(residuals);
            let mut gnorm = zero;
            for (c, &column_norm) in qr.column_norms.iter().enumerate() {
                if column_norm != zero {
                    gnorm = max(gnorm, (gradient[c] / (fnorm * column_norm)).abs());
                }
            }
            if gnorm <= self.gtol {
                return Ok(report(x, fnorm, evaluations, TerminationReason::Gtol));
            }

            // Gauss-Newton step (errors on a rank-deficient Jacobian).
            let p = qr.solve_least_squares(Vector::new(residuals))?;

            // Take a step, shrinking it (when backtracking) until it is finite and lowers the
            // cost. A non-finite or cost-increasing trial triggers another halving.
            let mut alpha = D::Scalar::ONE;
            let mut tries = 0;
            let (x_new, residuals_new, fnorm_new) = loop {
                let candidate: [D::Scalar; N] = core::array::from_fn(|k| x[k] - alpha * p[k]);
                let trial = f.eval(&candidate);
                evaluations += 1;
                let finite = is_finite(&trial);
                let candidate_fnorm = if finite {
                    enorm(&trial)
                } else {
                    D::Scalar::INFINITY
                };

                if !self.backtracking {
                    if !finite {
                        return Err(SolveError::NonFinite);
                    }
                    break (candidate, trial, candidate_fnorm);
                }
                if (finite && candidate_fnorm < fnorm) || tries >= MAX_BACKTRACK {
                    if !finite {
                        return Err(SolveError::NonFinite);
                    }
                    break (candidate, trial, candidate_fnorm);
                }
                alpha *= half;
                tries += 1;
            };

            let step_norm = alpha * enorm(p.as_array());
            let previous_fnorm = fnorm;
            x = x_new;
            residuals = residuals_new;
            fnorm = fnorm_new;
            let xnorm = enorm(&x);

            // Convergence on the step and on the cost reduction.
            if step_norm <= self.xtol * xnorm {
                return Ok(report(x, fnorm, evaluations, TerminationReason::Xtol));
            }
            if (previous_fnorm - fnorm).abs() <= self.ftol * previous_fnorm {
                return Ok(report(x, fnorm, evaluations, TerminationReason::Ftol));
            }
        }

        Err(SolveError::DidNotConverge {
            iters: evaluations,
            residual: fnorm.to_f64(),
        })
    }
}
