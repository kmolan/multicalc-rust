//! Newton and damped Newton solvers for square systems.

use crate::linear_algebra::qr::enorm;
use crate::linear_algebra::{Matrix, Vector};
use crate::numerical_derivative::autodiff::AutoDiffMulti;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::numerical_derivative::jacobian::Jacobian;
use crate::root_finding::{RootReportN, RootTermination, all_finite};
use crate::scalar::{Numeric, VectorFn};
use crate::utils::error_codes::CalcError;

/// Maximum step halvings per iteration when backtracking is enabled.
const MAX_BACKTRACK: usize = 20;

/// A Newton solver for square systems `F: Rⁿ → Rⁿ`, optionally damped by a backtracking
/// line search.
///
/// Each iteration solves `J·Δ = −F(x)` for the Newton step and updates `x ← x + α·Δ`.
/// The Jacobian defaults to exact autodiff via [`AutoDiffMulti`]; any
/// [`DerivatorMultiVariable`] can be passed to [`from_derivator`](NewtonSystem::from_derivator).
///
/// With `backtracking` off (the default), the full Newton step is taken each iteration.
/// With it on, the step length is halved until `‖F‖` decreases, rescuing iterates that
/// would otherwise overshoot a root.
///
/// Cost per iteration: 1 residual evaluation + N² partial derivatives (1 Jacobian) + 1
/// N×N LU solve (+ ≤ `MAX_BACKTRACK` extra residual evaluations when backtracking is on).
///
/// # Examples
/// ```
/// use multicalc::root_finding::NewtonSystem;
/// use multicalc::scalar::c;
/// use multicalc::scalar_fn_vec;
///
/// // Find (x, y) where x² + y² = 4 and xy = 1.
/// let f = scalar_fn_vec!(|v: &[f64; 2]| [
///     c(-4.0) + v[0] * v[0] + v[1] * v[1],
///     c(-1.0) + v[0] * v[1],
/// ]);
/// let solver: NewtonSystem = NewtonSystem::default();
/// let report = solver.solve(&f, &[1.5_f64, 0.8]).unwrap();
/// assert!(report.residual_norm < 1e-12);
/// ```
pub struct NewtonSystem<D: DerivatorMultiVariable = AutoDiffMulti> {
    derivator: D,
    xtol: D::Scalar,
    ftol: D::Scalar,
    max_iterations: usize,
    backtracking: bool,
}

impl<D: DerivatorMultiVariable + Default> Default for NewtonSystem<D> {
    fn default() -> Self {
        Self::from_derivator(D::default())
    }
}

impl<D: DerivatorMultiVariable> NewtonSystem<D> {
    /// Builds a solver with the given derivator and default settings:
    /// tolerances of `30 × EPSILON`, budget of 100 iterations, backtracking off.
    pub fn from_derivator(derivator: D) -> Self {
        let tol = D::Scalar::EPSILON * D::Scalar::from_f64(30.0);
        NewtonSystem {
            derivator,
            xtol: tol,
            ftol: tol,
            max_iterations: 100,
            backtracking: false,
        }
    }

    /// Sets the step-size tolerance (relative: compared against `xtol × (1 + ‖x‖)`).
    #[must_use]
    pub fn with_xtol(mut self, xtol: D::Scalar) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the residual tolerance: the solver stops when `‖F(x)‖ ≤ ftol`.
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

    /// Enables or disables backtracking: when on, each Newton step is halved until `‖F‖`
    /// decreases or the safeguard runs out. Off by default.
    #[must_use]
    pub fn with_backtracking(mut self, backtracking: bool) -> Self {
        self.backtracking = backtracking;
        self
    }

    /// Finds a root of the square system `F` starting from `x0`.
    ///
    /// Returns the root estimate and termination reason, or an error:
    /// [`NonFiniteValue`](CalcError::NonFiniteValue) if `F` or its Jacobian is non-finite,
    /// [`SingularMatrix`](CalcError::SingularMatrix) if the Jacobian is singular at a step, or
    /// [`DidNotConverge`](CalcError::DidNotConverge) if the budget is exhausted.
    pub fn solve<F, const N: usize>(
        &self,
        f: &F,
        x0: &[D::Scalar; N],
    ) -> Result<RootReportN<N, D::Scalar>, CalcError>
    where
        D: Clone,
        F: VectorFn<N, N>,
    {
        let one = D::Scalar::ONE;
        let half = D::Scalar::HALF;

        let jacobian = Jacobian::from_derivator(self.derivator.clone());

        let mut x = *x0;
        let mut r = f.eval(&x);
        if !all_finite(&r) {
            return Err(CalcError::NonFiniteValue);
        }
        let mut fnorm = enorm(&r);

        for iter in 1..=self.max_iterations {
            if fnorm <= self.ftol {
                return Ok(RootReportN {
                    root: x,
                    residual_norm: fnorm,
                    iterations: iter,
                    termination: RootTermination::ResidualTolerance,
                });
            }

            let jac = jacobian.get(f, &x)?;
            if jac.iter().any(|row| !all_finite(row)) {
                return Err(CalcError::NonFiniteValue);
            }

            let j: Matrix<N, N, D::Scalar> = Matrix::from_fn(|ri, c| jac[ri][c]);
            let neg_r: [D::Scalar; N] = core::array::from_fn(|i| -r[i]);
            // Solve J·step = -F; returns SingularMatrix if J is rank-deficient.
            let step = j.solve(Vector::new(neg_r))?;

            // Try the full Newton step; when backtracking is on, halve alpha until ‖F‖
            // decreases or the per-iteration safeguard runs out.
            let mut alpha = one;
            let mut tries = 0usize;
            let (x_new, r_new, fnorm_new) = loop {
                let candidate: [D::Scalar; N] = core::array::from_fn(|k| x[k] + alpha * step[k]);
                let trial = f.eval(&candidate);
                let trial_finite = all_finite(&trial);
                let trial_fnorm = if trial_finite {
                    enorm(&trial)
                } else {
                    D::Scalar::INFINITY
                };

                if !self.backtracking {
                    if !trial_finite {
                        return Err(CalcError::NonFiniteValue);
                    }
                    break (candidate, trial, trial_fnorm);
                }
                if (trial_finite && trial_fnorm < fnorm) || tries >= MAX_BACKTRACK {
                    if !trial_finite {
                        return Err(CalcError::NonFiniteValue);
                    }
                    break (candidate, trial, trial_fnorm);
                }
                alpha *= half;
                tries += 1;
            };

            let step_norm = alpha * enorm(step.as_array());
            let xnorm = enorm(&x_new);
            x = x_new;
            r = r_new;
            fnorm = fnorm_new;

            if step_norm <= self.xtol * (one + xnorm) {
                return Ok(RootReportN {
                    root: x,
                    residual_norm: fnorm,
                    iterations: iter,
                    termination: RootTermination::StepTolerance,
                });
            }
        }

        Err(CalcError::DidNotConverge)
    }
}
