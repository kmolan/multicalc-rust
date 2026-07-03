//! The Levenberg-Marquardt least-squares solver (MINPACK `lmder` driver).

use crate::linear_algebra::qr::{PivotedQr, enorm, max, min};
use crate::linear_algebra::{Matrix, Vector};
use crate::numerical_derivative::autodiff::AutoDiffMulti;
use crate::numerical_derivative::derivator::DerivatorMultiVariable;
use crate::numerical_derivative::jacobian::Jacobian;
use crate::optimization::trust_region::determine_lambda_and_parameter_update;
use crate::optimization::{MinimizationReport, TerminationReason, is_finite, report};
use crate::scalar::{Numeric, VectorFn};
use crate::utils::error_codes::CalcError;

/// Safety cap on the trust-region retries within one outer iteration; the `xtol` test ends the
/// retries long before this on any real problem.
const MAX_TRUST_REGION_TRIALS: usize = 100;

/// A Levenberg-Marquardt solver, configured by a builder.
///
/// The Jacobian defaults to exact autodiff ([`AutoDiffMulti`]); pass a finite-difference
/// derivator to [`from_derivator`](LevenbergMarquardt::from_derivator) to use that instead.
///
/// # Examples
/// ```
/// use multicalc::optimization::LevenbergMarquardt;
/// use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
/// use multicalc::scalar::c;
/// use multicalc::scalar_fn_vec;
///
/// // Minimize the Rosenbrock residual; the minimum is at (1, 1).
/// let f = scalar_fn_vec!(|v: &[f64; 2]| [c(10.0) * (v[1] - v[0] * v[0]), c(1.0) - v[0]]);
/// let report = LevenbergMarquardt::<AutoDiffMulti>::default().minimize(&f, &[-1.2, 1.0]).unwrap();
/// assert!((report.solution[0] - 1.0).abs() < 1e-6);
/// assert!((report.solution[1] - 1.0).abs() < 1e-6);
/// ```
pub struct LevenbergMarquardt<D: DerivatorMultiVariable = AutoDiffMulti> {
    derivator: D,
    ftol: D::Scalar,
    xtol: D::Scalar,
    gtol: D::Scalar,
    stepbound: D::Scalar,
    patience: usize,
    scale_diag: bool,
}

impl<D: DerivatorMultiVariable + Default> Default for LevenbergMarquardt<D> {
    fn default() -> Self {
        Self::from_derivator(D::default())
    }
}

impl<D: DerivatorMultiVariable> LevenbergMarquardt<D> {
    /// Builds a solver with a specific differentiation backend and default settings:
    /// tolerances of `30·EPSILON`, step bound `100`, patience `100`, column scaling on.
    pub fn from_derivator(derivator: D) -> Self {
        let tol = D::Scalar::EPSILON * D::Scalar::from_f64(30.0);
        LevenbergMarquardt {
            derivator,
            ftol: tol,
            xtol: tol,
            gtol: tol,
            stepbound: D::Scalar::from_f64(100.0),
            patience: 100,
            scale_diag: true,
        }
    }

    /// Sets the relative tolerance on the sum of squared residuals.
    #[must_use]
    pub fn with_ftol(mut self, ftol: D::Scalar) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the relative tolerance on the parameter step.
    #[must_use]
    pub fn with_xtol(mut self, xtol: D::Scalar) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the tolerance on the scaled gradient norm.
    #[must_use]
    pub fn with_gtol(mut self, gtol: D::Scalar) -> Self {
        self.gtol = gtol;
        self
    }

    /// Sets the factor bounding the initial trust-region radius.
    #[must_use]
    pub fn with_stepbound(mut self, stepbound: D::Scalar) -> Self {
        self.stepbound = stepbound;
        self
    }

    /// Sets the maximum number of outer iterations.
    #[must_use]
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Turns automatic column scaling on or off (off means an identity scaling).
    #[must_use]
    pub fn with_scale_diag(mut self, scale_diag: bool) -> Self {
        self.scale_diag = scale_diag;
        self
    }

    /// Minimizes `‖f(x)‖²` starting from `x0`.
    ///
    /// Returns the converged point and which test stopped it, or a
    /// [`CalcError`]: [`NonFiniteValue`](CalcError::NonFiniteValue) if a residual or Jacobian is
    /// non-finite, [`Underdetermined`](CalcError::Underdetermined) if there are fewer residuals
    /// than parameters, or [`DidNotConverge`](CalcError::DidNotConverge) if the budget runs out.
    pub fn minimize<F, const N: usize, const M: usize>(
        &self,
        f: &F,
        x0: &[D::Scalar; N],
    ) -> Result<MinimizationReport<N, D::Scalar>, CalcError>
    where
        D: Clone,
        F: VectorFn<N, M>,
    {
        let zero = D::Scalar::ZERO;
        let one = D::Scalar::ONE;
        let p1 = D::Scalar::from_f64(0.1);
        let p25 = D::Scalar::from_f64(0.25);
        let p5 = D::Scalar::HALF;
        let p75 = D::Scalar::from_f64(0.75);
        let p0001 = D::Scalar::from_f64(1.0e-4);

        let jacobian = Jacobian::from_derivator(self.derivator.clone());

        let mut x = *x0;
        let mut residuals = f.eval(&x);
        if !is_finite(&residuals) {
            return Err(CalcError::NonFiniteValue);
        }
        let mut fnorm = enorm(&residuals);

        let mut diag = [zero; N];
        let mut delta = zero;
        let mut xnorm = zero;
        let mut par = zero;
        let mut evaluations = 1usize;
        let mut first = true;

        for _ in 0..self.patience {
            let jac = jacobian.get(f, &x)?;
            if !jac.iter().all(is_finite) {
                return Err(CalcError::NonFiniteValue);
            }
            let dls = PivotedQr::decompose(Matrix::from_fn(|r, c| jac[r][c]))?
                .into_damped(Vector::new(residuals));

            if first {
                for (slot, &column_norm) in diag.iter_mut().zip(dls.column_norms.iter()) {
                    *slot = if column_norm == zero { one } else { column_norm };
                }
                xnorm = enorm(&core::array::from_fn::<_, N, _>(|j| diag[j] * x[j]));
                delta = self.stepbound * xnorm;
                if delta == zero {
                    delta = self.stepbound;
                }
            }

            // Gradient convergence test.
            if dls.max_a_t_b_scaled(fnorm) <= self.gtol {
                return Ok(report(x, fnorm, evaluations, TerminationReason::Gtol));
            }

            // Grow the scaling by the current column norms.
            if self.scale_diag {
                for (slot, &column_norm) in diag.iter_mut().zip(dls.column_norms.iter()) {
                    *slot = max(*slot, column_norm);
                }
            }

            for _ in 0..MAX_TRUST_REGION_TRIALS {
                let update = determine_lambda_and_parameter_update(&dls, &diag, delta, par);
                par = update.lambda;
                let p = update.step;
                let pnorm = enorm(&core::array::from_fn::<_, N, _>(|j| diag[j] * p[j]));
                if first {
                    delta = min(delta, pnorm);
                    first = false;
                }

                let x_new: [D::Scalar; N] = core::array::from_fn(|j| x[j] - p[j]);
                let residuals_new = f.eval(&x_new);
                evaluations += 1;
                if !is_finite(&residuals_new) {
                    return Err(CalcError::NonFiniteValue);
                }
                let fnorm1 = enorm(&residuals_new);

                // Actual reduction in the objective.
                let actred = if p1 * fnorm1 < fnorm {
                    one - (fnorm1 / fnorm) * (fnorm1 / fnorm)
                } else {
                    -one
                };

                // Predicted reduction and directional derivative.
                let temp1 = dls.a_x_norm(&p) / fnorm;
                let temp2 = (par.sqrt() * pnorm) / fnorm;
                let prered = temp1 * temp1 + (temp2 * temp2) / p5;
                let dirder = -(temp1 * temp1 + temp2 * temp2);

                let ratio = if prered != zero { actred / prered } else { zero };

                // Adjust the trust-region radius and damping by the gain ratio.
                if ratio <= p25 {
                    let mut temp = if actred >= zero {
                        p5
                    } else {
                        let denom = dirder + p5 * actred;
                        if denom != zero {
                            p5 * dirder / denom
                        } else {
                            p5
                        }
                    };
                    if p1 * fnorm1 >= fnorm || temp < p1 {
                        temp = p1;
                    }
                    delta = temp * min(delta, pnorm / p1);
                    par /= temp;
                } else if par == zero || ratio >= p75 {
                    delta = pnorm / p5;
                    par = p5 * par;
                }

                // Accept the step when the ratio is high enough.
                let accepted = ratio >= p0001;
                if accepted {
                    x = x_new;
                    residuals = residuals_new;
                    fnorm = fnorm1;
                    xnorm = enorm(&core::array::from_fn::<_, N, _>(|j| diag[j] * x[j]));
                }

                // Convergence tests.
                if actred.abs() <= self.ftol && prered <= self.ftol && p5 * ratio <= one {
                    return Ok(report(x, fnorm, evaluations, TerminationReason::Ftol));
                }
                if delta <= self.xtol * xnorm {
                    return Ok(report(x, fnorm, evaluations, TerminationReason::Xtol));
                }

                if accepted {
                    break;
                }
            }
        }

        Err(CalcError::DidNotConverge)
    }
}
