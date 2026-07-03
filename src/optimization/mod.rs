//! Nonlinear least-squares optimization.
//!
//! [`LevenbergMarquardt`] and its undamped sibling [`GaussNewton`] minimize the sum of squared
//! residuals of a `VectorFn`, reporting the outcome as a [`MinimizationReport`] whose
//! [`TerminationReason`] says which test converged. The Jacobian is taken by exact autodiff by
//! default, or by finite differences if a finite-difference derivator is supplied. Failures
//! (non-finite values, an underdetermined system, no convergence within the budget) come back as a
//! [`CalcError`](crate::utils::error_codes::CalcError).
//!
//! The Levenberg-Marquardt driver ports MINPACK's `lmder`/`lmpar` (Moré, Garbow, Hillstrom; public
//! domain, netlib), following Moré (1978), "The Levenberg-Marquardt algorithm: Implementation and
//! theory", and Nocedal & Wright, *Numerical Optimization*, chapters 4 and 10. This is a clean-room, 
//! fixed-size `no_std` reimplementation on this crate's own [`Vector`](crate::linear_algebra::Vector) 
//! and [`Matrix`](crate::linear_algebra::Matrix) types.

pub mod gauss_newton;
pub mod levenberg_marquardt;
pub(crate) mod trust_region;

pub use gauss_newton::GaussNewton;
pub use levenberg_marquardt::LevenbergMarquardt;

use crate::scalar::Numeric;

/// Which convergence test stopped a solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TerminationReason {
    /// The sum of squared residuals stopped decreasing (relative reduction within `ftol`).
    Ftol,
    /// The parameter step became negligible (relative step within `xtol`).
    Xtol,
    /// The scaled gradient became negligible (within `gtol`): the residual is orthogonal to the
    /// Jacobian columns, so a stationary point was reached.
    Gtol,
}

/// The outcome of a solver run.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct MinimizationReport<const N: usize, T = f64> {
    /// The parameter values at the final point.
    pub solution: [T; N],
    /// The final objective value, half the sum of squared residuals.
    pub objective_function: T,
    /// How many times the residual function was evaluated.
    pub evaluations: usize,
    /// Why the solver stopped.
    pub termination: TerminationReason,
}

/// Whether every element of `v` is finite.
pub(crate) fn is_finite<const K: usize, T: Numeric>(v: &[T; K]) -> bool {
    v.iter().all(|value| value.is_finite())
}

/// Assembles a report at the final point (objective is half the sum of squared residuals).
pub(crate) fn report<const N: usize, T: Numeric>(
    solution: [T; N],
    residual_norm: T,
    evaluations: usize,
    termination: TerminationReason,
) -> MinimizationReport<N, T> {
    MinimizationReport {
        solution,
        objective_function: T::HALF * residual_norm * residual_norm,
        evaluations,
        termination,
    }
}

#[cfg(test)]
mod test;
