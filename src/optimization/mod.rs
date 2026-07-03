//! Nonlinear least-squares optimization.
//!
//! [`LevenbergMarquardt`] minimizes the sum of squared residuals of a `VectorFn`, reporting the
//! outcome as a [`MinimizationReport`] whose [`TerminationReason`] says which test converged.
//! Failures (non-finite values, an underdetermined system, no convergence within the budget) come
//! back as a [`CalcError`](crate::utils::error_codes::CalcError).

pub mod levenberg_marquardt;
pub(crate) mod trust_region;

pub use levenberg_marquardt::LevenbergMarquardt;

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

#[cfg(test)]
mod test;
