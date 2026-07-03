//! Types shared by the nonlinear least-squares solvers: the [`MinimizationReport`] a solver
//! returns and the [`TerminationReason`] that explains why it stopped.

/// Why a solver stopped.
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
    /// The iteration or evaluation budget was exhausted before convergence.
    MaxIterations,
    /// A residual or Jacobian value was infinite or NaN.
    NonFinite,
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
