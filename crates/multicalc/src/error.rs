//! Error types for the crate. Each module family has its own enum; [`CalcError`] is the umbrella
//! they all convert into.

/// Errors from the linear-algebra module (factorizations, solves, inverses).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LinalgError {
    /// A matrix was singular or rank-deficient where a solve required full rank.
    Singular,
    /// A matrix was not positive definite.
    NotPositiveDefinite,
    /// A least-squares system had fewer rows than columns (`M < N`).
    Underdetermined,
    /// A matrix entry was infinite or NaN.
    NonFinite,
}

/// Errors from the differentiation modules (finite differences, autodiff, Jacobian, Hessian,
/// Taylor approximation, and the curl/divergence operators).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DiffError {
    /// The requested derivative order was zero.
    OrderZero,
    /// The requested derivative order is beyond what this differentiator supports.
    OrderUnsupported,
    /// A finite-difference step size of zero was supplied.
    StepSizeZero,
    /// A variable index was outside the bounds of the point array.
    IndexOutOfRange,
    /// An empty set of functions was supplied where at least one was required.
    EmptyFunctionSet,
}

/// Errors from the integration modules (Gaussian quadrature, iterative integration, ODE solvers,
/// and the line/flux integrals).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum IntegrateError {
    /// The number of integration iterations was zero.
    IterationsZero,
    /// A lower limit was not strictly less than its upper limit.
    LimitsIllDefined,
    /// The requested Gaussian quadrature order is outside the supported range.
    QuadratureOrderOutOfRange,
    /// The adaptive step size fell below the configured minimum.
    StepSizeTooSmall,
    /// The integrator ran out of its step budget before reaching the target.
    DidNotConverge {
        /// Steps taken before the budget was exhausted.
        steps: usize,
    },
    /// An integrand or state value was infinite or NaN.
    NonFinite,
}

/// Errors from the solver modules (root finding, Gauss-Newton, Levenberg-Marquardt).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SolveError {
    /// The solver ran out of its iteration/evaluation budget before converging.
    DidNotConverge {
        /// Iterations (or evaluations) spent before giving up.
        iters: usize,
        /// Residual norm at the final iterate.
        residual: f64,
    },
    /// A residual or Jacobian value was infinite or NaN.
    NonFinite,
    /// The bracket endpoints did not enclose a sign change.
    InvalidBracket,
    /// A linear-algebra step inside the solver failed.
    Linalg(LinalgError),
    /// A derivative or Jacobian step inside the solver failed.
    Diff(DiffError),
}

/// Umbrella over the per-module-family errors. Fallible operations return their family enum; this
/// type collects them where one error type must span families.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CalcError {
    /// A linear-algebra error.
    Linalg(LinalgError),
    /// A solver error.
    Solve(SolveError),
    /// An integration error.
    Integrate(IntegrateError),
    /// A differentiation error.
    Differentiate(DiffError),
}

impl From<LinalgError> for SolveError {
    fn from(e: LinalgError) -> Self {
        SolveError::Linalg(e)
    }
}
impl From<DiffError> for SolveError {
    fn from(e: DiffError) -> Self {
        SolveError::Diff(e)
    }
}
impl From<LinalgError> for CalcError {
    fn from(e: LinalgError) -> Self {
        CalcError::Linalg(e)
    }
}
impl From<DiffError> for CalcError {
    fn from(e: DiffError) -> Self {
        CalcError::Differentiate(e)
    }
}
impl From<IntegrateError> for CalcError {
    fn from(e: IntegrateError) -> Self {
        CalcError::Integrate(e)
    }
}
impl From<SolveError> for CalcError {
    fn from(e: SolveError) -> Self {
        CalcError::Solve(e)
    }
}

impl core::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            LinalgError::Singular => "matrix is singular or rank-deficient",
            LinalgError::NotPositiveDefinite => "matrix is not positive definite",
            LinalgError::Underdetermined => "system is underdetermined (M < N)",
            LinalgError::NonFinite => "matrix contained a non-finite value",
        })
    }
}

impl core::fmt::Display for DiffError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            DiffError::OrderZero => "derivative order cannot be zero",
            DiffError::OrderUnsupported => "derivative order is not supported",
            DiffError::StepSizeZero => "step size cannot be zero",
            DiffError::IndexOutOfRange => "variable index out of range",
            DiffError::EmptyFunctionSet => "function set cannot be empty",
        })
    }
}

impl core::fmt::Display for IntegrateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IntegrateError::IterationsZero => f.write_str("number of iterations cannot be zero"),
            IntegrateError::LimitsIllDefined => {
                f.write_str("lower limit must be strictly less than upper limit")
            }
            IntegrateError::QuadratureOrderOutOfRange => {
                f.write_str("quadrature order is out of supported range")
            }
            IntegrateError::StepSizeTooSmall => {
                f.write_str("adaptive step size fell below the minimum")
            }
            IntegrateError::DidNotConverge { steps } => {
                write!(f, "integrator did not converge within {steps} steps")
            }
            IntegrateError::NonFinite => f.write_str("integrand or state contained a non-finite value"),
        }
    }
}

impl core::fmt::Display for SolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SolveError::DidNotConverge { iters, residual } => {
                write!(f, "solver did not converge after {iters} iterations (residual {residual})")
            }
            SolveError::NonFinite => f.write_str("residual or Jacobian contained a non-finite value"),
            SolveError::InvalidBracket => f.write_str("bracket endpoints must enclose a sign change"),
            SolveError::Linalg(e) => write!(f, "{e}"),
            SolveError::Diff(e) => write!(f, "{e}"),
        }
    }
}

impl core::fmt::Display for CalcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CalcError::Linalg(e) => write!(f, "{e}"),
            CalcError::Solve(e) => write!(f, "{e}"),
            CalcError::Integrate(e) => write!(f, "{e}"),
            CalcError::Differentiate(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for LinalgError {}
impl core::error::Error for DiffError {}
impl core::error::Error for IntegrateError {}

impl core::error::Error for SolveError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            SolveError::Linalg(e) => Some(e),
            SolveError::Diff(e) => Some(e),
            _ => None,
        }
    }
}

impl core::error::Error for CalcError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            CalcError::Linalg(e) => Some(e),
            CalcError::Solve(e) => Some(e),
            CalcError::Integrate(e) => Some(e),
            CalcError::Differentiate(e) => Some(e),
        }
    }
}
