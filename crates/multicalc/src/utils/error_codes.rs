/// Errors returned by the fallible operations in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CalcError {
    /// The requested derivative order was zero.
    DerivativeOrderZero,
    /// The requested derivative order is beyond what this differentiator supports.
    DerivativeOrderUnsupported,
    /// A finite-difference step size of zero was supplied.
    StepSizeZero,
    /// A variable index was outside the bounds of the point array.
    IndexOutOfRange,
    /// The number of integration iterations was zero.
    IterationsZero,
    /// A lower integration limit was not strictly less than its upper limit.
    IntegrationLimitsIllDefined,
    /// The requested Gaussian quadrature order is outside the supported range.
    QuadratureOrderOutOfRange,
    /// An empty set of functions was supplied where at least one was required.
    EmptyFunctionSet,
    /// A least-squares system had fewer rows than columns (`M < N`).
    Underdetermined,
    /// A matrix was singular or rank-deficient where a solve required full rank.
    SingularMatrix,
    /// A matrix was not positive definite.
    NotPositiveDefinite,
    /// A solver ran out of its iteration or evaluation budget before converging.
    DidNotConverge,
    /// A residual or Jacobian value was infinite or NaN.
    NonFiniteValue,
    /// The bracket endpoints did not enclose a sign change.
    InvalidBracket,
}

impl core::fmt::Display for CalcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            CalcError::DerivativeOrderZero => "derivative order cannot be zero",
            CalcError::DerivativeOrderUnsupported => "derivative order is not supported",
            CalcError::StepSizeZero => "step size cannot be zero",
            CalcError::IndexOutOfRange => "variable index out of range",
            CalcError::IterationsZero => "number of iterations cannot be zero",
            CalcError::IntegrationLimitsIllDefined => {
                "lower limit must be strictly less than upper limit"
            }
            CalcError::QuadratureOrderOutOfRange => "quadrature order is out of supported range",
            CalcError::EmptyFunctionSet => "function set cannot be empty",
            CalcError::Underdetermined => "system is underdetermined (M < N)",
            CalcError::SingularMatrix => "matrix is singular or rank-deficient",
            CalcError::NotPositiveDefinite => "matrix is not positive definite",
            CalcError::DidNotConverge => {
                "solver did not converge within the iteration/evaluation budget"
            }
            CalcError::NonFiniteValue => "residual or Jacobian contained a non-finite value",
            CalcError::InvalidBracket => "bracket endpoints must enclose a sign change",
        })
    }
}

impl core::error::Error for CalcError {}
