/// Errors returned by the fallible operations in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CalcError {
    /// The requested derivative order was zero.
    DerivativeOrderZero,
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
}

impl core::fmt::Display for CalcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            CalcError::DerivativeOrderZero => "derivative order cannot be zero",
            CalcError::StepSizeZero => "step size cannot be zero",
            CalcError::IndexOutOfRange => "variable index out of range",
            CalcError::IterationsZero => "number of iterations cannot be zero",
            CalcError::IntegrationLimitsIllDefined => {
                "lower limit must be strictly less than upper limit"
            }
            CalcError::QuadratureOrderOutOfRange => "quadrature order is out of supported range",
            CalcError::EmptyFunctionSet => "function set cannot be empty",
        })
    }
}

impl core::error::Error for CalcError {}
