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
    /// A variable index was `>=` the number of variables in the point.
    IndexOutOfRange,
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

/// Errors from the kinematics module (plant geometry and kinematic maps).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum KinematicsError {
    /// A geometric parameter (wheel radius, track width) was not strictly positive.
    NonPositiveParameter,
    /// A geometric parameter was infinite or NaN.
    NonFinite,
}

/// Errors from the estimation module (Kalman filtering).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum EstimationError {
    /// The innovation covariance was not positive definite — the gain solve failed.
    NotPositiveDefinite,
    /// A state, covariance, or measurement value was infinite or NaN.
    NonFinite,
    /// A Jacobian step inside the filter failed.
    Diff(DiffError),
    /// Every particle weight underflowed to zero — the measurement is incompatible with the whole
    /// cloud.
    WeightsDegenerate,
}

/// Errors from the control module (feedback controllers, filters, path-following laws).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ControlError {
    /// A gain, timestep, limit, or filter coefficient was infinite or NaN.
    NonFinite,
    /// The control timestep `dt` was not strictly positive.
    NonPositiveTimestep,
    /// Output saturation limits were given with minimum greater than maximum.
    InvalidOutputLimits,
    /// A low-pass smoothing coefficient was outside the closed interval [0, 1].
    FilterCoefficientOutOfRange,
    /// The pure-pursuit lookahead distance was not strictly positive.
    NonPositiveLookaheadDistance,
    /// The gap-follower was instantiated with fewer than two beams.
    InvalidBeamCount,
    /// A field of view or frontal half-angle was outside its valid range.
    InvalidFieldOfView,
    /// A maximum range or gap threshold was not strictly positive, or the threshold exceeded the range.
    NonPositiveRange,
    /// A chassis width was not strictly positive, or half of it reached the maximum range.
    NonPositiveChassisWidth,
    /// A cruise speed or turn gain was not strictly positive.
    NonPositiveSpeed,
    /// A stopping distance was negative or not strictly less than the clear distance.
    InvalidSpeedScaling,
    /// A goal bias was negative.
    NegativeGoalBias,
}

/// Errors from the motion module (waypoint paths and their geometric queries).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MotionError {
    /// A waypoint coordinate was infinite or NaN.
    NonFinite,
    /// More waypoints were supplied than the path capacity allows.
    CapacityExceeded,
    /// A query required more waypoints than the path contains.
    PathTooShort,
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
    /// A kinematics error.
    Kinematics(KinematicsError),
    /// An estimation error.
    Estimation(EstimationError),
    /// A control error.
    Control(ControlError),
    /// A motion error.
    Motion(MotionError),
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
impl From<DiffError> for EstimationError {
    fn from(e: DiffError) -> Self {
        EstimationError::Diff(e)
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
impl From<KinematicsError> for CalcError {
    fn from(e: KinematicsError) -> Self {
        CalcError::Kinematics(e)
    }
}
impl From<EstimationError> for CalcError {
    fn from(e: EstimationError) -> Self {
        CalcError::Estimation(e)
    }
}
impl From<ControlError> for CalcError {
    fn from(e: ControlError) -> Self {
        CalcError::Control(e)
    }
}
impl From<MotionError> for CalcError {
    fn from(e: MotionError) -> Self {
        CalcError::Motion(e)
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
            IntegrateError::NonFinite => {
                f.write_str("integrand or state contained a non-finite value")
            }
            IntegrateError::IndexOutOfRange => f.write_str("variable index out of range"),
        }
    }
}

impl core::fmt::Display for SolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SolveError::DidNotConverge { iters, residual } => {
                write!(
                    f,
                    "solver did not converge after {iters} iterations (residual {residual})"
                )
            }
            SolveError::NonFinite => {
                f.write_str("residual or Jacobian contained a non-finite value")
            }
            SolveError::InvalidBracket => {
                f.write_str("bracket endpoints must enclose a sign change")
            }
            SolveError::Linalg(e) => write!(f, "{e}"),
            SolveError::Diff(e) => write!(f, "{e}"),
        }
    }
}

impl core::fmt::Display for KinematicsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            KinematicsError::NonPositiveParameter => {
                "geometric parameter must be strictly positive"
            }
            KinematicsError::NonFinite => "geometric parameter was not finite",
        })
    }
}

impl core::fmt::Display for EstimationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EstimationError::NotPositiveDefinite => {
                f.write_str("innovation covariance is not positive definite")
            }
            EstimationError::NonFinite => f.write_str("filter value was not finite"),
            EstimationError::Diff(e) => write!(f, "{e}"),
            EstimationError::WeightsDegenerate => f.write_str("all particle weights were zero"),
        }
    }
}

impl core::fmt::Display for ControlError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            ControlError::NonFinite => {
                "gain, timestep, limit, or filter coefficient was not finite"
            }
            ControlError::NonPositiveTimestep => "control timestep must be strictly positive",
            ControlError::InvalidOutputLimits => "output minimum must not exceed output maximum",
            ControlError::FilterCoefficientOutOfRange => {
                "low-pass smoothing coefficient must lie in [0, 1]"
            }
            ControlError::NonPositiveLookaheadDistance => {
                "lookahead distance must be strictly positive"
            }
            ControlError::InvalidBeamCount => "gap-follower needs at least two beams",
            ControlError::InvalidFieldOfView => {
                "field of view must lie in (0, 2π] and the frontal half-angle within half of it"
            }
            ControlError::NonPositiveRange => {
                "maximum range and gap threshold must be strictly positive, with the threshold no larger than the range"
            }
            ControlError::NonPositiveChassisWidth => {
                "chassis width must be strictly positive and less than twice the maximum range"
            }
            ControlError::NonPositiveSpeed => {
                "cruise speed and turn gain must be strictly positive"
            }
            ControlError::InvalidSpeedScaling => {
                "stopping distance must be non-negative and strictly less than the clear distance"
            }
            ControlError::NegativeGoalBias => "goal bias must not be negative",
        })
    }
}

impl core::fmt::Display for MotionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            MotionError::NonFinite => "waypoint coordinate was not finite",
            MotionError::CapacityExceeded => "more waypoints than the path capacity allows",
            MotionError::PathTooShort => "query required more waypoints than the path contains",
        })
    }
}

impl core::fmt::Display for CalcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CalcError::Linalg(e) => write!(f, "{e}"),
            CalcError::Solve(e) => write!(f, "{e}"),
            CalcError::Integrate(e) => write!(f, "{e}"),
            CalcError::Differentiate(e) => write!(f, "{e}"),
            CalcError::Kinematics(e) => write!(f, "{e}"),
            CalcError::Estimation(e) => write!(f, "{e}"),
            CalcError::Control(e) => write!(f, "{e}"),
            CalcError::Motion(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for LinalgError {}
impl core::error::Error for DiffError {}
impl core::error::Error for IntegrateError {}
impl core::error::Error for KinematicsError {}
impl core::error::Error for ControlError {}
impl core::error::Error for MotionError {}

impl core::error::Error for EstimationError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            EstimationError::Diff(e) => Some(e),
            _ => None,
        }
    }
}

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
            CalcError::Kinematics(e) => Some(e),
            CalcError::Estimation(e) => Some(e),
            CalcError::Control(e) => Some(e),
            CalcError::Motion(e) => Some(e),
        }
    }
}
