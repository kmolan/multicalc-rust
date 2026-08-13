//! Error types for the crate. Each module family has its own enum; [`CalcError`] is the umbrella
//! they all convert into.

/// Errors from linear algebra and matrix-based discretization.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LinalgError {
    /// A matrix was singular or rank-deficient where a solve required full rank.
    Singular,
    /// A matrix had a pivot too small relative to its scale for a reliable solve.
    IllConditioned,
    /// A matrix was not positive definite.
    NotPositiveDefinite,
    /// A least-squares system had fewer rows than columns (`M < N`).
    Underdetermined,
    /// A matrix entry was infinite or NaN.
    NonFinite,
    /// A matrix was not the same read across the diagonal.
    NotSymmetric,
    /// A discretization timestep was negative, infinite, or NaN.
    InvalidTimestep,
    /// A matrix-equation solver ran out of its iteration budget before settling.
    DidNotConverge {
        /// Iterations spent before giving up.
        iters: usize,
    },
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

/// Errors from the kinematics module (plant geometry, kinematic maps, and jointed models).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum KinematicsError {
    /// A geometric parameter (wheel radius, track width) was not strictly positive.
    NonPositiveParameter,
    /// A geometric, model, or configuration value was infinite or NaN.
    NonFinite,
    /// More joints were supplied than the tree can hold.
    CapacityExceeded,
    /// The list of joints and the list of what they hang off are different lengths.
    JointCountMismatch,
    /// A joint hangs off something that is not an earlier joint in the tree.
    ParentOutOfOrder,
    /// A joint's axis is all zeros, which names no direction.
    AxisHasNoDirection,
    /// A joint's lower limit is above its upper limit.
    LimitsReversed,
    /// A tool frame was named that the tree does not have.
    ToolIndexOutOfRange,
    /// The solved poses came from a tree with a different joint count.
    StateShapeMismatch,
    /// A joint weight was zero or negative, so it names no cost of moving.
    NonPositiveWeight,
    /// A matrix step inside the solver failed.
    Linalg(LinalgError),
}

/// Errors from the spatial module (rigid-body inertia).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SpatialError {
    /// A body mass was zero or negative.
    NonPositiveMass,
    /// A mass, centre-of-mass, or inertia value was infinite or NaN.
    NonFinite,
    /// A rotational inertia was not the same read across the diagonal.
    NotSymmetric,
    /// A rotational inertia had a diagonal entry that was zero or negative.
    NonPositiveInertia,
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
    /// A filter tuning value did not describe a usable spread of sigma points.
    InvalidTuning,
    /// A measurement model was asked to read a state component that does not exist.
    StateIndexOutOfRange,
}

/// Errors from the signal-processing module (filters, smoothers, and signal conditioning).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SignalError {
    /// A frequency, gain, threshold, timestep, or coefficient was infinite or NaN.
    NonFinite,
    /// The sampling timestep was not strictly positive.
    NonPositiveTimestep,
    /// A smoothing coefficient was outside the closed interval [0, 1].
    CoefficientOutOfRange,
    /// A filter frequency was not strictly positive, or reached half the sampling rate.
    FrequencyOutOfRange,
    /// A quality factor was not strictly positive.
    NonPositiveQualityFactor,
    /// A deadband threshold was negative.
    NegativeThreshold,
    /// Switching thresholds were given with the lower one at or above the upper one.
    ThresholdsOutOfOrder,
    /// A rate limit was not strictly positive.
    NonPositiveRate,
    /// A window length of zero was requested.
    WindowTooShort,
    /// An even window length was requested where the middle sample has to be well defined.
    WindowEvenLength,
    /// More polynomial terms were requested than the window has samples to fit them.
    PolynomialOrderTooHigh,
    /// A cascade section index was past the end of the cascade.
    SectionIndexOutOfRange,
    /// A linear-algebra step inside a filter's setup failed.
    Linalg(LinalgError),
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
    #[deprecated(
        since = "0.10.0",
        note = "filters now report SignalError::CoefficientOutOfRange"
    )]
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
    /// A controller gain was zero or negative.
    NonPositiveGain,
    /// A rotational inertia did not read the same across the diagonal.
    NotSymmetricInertia,
    /// A rotational inertia was not positive definite.
    NonPositiveInertia,
    /// A wanted acceleration cancelled gravity exactly, leaving no direction to push in.
    UndefinedThrustDirection,
    /// The push would be straight along the wanted heading, so the heading cannot be set.
    UndefinedHeadingDirection,
    /// A matrix step inside a controller's setup failed.
    Linalg(LinalgError),
    /// A filter setup error.
    Signal(SignalError),
}

/// Errors from the dynamics module (how a rigid body moves under the forces on it).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DynamicsError {
    /// A mass, inertia, or gravity value was infinite or NaN.
    NonFinite,
    /// A rotational inertia was not positive definite.
    NonPositiveInertia,
    /// The rotational inertia could not be inverted.
    Linalg(LinalgError),
}

/// Errors from the plant module (actuator models and sharing a command out across actuators).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PlantError {
    /// A rotor position, ratio, thrust limit, lag time, or tick length was infinite or NaN.
    NonFinite,
    /// An arm length was zero or negative.
    NonPositiveArmLength,
    /// A rotor's turning force per unit of push was zero or negative.
    NonPositiveTorqueRatio,
    /// Thrust limits were given with the smallest at or above the largest.
    InvalidThrustLimits,
    /// The rotors are placed so that some wanted turn cannot be produced at all.
    RotorLayoutNotIndependent,
    /// A rotor's lag time was zero or negative.
    NonPositiveTimeConstant,
    /// A tick length was zero or negative.
    NonPositiveTimestep,
    /// The rotor layout could not be turned into a set of per-rotor commands.
    Linalg(LinalgError),
}

/// Errors from the mapping module (occupancy grids, ray casting, and scan geometry).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MappingError {
    /// A cell size, origin coordinate, angle, or range was infinite or NaN.
    NonFinite,
    /// A cell size was zero or negative.
    NonPositiveResolution,
    /// A grid was asked for with no columns or no rows.
    EmptyGrid,
    /// A grid's columns and rows multiply out to more cells than can be counted.
    GridTooLarge,
    /// A scan was asked for with fewer than two beams.
    TooFewBeams,
    /// A field of view was outside the range one turn can cover.
    InvalidFieldOfView,
    /// A sensing range was zero or negative.
    NonPositiveRange,
    /// A closest-visible range was negative, or reached the range the scan can see to.
    InvalidRangeLimits,
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
    /// There is not exactly one duration for each pair of waypoints.
    SegmentCountMismatch,
    /// A segment duration was zero or negative.
    DurationNotPositive,
    /// The planner holds fewer free derivatives than this many segments needs.
    WorkspaceTooSmall,
    /// The trajectory's linear system could not be factorized.
    Linalg(LinalgError),
    /// A polynomial the trajectory is built from could not be formed.
    Polynomial(PolynomialError),
}

/// Errors from polynomial construction, evaluation, and root finding.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PolynomialError {
    /// A coefficient, node, or sample was infinite or NaN.
    NonFinite,
    /// The highest coefficient is zero, so the polynomial is not of the degree asked for.
    LeadingCoefficientZero,
    /// The result needs more coefficients than the polynomial has room for.
    DegreeOverflow,
    /// More terms than the polynomial can hold.
    CapacityExceeded,
    /// There is nothing to evaluate.
    Empty,
    /// A variable index past the number of variables the polynomial has.
    VariableOutOfRange,
    /// Two interpolation points share the same position.
    DuplicateNode,
    /// Fewer samples were given than the number of coefficients to fit.
    TooFewSamples,
    /// A piece covers zero or a negative amount of the parameter.
    SpanNotPositive,
    /// Root isolation ran out of steps before separating every root.
    DidNotConverge {
        /// How many halving steps were taken.
        steps: usize,
    },
    /// A fit or an endpoint solve could not be factorized.
    Linalg(LinalgError),
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
    /// A spatial error.
    Spatial(SpatialError),
    /// An estimation error.
    Estimation(EstimationError),
    /// A signal-processing error.
    Signal(SignalError),
    /// A control error.
    Control(ControlError),
    /// A dynamics error.
    Dynamics(DynamicsError),
    /// A plant error.
    Plant(PlantError),
    /// A mapping error.
    Mapping(MappingError),
    /// A motion error.
    Motion(MotionError),
    /// A polynomial error.
    Polynomial(PolynomialError),
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
impl From<LinalgError> for KinematicsError {
    fn from(e: LinalgError) -> Self {
        KinematicsError::Linalg(e)
    }
}
impl From<DiffError> for EstimationError {
    fn from(e: DiffError) -> Self {
        EstimationError::Diff(e)
    }
}
impl From<LinalgError> for SignalError {
    fn from(e: LinalgError) -> Self {
        SignalError::Linalg(e)
    }
}
impl From<LinalgError> for PolynomialError {
    fn from(e: LinalgError) -> Self {
        PolynomialError::Linalg(e)
    }
}
impl From<LinalgError> for MotionError {
    fn from(e: LinalgError) -> Self {
        MotionError::Linalg(e)
    }
}
impl From<PolynomialError> for MotionError {
    fn from(e: PolynomialError) -> Self {
        MotionError::Polynomial(e)
    }
}
impl From<SignalError> for ControlError {
    fn from(e: SignalError) -> Self {
        ControlError::Signal(e)
    }
}
impl From<LinalgError> for ControlError {
    fn from(e: LinalgError) -> Self {
        ControlError::Linalg(e)
    }
}
impl From<LinalgError> for DynamicsError {
    fn from(e: LinalgError) -> Self {
        DynamicsError::Linalg(e)
    }
}
impl From<LinalgError> for PlantError {
    fn from(e: LinalgError) -> Self {
        PlantError::Linalg(e)
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
impl From<SpatialError> for CalcError {
    fn from(e: SpatialError) -> Self {
        CalcError::Spatial(e)
    }
}
impl From<EstimationError> for CalcError {
    fn from(e: EstimationError) -> Self {
        CalcError::Estimation(e)
    }
}
impl From<SignalError> for CalcError {
    fn from(e: SignalError) -> Self {
        CalcError::Signal(e)
    }
}
impl From<ControlError> for CalcError {
    fn from(e: ControlError) -> Self {
        CalcError::Control(e)
    }
}
impl From<DynamicsError> for CalcError {
    fn from(e: DynamicsError) -> Self {
        CalcError::Dynamics(e)
    }
}
impl From<PlantError> for CalcError {
    fn from(e: PlantError) -> Self {
        CalcError::Plant(e)
    }
}
impl From<MappingError> for CalcError {
    fn from(e: MappingError) -> Self {
        CalcError::Mapping(e)
    }
}
impl From<MotionError> for CalcError {
    fn from(e: MotionError) -> Self {
        CalcError::Motion(e)
    }
}
impl From<PolynomialError> for CalcError {
    fn from(e: PolynomialError) -> Self {
        CalcError::Polynomial(e)
    }
}

impl core::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LinalgError::Singular => f.write_str("matrix is singular or rank-deficient"),
            LinalgError::IllConditioned => {
                f.write_str("matrix is too ill-conditioned to solve reliably")
            }
            LinalgError::NotPositiveDefinite => f.write_str("matrix is not positive definite"),
            LinalgError::Underdetermined => f.write_str("system is underdetermined (M < N)"),
            LinalgError::NonFinite => f.write_str("matrix contained a non-finite value"),
            LinalgError::NotSymmetric => {
                f.write_str("matrix must read the same across the diagonal")
            }
            LinalgError::InvalidTimestep => f.write_str("timestep must be finite and non-negative"),
            LinalgError::DidNotConverge { iters } => {
                write!(f, "matrix equation did not settle after {iters} iterations")
            }
        }
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
            SolveError::DidNotConverge { iters } => {
                write!(f, "solver did not converge after {iters} iterations")
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
        let text = match self {
            KinematicsError::NonPositiveParameter => {
                "geometric parameter must be strictly positive"
            }
            KinematicsError::NonFinite => "geometric, model, or configuration value was not finite",
            KinematicsError::CapacityExceeded => "more joints than the tree can hold",
            KinematicsError::JointCountMismatch => {
                "the joint list and the parent list are different lengths"
            }
            KinematicsError::ParentOutOfOrder => {
                "a joint hangs off something that is not an earlier joint"
            }
            KinematicsError::AxisHasNoDirection => {
                "joint axis was all zeros, which names no direction"
            }
            KinematicsError::LimitsReversed => "a joint's lower limit is above its upper limit",
            KinematicsError::ToolIndexOutOfRange => {
                "a tool frame was named that the tree does not have"
            }
            KinematicsError::StateShapeMismatch => {
                "solved poses came from a tree with a different joint count"
            }
            KinematicsError::NonPositiveWeight => "a joint weight was zero or negative",
            KinematicsError::Linalg(e) => return write!(f, "{e}"),
        };
        f.write_str(text)
    }
}

impl core::fmt::Display for SpatialError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            SpatialError::NonPositiveMass => "body mass must be strictly positive",
            SpatialError::NonFinite => "mass, centre of mass, or inertia was not finite",
            SpatialError::NotSymmetric => {
                "rotational inertia must read the same across the diagonal"
            }
            SpatialError::NonPositiveInertia => {
                "rotational inertia diagonal entries must be strictly positive"
            }
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
            EstimationError::InvalidTuning => f.write_str("invalid filter tuning"),
            EstimationError::StateIndexOutOfRange => {
                f.write_str("measurement model refers to a state component that does not exist")
            }
        }
    }
}

impl core::fmt::Display for SignalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SignalError::NonFinite => f.write_str("filter parameter contained a non-finite value"),
            SignalError::NonPositiveTimestep => f.write_str("timestep must be strictly positive"),
            SignalError::CoefficientOutOfRange => {
                f.write_str("smoothing coefficient must lie in [0, 1]")
            }
            SignalError::FrequencyOutOfRange => {
                f.write_str("frequency must be above zero and below half the sampling rate")
            }
            SignalError::NonPositiveQualityFactor => {
                f.write_str("quality factor must be strictly positive")
            }
            SignalError::NegativeThreshold => f.write_str("deadband threshold cannot be negative"),
            SignalError::ThresholdsOutOfOrder => {
                f.write_str("lower switching threshold must be below the upper one")
            }
            SignalError::NonPositiveRate => f.write_str("rate limit must be strictly positive"),
            SignalError::WindowTooShort => f.write_str("window length cannot be zero"),
            SignalError::WindowEvenLength => f.write_str("window length must be odd"),
            SignalError::PolynomialOrderTooHigh => {
                f.write_str("window is too short for the number of polynomial terms")
            }
            SignalError::SectionIndexOutOfRange => {
                f.write_str("cascade section index out of range")
            }
            SignalError::Linalg(e) => write!(f, "filter setup failed: {e}"),
        }
    }
}

impl core::fmt::Display for ControlError {
    // The deprecated variant still needs a message for as long as it is part of the enum.
    #[allow(deprecated)]
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
            ControlError::NonPositiveGain => "controller gain must be strictly positive",
            ControlError::NotSymmetricInertia => {
                "rotational inertia must read the same across the diagonal"
            }
            ControlError::NonPositiveInertia => "rotational inertia must be positive definite",
            ControlError::UndefinedThrustDirection => {
                "wanted acceleration cancels gravity, leaving no direction to push in"
            }
            ControlError::UndefinedHeadingDirection => {
                "the push is straight along the wanted heading, so the heading cannot be set"
            }
            ControlError::Linalg(e) => return write!(f, "{e}"),
            ControlError::Signal(e) => return write!(f, "{e}"),
        })
    }
}

impl core::fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DynamicsError::NonFinite => f.write_str("body property was not finite"),
            DynamicsError::NonPositiveInertia => {
                f.write_str("rotational inertia is not positive definite")
            }
            DynamicsError::Linalg(e) => write!(f, "rotational inertia could not be inverted: {e}"),
        }
    }
}

impl core::fmt::Display for PlantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlantError::NonFinite => f.write_str("rotor property was not finite"),
            PlantError::NonPositiveArmLength => f.write_str("arm length must be strictly positive"),
            PlantError::NonPositiveTorqueRatio => {
                f.write_str("turning force per unit of push must be strictly positive")
            }
            PlantError::InvalidThrustLimits => {
                f.write_str("smallest thrust must be below the largest")
            }
            PlantError::RotorLayoutNotIndependent => {
                f.write_str("rotor layout cannot produce every wanted push and turn")
            }
            PlantError::NonPositiveTimeConstant => {
                f.write_str("rotor lag time must be strictly positive")
            }
            PlantError::NonPositiveTimestep => f.write_str("tick length must be strictly positive"),
            PlantError::Linalg(e) => write!(f, "rotor layout could not be inverted: {e}"),
        }
    }
}

impl core::fmt::Display for MappingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MappingError::NonFinite => f.write_str("map value contained a non-finite value"),
            MappingError::NonPositiveResolution => {
                f.write_str("cell size must be strictly positive")
            }
            MappingError::EmptyGrid => {
                f.write_str("grid must have at least one column and one row")
            }
            MappingError::GridTooLarge => f.write_str("grid has more cells than can be counted"),
            MappingError::TooFewBeams => f.write_str("a scan needs at least two beams"),
            MappingError::InvalidFieldOfView => f.write_str("field of view must lie in (0, 2π]"),
            MappingError::NonPositiveRange => {
                f.write_str("sensing range must be strictly positive")
            }
            MappingError::InvalidRangeLimits => f.write_str(
                "closest visible range must be non-negative and below the sensing range",
            ),
        }
    }
}

impl core::fmt::Display for MotionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MotionError::NonFinite => f.write_str("waypoint coordinate was not finite"),
            MotionError::CapacityExceeded => {
                f.write_str("more waypoints than the path capacity allows")
            }
            MotionError::PathTooShort => {
                f.write_str("query required more waypoints than the path contains")
            }
            MotionError::SegmentCountMismatch => {
                f.write_str("one duration is needed for each pair of waypoints")
            }
            MotionError::DurationNotPositive => {
                f.write_str("segment duration was zero or negative")
            }
            MotionError::WorkspaceTooSmall => {
                f.write_str("more segments than the planner's free-derivative capacity holds")
            }
            MotionError::Linalg(e) => write!(f, "trajectory system could not be solved: {e}"),
            MotionError::Polynomial(e) => write!(f, "trajectory piece could not be formed: {e}"),
        }
    }
}

impl core::fmt::Display for PolynomialError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PolynomialError::NonFinite => f.write_str("polynomial value was not finite"),
            PolynomialError::LeadingCoefficientZero => {
                f.write_str("highest polynomial coefficient is zero")
            }
            PolynomialError::DegreeOverflow => {
                f.write_str("result needs more coefficients than the polynomial holds")
            }
            PolynomialError::CapacityExceeded => {
                f.write_str("more terms than the polynomial holds")
            }
            PolynomialError::Empty => f.write_str("there is nothing to evaluate"),
            PolynomialError::VariableOutOfRange => {
                f.write_str("variable index past the number of variables")
            }
            PolynomialError::DuplicateNode => {
                f.write_str("two interpolation points share the same position")
            }
            PolynomialError::TooFewSamples => f.write_str("fewer samples than coefficients to fit"),
            PolynomialError::SpanNotPositive => f.write_str("piece span was zero or negative"),
            PolynomialError::DidNotConverge { steps } => {
                write!(f, "root isolation stopped after {steps} steps")
            }
            PolynomialError::Linalg(e) => {
                write!(f, "polynomial system could not be solved: {e}")
            }
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
            CalcError::Kinematics(e) => write!(f, "{e}"),
            CalcError::Spatial(e) => write!(f, "{e}"),
            CalcError::Estimation(e) => write!(f, "{e}"),
            CalcError::Signal(e) => write!(f, "{e}"),
            CalcError::Control(e) => write!(f, "{e}"),
            CalcError::Dynamics(e) => write!(f, "{e}"),
            CalcError::Plant(e) => write!(f, "{e}"),
            CalcError::Mapping(e) => write!(f, "{e}"),
            CalcError::Motion(e) => write!(f, "{e}"),
            CalcError::Polynomial(e) => write!(f, "{e}"),
        }
    }
}

impl core::error::Error for LinalgError {}
impl core::error::Error for DiffError {}
impl core::error::Error for IntegrateError {}
impl core::error::Error for KinematicsError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            KinematicsError::Linalg(e) => Some(e),
            _ => None,
        }
    }
}
impl core::error::Error for SpatialError {}
impl core::error::Error for MappingError {}

impl core::error::Error for DynamicsError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            DynamicsError::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

impl core::error::Error for PlantError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            PlantError::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

impl core::error::Error for MotionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            MotionError::Linalg(e) => Some(e),
            MotionError::Polynomial(e) => Some(e),
            _ => None,
        }
    }
}

impl core::error::Error for PolynomialError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            PolynomialError::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

impl core::error::Error for SignalError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            SignalError::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

impl core::error::Error for ControlError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            ControlError::Signal(e) => Some(e),
            ControlError::Linalg(e) => Some(e),
            _ => None,
        }
    }
}

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
            CalcError::Spatial(e) => Some(e),
            CalcError::Estimation(e) => Some(e),
            CalcError::Signal(e) => Some(e),
            CalcError::Control(e) => Some(e),
            CalcError::Dynamics(e) => Some(e),
            CalcError::Plant(e) => Some(e),
            CalcError::Mapping(e) => Some(e),
            CalcError::Motion(e) => Some(e),
            CalcError::Polynomial(e) => Some(e),
        }
    }
}
