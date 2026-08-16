#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Each tutorial page is compiled as a doctest so its examples cannot go stale.
#[cfg(all(doctest, feature = "alloc"))]
mod tutorial_examples;

#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
extern crate alloc;

/// Re-export of [`libm`], so `no_std` users can reach transcendental functions
/// (`libm::sin`, `libm::exp`, …) without taking their own dependency.
pub use libm;

/// The scalar trait the calculus modules are generic over (implemented for `f32` and `f64`).
pub use scalar::Numeric;

/// Forward-mode dual number giving exact first derivatives (it implements [`Numeric`]).
pub use scalar::Dual;

/// Hyper-dual number giving exact first and second derivatives (it implements [`Numeric`]).
pub use scalar::HyperDual;

/// Jet (truncated Taylor series) giving exact derivatives to arbitrary order (it implements [`Numeric`]).
pub use scalar::Jet;

/// Scalar-function abstraction evaluable at any [`Numeric`] scalar, so one formula drives both
/// finite differences and autodiff.
pub use scalar::{ScalarFn, ScalarFnN, VectorFn};

/// The plain scalar wrapper, and the marker for numeric constants inside a `scalar_fn!` body.
pub use scalar::{Const, Primal, c};

/// Differentiation: the autodiff and finite-difference backends, their shared traits, and the
/// derivative-matrix types.
pub use numerical_derivative::{
    AutoDiffMulti, AutoDiffSingle, DerivatorMultiVariable, DerivatorSingleVariable,
    FiniteDifferenceConfig, FiniteDifferenceMode, FiniteDifferenceMulti, FiniteDifferenceSingle,
    Hessian, Jacobian,
};

/// One-call derivatives, needing no imported trait and no configured backend.
pub use numerical_derivative::{derivative, partial, second_derivative};

/// Integration: the iterative and Gaussian-quadrature backends and their shared traits.
pub use numerical_integration::{
    GaussianConfig, GaussianMulti, GaussianQuadratureMethod, GaussianSingle,
    IntegratorMultiVariable, IntegratorSingleVariable, IterativeConfig, IterativeMethod,
    IterativeMulti, IterativeSingle, SummationMethod,
};

/// One-call integration over an interval, on the same terms.
pub use numerical_integration::integral;

/// Linear and quadratic Taylor models with goodness-of-fit metrics.
pub use approximation::{
    LinearApproximation, LinearApproximationPredictionMetrics, LinearApproximator,
    QuadraticApproximation, QuadraticApproximationPredictionMetrics, QuadraticApproximator,
};

/// Fixed-size, stack-allocated vector and matrix types.
pub use linear_algebra::{Matrix, Vector};

/// Type aliases for ease of life
pub use linear_algebra::{Matrix2D, Matrix3D, Matrix4D, Matrix6D, Vector2D, Vector3D, Vector6D};

/// Solvers for the two matrix equations behind optimal linear feedback.
pub use linear_algebra::{solve_discrete_lyapunov, solve_discrete_riccati};

/// Zero-order-hold, Van Loan, and white-noise discretization of continuous-time linear systems.
pub use discretization::{q_discrete_white_noise, van_loan, zoh};

/// Fixed-step RK4, adaptive RK45 (Dormand–Prince), and an orientation integrator
pub use ode::{ExponentialMap, Rk4, Rk45};

/// Filters, smoothers, and signal conditioning.
pub use signal_processing::{
    Biquad, BiquadCascade, BiquadCoefficients, Deadband, Hysteresis, MovingAverage,
    MultiChannelBiquad, OnePoleLowPass, RunningMedian, SavitzkyGolay, SlewRateLimiter,
    harmonic_notch_coefficients,
};

/// Quaternion
pub use spatial::Quaternion;

/// SO(2)/SE(2)/SO(3)/SE(3) Lie groups for 2D and 3D rotations and rigid-body transforms.
pub use spatial::{SE2, SE3, SO2, SO3};

/// Typed spatial velocity and force in the linear-first `[v; ω]` / `[force; torque]` ordering.
pub use spatial::{Twist, Wrench};

/// A rigid body's mass, centre of mass, and rotational inertia.
pub use spatial::SpatialInertia;

/// The pose and velocity of a body free to move in all six directions.
pub use spatial::FreeJointState;

/// Differential-drive kinematics and SE(2) odometry.
pub use kinematics::{BodyArc, BodyTwist, DifferentialDrive, WheelRotations, WheelVelocities};

/// Robot models, forward and inverse kinematics, and geometric Jacobians.
pub use kinematics::{
    CollisionQuery, CollisionReport, CollisionSource, InverseKinematics, InverseKinematicsReport,
    InverseKinematicsTermination, JacobianFrame, Joint, JointKind, JointParent, KinematicJacobian,
    KinematicTree, KinematicTreeState, MultiStartInverseKinematics, MultiStartReport, Primitive,
    SecondaryObjective, SingularityKind,
};

/// Occupancry grid and scan geometry
pub use mapping::{MutableOccupancyMap, OccupancyMap, ScanGeometry};

/// Heap-based occupancy grid for large maps
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use mapping::DynamicOccupancyGrid;

/// Linear Kalman filter, Extended Kalman filter, and Unscented Kalman filter
pub use estimation::{
    CovarianceUpdate, ExtendedKalmanFilter, KalmanFilter, KalmanModel, UnscentedKalmanFilter,
};

/// Error-state filter for an IMU.
pub use estimation::{ErrorStateKalmanFilter, ImuNoise, NominalState, NominalStateFn};

/// Light attitude filters for a turn-rate sensor fused with an accelerometer and a magnetometer.
pub use estimation::{MadgwickFilter, MahonyFilter};

/// Ready-made filter models: a turning-arc process model, a sensor reading part of the state, and
/// an angle-aware residual.
pub use estimation::{ConstantTurnAndSpeed, DirectMeasurement, residual_with_wrapped_angles};

/// Particle filter (bootstrap/SIR) with pluggable resampling and measurement likelihood.
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use estimation::{GaussianLikelihood, Likelihood, ParticleFilter, ResamplingScheme};

/// Monte Carlo Localization using particle filter estimation.
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use estimation::{BeamModel, InitialParticleCloud, MonteCarloLocalizer};

/// Seedable pseudo-random generator and the trait its uniform and normal draws come from.
pub use random::{Pcg32, RandomSource};

/// The Levenberg-Marquardt and Gauss-Newton least-squares solvers and their result types.
pub use optimization::{GaussNewton, LevenbergMarquardt, MinimizationReport, TerminationReason};

/// Bracketed and Newton root finders for scalar equations and square systems.
pub use root_finding::{Bisection, Newton, NewtonSystem, RootReport, RootReportN, RootTermination};

/// Polynomials by their coefficients, in pieces, and in several variables.
pub use polynomial::{
    MultivariatePolynomial, MultivariateTerm, PiecewisePolynomial, Polynomial, RealRoots,
};

/// Feedback control: PID, optimal linear state feedback, attitude control on rotations, the
/// pure-pursuit path-following law, and Follow-the-Gap reactive avoidance.
pub use control::{
    Curvature, FollowTheGap, FollowTheGapOutput, GeometricAttitudeController, Lqr, Pid,
    ThrustCommand, pure_pursuit_curvature, thrust_command_from_acceleration,
};

/// Waypoint paths, planned trajectories, point-to-point motion profiles, and their arc-length,
/// closest-point and lookahead queries.
pub use motion::{
    BoundaryDerivatives, EndOfPath, MinimumSnapPlanner, MotionProfile, MotionProfilePlanner,
    PathProjection, PolylinePath, ProfileLimits, ProfileState, ProfileStrategy,
    SynchronizedProfile, SynchronizedState, durations_from_average_speed,
};

/// A single rigid body's motion under the forces on it, and the state an integrator carries.
pub use dynamics::{RigidBody, RigidBodyAcceleration};

/// Sharing a wanted push and turn out across a set of rotors, and how quickly a rotor catches up
/// to what it was asked for.
pub use plant::{MultirotorMixer, RotorCommands, RotorLag, RotorSpin};

/// Per-module-family error enums and the umbrella they convert into.
pub use error::{
    CalcError, ControlError, DiffError, DynamicsError, EstimationError, IntegrateError,
    KinematicsError, LinalgError, MappingError, MotionError, PlantError, PolynomialError,
    SignalError, SolveError, SpatialError,
};

pub mod approximation;
pub mod control;
pub mod discretization;
pub mod dynamics;
pub mod error;
pub mod estimation;
pub mod gaussian_tables;
pub mod kinematics;
pub mod linear_algebra;
pub mod mapping;
pub mod motion;
pub mod numerical_derivative;
pub mod numerical_integration;
pub mod ode;
pub mod optimization;
pub mod plant;
pub mod polynomial;
pub mod prelude;
pub mod random;
pub mod root_finding;
pub mod scalar;
pub mod signal_processing;
pub mod spatial;
mod utils;
pub mod vector_field;
