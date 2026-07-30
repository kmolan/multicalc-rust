#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Some guide examples use the alloc-only types, so they are compiled under `alloc` only.
#[cfg(all(doctest, feature = "alloc"))]
#[doc = include_str!("../GUIDE.md")]
pub struct GuideExamples;

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

/// Zero-order-hold, Van Loan, and white-noise discretization of continuous-time linear systems.
pub use discretization::{q_discrete_white_noise, van_loan, zoh};

/// Fixed-step RK4 and adaptive RK45 (Dormand–Prince) ODE integrators.
pub use ode::{Rk4, Rk45};

/// Filters, smoothers, and signal conditioning.
pub use signal_processing::OnePoleLowPass;

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

/// Linear Kalman filter and Extended Kalman filter
pub use estimation::{CovarianceUpdate, ExtendedKalmanFilter, KalmanFilter, KalmanModel};

/// Particle filter (bootstrap/SIR) with pluggable resampling and measurement likelihood.
#[cfg(feature = "alloc")]
#[cfg_attr(docsrs, doc(cfg(feature = "alloc")))]
pub use estimation::{GaussianLikelihood, Likelihood, ParticleFilter, ResamplingScheme};

/// Seedable pseudo-random generator and the trait its uniform and normal draws come from.
pub use random::{Pcg32, RandomSource};

/// The Levenberg-Marquardt and Gauss-Newton least-squares solvers and their result types.
pub use optimization::{GaussNewton, LevenbergMarquardt, MinimizationReport, TerminationReason};

/// Bracketed and Newton root finders for scalar equations and square systems.
pub use root_finding::{Bisection, Newton, NewtonSystem, RootReport, RootReportN, RootTermination};

/// Feedback control: PID, the pure-pursuit path-following law, and Follow-the-Gap reactive
/// avoidance.
pub use control::{Curvature, FollowTheGap, FollowTheGapOutput, Pid, pure_pursuit_curvature};

/// Waypoint paths and their arc-length, closest-point, and lookahead queries.
pub use motion::{EndOfPath, PathProjection, PolylinePath};

/// Per-module-family error enums and the umbrella they convert into.
pub use error::{
    CalcError, ControlError, DiffError, EstimationError, IntegrateError, KinematicsError,
    LinalgError, MotionError, SignalError, SolveError, SpatialError,
};

pub mod approximation;
pub mod control;
pub mod discretization;
pub mod error;
pub mod estimation;
pub mod gaussian_tables;
pub mod kinematics;
pub mod linear_algebra;
pub mod motion;
pub mod numerical_derivative;
pub mod numerical_integration;
pub mod ode;
pub mod optimization;
pub mod prelude;
pub mod random;
pub mod root_finding;
pub mod scalar;
pub mod signal_processing;
pub mod spatial;
mod utils;
pub mod vector_field;
