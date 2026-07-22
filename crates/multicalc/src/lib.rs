#![doc = include_str!("../README.md")]
#![no_std]

#[cfg(feature = "alloc")]
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

/// Fixed-size, stack-allocated vector and matrix types.
pub use linear_algebra::{Matrix, Vector};

/// Zero-order-hold, Van Loan, and white-noise discretization of continuous-time linear systems.
pub use discretization::{q_discrete_white_noise, van_loan, zoh};

/// Fixed-step RK4 and adaptive RK45 (Dormand–Prince) ODE integrators.
pub use ode::{Rk4, Rk45};

/// Quaternion
pub use spatial::Quaternion;

/// SO(2)/SE(2)/SO(3)/SE(3) Lie groups for 2D and 3D rotations and rigid-body transforms.
pub use spatial::{SE2, SE3, SO2, SO3};

/// Typed spatial velocity and force in the linear-first `[v; ω]` / `[force; torque]` ordering.
pub use spatial::{Twist, Wrench};

/// Differential-drive kinematics and SE(2) odometry.
pub use kinematics::{BodyArc, BodyTwist, DifferentialDrive, WheelRotations, WheelVelocities};

/// Linear Kalman filter and Extended Kalman filter
pub use estimation::{ExtendedKalmanFilter, KalmanFilter};

/// Particle filter (bootstrap/SIR) with pluggable resampling and measurement likelihood.
#[cfg(feature = "alloc")]
pub use estimation::{GaussianLikelihood, Likelihood, ParticleFilter, ResamplingScheme};

/// Seedable pseudo-random generator and the trait its uniform and normal draws come from.
pub use random::{Pcg32, RandomSource};

/// The Levenberg-Marquardt and Gauss-Newton least-squares solvers and their result types.
pub use optimization::{GaussNewton, LevenbergMarquardt, MinimizationReport, TerminationReason};

/// Bracketed and Newton root finders for scalar equations and square systems.
pub use root_finding::{Bisection, Newton, NewtonSystem, RootReport, RootReportN, RootTermination};

/// Feedback control: PID, one-pole derivative filter, the pure-pursuit path-following law, and
/// Follow-the-Gap reactive avoidance.
pub use control::{
    Curvature, FollowTheGap, FollowTheGapOutput, OnePoleLowPass, Pid, pure_pursuit_curvature,
};

/// Waypoint paths and their arc-length, closest-point, and lookahead queries.
pub use motion::{EndOfPath, PathProjection, PolylinePath};

/// Per-module-family error enums and the umbrella they convert into.
pub use error::{
    CalcError, ControlError, DiffError, EstimationError, IntegrateError, KinematicsError,
    LinalgError, MotionError, SolveError,
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
pub mod random;
pub mod root_finding;
pub mod scalar;
pub mod spatial;
pub mod utils;
pub mod vector_field;
