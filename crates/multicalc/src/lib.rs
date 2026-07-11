//! `multicalc` — single- and multi-variable calculus for `no_std` Rust: derivatives,
//! integrals, Jacobians and Hessians, vector-field operators, and Taylor approximation.
//!
//! Operations are generic over the [`Numeric`] scalar trait — implemented for `f32` and `f64`,
//! defaulting to `f64` — with transcendentals from [`libm`] so it works without `std`.
//! Fallible operations return [`utils::error_codes::CalcError`].
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

/// Quaternion
pub use spatial::Quaternion;

/// The Levenberg-Marquardt and Gauss-Newton least-squares solvers and their result types.
pub use optimization::{GaussNewton, LevenbergMarquardt, MinimizationReport, TerminationReason};

/// Bracketed and Newton root finders for scalar equations and square systems.
pub use root_finding::{Bisection, Newton, NewtonSystem, RootReport, RootReportN, RootTermination};

pub mod approximation;
pub mod gaussian_tables;
pub mod linear_algebra;
pub mod numerical_derivative;
pub mod numerical_integration;
pub mod optimization;
pub mod root_finding;
pub mod scalar;
pub mod spatial;
pub mod utils;
pub mod vector_field;
