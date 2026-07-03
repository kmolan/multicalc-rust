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

/// The outcome and stopping reason reported by the optimization solvers.
pub use optimization::{MinimizationReport, TerminationReason};

pub mod approximation;
pub mod gaussian_tables;
pub mod linear_algebra;
pub mod numerical_derivative;
pub mod numerical_integration;
pub mod optimization;
pub mod scalar;
pub mod utils;
pub mod vector_field;
