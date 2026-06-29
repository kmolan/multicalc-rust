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
pub use numeric::Numeric;

pub mod approximation;
pub mod gaussian_tables;
pub mod numeric;
pub mod numerical_derivative;
pub mod numerical_integration;
pub mod utils;
pub mod vector_field;
