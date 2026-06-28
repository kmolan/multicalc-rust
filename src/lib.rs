//! `multicalc` — single- and multi-variable calculus for `no_std` Rust: derivatives,
//! integrals, Jacobians and Hessians, vector-field operators, and Taylor approximation.
//!
//! All math is on `f64`, with transcendentals from [`libm`] so it works without `std`.
//! Fallible operations return [`utils::error_codes::CalcError`].
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Re-export of [`libm`], so `no_std` users can reach transcendental functions
/// (`libm::sin`, `libm::exp`, …) without taking their own dependency.
pub use libm;

pub mod approximation;
pub mod gaussian_tables;
pub mod numerical_derivative;
pub mod numerical_integration;
pub mod utils;
pub mod vector_field;
