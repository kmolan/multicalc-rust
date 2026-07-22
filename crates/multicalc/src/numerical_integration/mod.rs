//! Numerical integration.
//!
//! - [`gaussian_integration`] — Gaussian quadrature (nodes from
//!   [`gaussian_tables`](crate::gaussian_tables)).
//! - [`iterative_integration`] — iterative refinement of a running estimate.
//! - [`integrator`] — the shared integrator traits; [`mode`] picks the method.

pub mod gaussian_integration;
pub mod integrator;
pub mod iterative_integration;
pub mod mode;

pub use crate::utils::summation::SummationMethod;
