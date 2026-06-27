#![allow(clippy::type_complexity)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

pub use libm;

pub mod approximation;
pub mod gaussian_tables;
pub mod numerical_derivative;
pub mod numerical_integration;
pub mod utils;
pub mod vector_field;
