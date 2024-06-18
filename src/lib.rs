#![allow(clippy::needless_return)]
#![allow(clippy::type_complexity)]

#![no_std]

#[cfg(feature = "heap")]
extern crate std;

pub mod utils;
pub mod numerical_integration;
pub mod numerical_derivative;
pub mod approximation;
pub mod vector_field;