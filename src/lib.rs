#![allow(clippy::needless_return)]
#![allow(clippy::type_complexity)]

#![cfg_attr(not(feature = "std"), no_std)]


pub mod utils;
pub mod numerical_integration;
pub mod numerical_derivative;
pub mod approximation;

#[cfg(feature = "std")]
pub mod vector_field;