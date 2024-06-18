#![allow(clippy::needless_return)]
#![allow(clippy::type_complexity)]

#![no_std]

#[cfg(feature = "vectors")]
extern crate std;

pub mod core;
pub mod utils;

#[cfg(feature = "vectors")]
pub mod vec;