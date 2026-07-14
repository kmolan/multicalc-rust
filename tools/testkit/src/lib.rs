//! Shared test support: tolerance helpers, structural checkers, and the named
//! problem registry, usable from host tests and the bare-metal smoke firmware.

#![no_std]

#[cfg(test)]
extern crate std;

pub mod problems;
pub mod tol;
