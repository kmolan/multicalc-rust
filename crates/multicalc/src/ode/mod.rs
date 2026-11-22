//! Ordinary differential equation integrators.
//!
//! - [`Rk4`] — fixed-step classic Runge–Kutta.
//! - [`Rk45`] — adaptive Dormand–Prince 5(4) with PI step control and cubic-Hermite dense output.
//! - [`ExponentialMap`] — orientation integrator.
//!
//! `Rk4` and `Rk45` are generic over the state `Vector<N, T>`; `ExponentialMap` works on an
//! [`SO3`](crate::spatial::SO3) orientation. All three take any [`Numeric`](crate::Numeric)
//! scalar, so the same integrator runs at `f32`/`f64` or through an autodiff scalar.

mod exponential_map;
mod rk4;
mod rk45;
mod tableau;

pub use exponential_map::ExponentialMap;
pub use rk4::Rk4;
pub use rk45::{Rk45, Step};
